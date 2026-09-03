"""Datasets for super-resolution and autoregressive super-resolution of SST.

Both datasets return **normalised, NaN-free** tensors:

``target``    ``(1, H, W)``   high-resolution SST, land set to 0
``condition`` ``(2, h, w)``   coarse SST (invalid cells 0) plus the coarse mask
``mask``      ``(1, H, W)``   high-resolution ocean mask, 1 over ocean
``previous``  ``(1, H, W)``   previous day's high-resolution SST (AR dataset)

Normalisation is a single global mean/standard deviation taken over every ocean
cell of every training day (computed by ``preprocess.py``).  Land is replaced by
zero *after* normalisation, i.e. by the mean of the ocean distribution, so the
network sees a smooth, finite field and the masked losses ignore those pixels
anyway.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from common import (
    LAND_FILL_VALUE,
    SOURCE_TIME_DIM,
    SOURCE_VARIABLE,
    consecutive_pair_starts,
    contiguous_runs,
    date_keys,
    mask_sha256,
    selected_indices,
)


class DerivedProduct:
    """Static grids, masks, and the coarse predictor produced by preprocessing."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"{self.path} is missing; run `pixi run preprocess` first"
            )
        with xr.open_dataset(self.path, engine="h5netcdf") as dataset:
            self.times = dataset["time"].values
            self.lat = dataset["lat"].values
            self.lon = dataset["lon"].values
            self.lat_lr = dataset["lat_lr"].values
            self.lon_lr = dataset["lon_lr"].values
            self.ocean_mask = dataset["ocean_mask"].values.astype(bool)
            self.ocean_mask_lr = dataset["ocean_mask_lr"].values.astype(bool)
            self.valid_fraction_lr = dataset["valid_fraction_lr"].values.astype(
                np.float32
            )
            self.sst_lr = dataset["sst_lr"].values.astype(np.float32)
            self.source_id = (
                dataset["source_id"].values.astype(np.int16)
                if "source_id" in dataset
                else np.zeros(len(self.times), dtype=np.int16)
            )
            self.source_index = (
                dataset["source_index"].values.astype(np.int64)
                if "source_index" in dataset
                else np.arange(len(self.times), dtype=np.int64)
            )
            self.attrs = dict(dataset.attrs)
        self.coarsen_factor = int(self.attrs["coarsen_factor"])

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.ocean_mask.shape)

    @property
    def coarse_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.ocean_mask_lr.shape)

    def verify(self, normalization: dict) -> None:
        """Fail loudly when the derived file and the statistics disagree."""
        digest = mask_sha256(self.ocean_mask, self.lat, self.lon)
        expected = normalization.get("ocean_mask_sha256")
        if expected is not None and digest != expected:
            raise ValueError(
                f"Ocean mask fingerprint mismatch: {digest} != {expected}. "
                "Re-run preprocessing so the statistics match the grid."
            )
        if int(normalization.get("coarsen_factor", self.coarsen_factor)) != (
            self.coarsen_factor
        ):
            raise ValueError(
                "Coarsening factor differs between the derived file "
                f"({self.coarsen_factor}) and the statistics "
                f"({normalization['coarsen_factor']})"
            )
        finite = np.isfinite(self.sst_lr[:, self.ocean_mask_lr])
        if not finite.all():
            raise ValueError(
                f"{int((~finite).sum())} non-finite coarse values inside the "
                "coarse ocean mask"
            )
        if len(self.source_id) != len(self.times) or len(self.source_index) != len(
            self.times
        ):
            raise ValueError("Source mapping length differs from the derived time axis")
        if np.any(self.source_id < 0) or np.any(self.source_index < 0):
            raise ValueError("Source mapping contains negative indices")


class _SourceReader:
    """Lazily opened, worker-safe reader for the raw high-resolution file.

    ``netCDF4`` handles cannot be shared across forked worker processes, so the
    handle is opened on first use inside whichever process performs the read and
    is excluded from pickling.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._handle: netCDF4.Dataset | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _open(self) -> netCDF4.Dataset:
        if self._handle is None:
            self._handle = netCDF4.Dataset(self.path, "r")
        return self._handle

    @property
    def length(self) -> int:
        return int(self._open().dimensions[SOURCE_TIME_DIM].size)

    def read(self, indices) -> np.ndarray:
        """Return ``(n, lat, lon)`` float32 with NaN over land."""
        handle = self._open()
        variable = handle.variables[SOURCE_VARIABLE]
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        blocks = []
        for start, stop, _ in contiguous_runs(indices):
            block = variable[start:stop]
            if block.ndim == 4:
                block = block[:, 0]
            blocks.append(np.ma.filled(block.astype(np.float32), np.nan))
        return np.concatenate(blocks, axis=0)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class _MultiSourceReader:
    """Map a derived concatenated time axis onto multiple immutable raw files."""

    def __init__(self, paths, source_id: np.ndarray, source_index: np.ndarray):
        self.readers = [_SourceReader(path) for path in paths]
        self.source_id = np.asarray(source_id, dtype=np.int16)
        self.source_index = np.asarray(source_index, dtype=np.int64)
        if not self.readers:
            raise ValueError("At least one source file is required")
        if self.source_id.max(initial=0) >= len(self.readers):
            raise ValueError("Derived source_id refers to an unconfigured source file")

    def read(self, indices) -> np.ndarray:
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        if np.any(indices < 0) or np.any(indices >= len(self.source_id)):
            raise IndexError("Derived source index is out of bounds")
        result = None
        ids = self.source_id[indices]
        local = self.source_index[indices]
        for source in np.unique(ids):
            positions = np.flatnonzero(ids == source)
            block = self.readers[int(source)].read(local[positions])
            if result is None:
                result = np.empty((len(indices), *block.shape[1:]), dtype=np.float32)
            result[positions] = block
        if result is None:
            raise ValueError("Cannot read an empty index collection")
        return result

    def close(self) -> None:
        for reader in self.readers:
            reader.close()


class SuperResolutionDataset(Dataset):
    """One sample per day: coarse predictor plus high-resolution target."""

    def __init__(
        self,
        config: dict,
        normalization: dict,
        date_ranges: list[list[str]],
        derived: DerivedProduct | None = None,
        preload: bool = False,
    ):
        self.config = config
        self.normalization = normalization
        self.date_ranges = date_ranges
        self.derived = derived or DerivedProduct(config["derived_path"])
        self.derived.verify(normalization)

        self.mean = float(normalization["sst_mean"])
        self.std = float(normalization["sst_std"])
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"Invalid normalisation std {self.std}")

        self.times = self.derived.times
        self.indices = selected_indices(self.times, date_ranges)
        source_paths = config.get("source_paths", [config.get("source_path")])
        if any(path is None for path in source_paths):
            raise ValueError("Configure source_path or source_paths")
        self.reader = _MultiSourceReader(
            source_paths, self.derived.source_id, self.derived.source_index
        )

        self.ocean_mask = self.derived.ocean_mask
        self.ocean_mask_lr = self.derived.ocean_mask_lr
        self._mask_tensor = torch.from_numpy(
            self.ocean_mask.astype(np.float32)[None]
        )
        self._condition = self._normalize_condition(self.derived.sst_lr)

        self._cache: np.ndarray | None = None
        if preload:
            self._preload()

    # -- normalisation ----------------------------------------------------
    def _normalize_condition(self, values: np.ndarray) -> np.ndarray:
        """Standardise the coarse field and append the coarse validity mask."""
        normalized = (values - self.mean) / self.std
        normalized = np.where(
            self.ocean_mask_lr[None], normalized, LAND_FILL_VALUE
        )
        normalized = np.nan_to_num(
            normalized, nan=LAND_FILL_VALUE, posinf=0.0, neginf=0.0
        )
        mask = np.broadcast_to(
            self.ocean_mask_lr[None].astype(np.float32), normalized.shape
        )
        return np.stack((normalized, mask), axis=1).astype(np.float32)

    def _normalize_target(self, values: np.ndarray) -> np.ndarray:
        """Standardise the high-resolution field and zero the land pixels."""
        normalized = (values - self.mean) / self.std
        normalized = np.where(
            self.ocean_mask[None], normalized, LAND_FILL_VALUE
        )
        normalized = np.nan_to_num(
            normalized, nan=LAND_FILL_VALUE, posinf=0.0, neginf=0.0
        )
        return normalized[:, None].astype(np.float32)

    # -- storage ----------------------------------------------------------
    def _preload(self) -> None:
        print(
            f"[data] preloading {len(self.indices)} high-resolution days",
            flush=True,
        )
        chunk = 256
        cache = np.empty(
            (len(self.indices), 1, *self.derived.shape), dtype=np.float32
        )
        for start in range(0, len(self.indices), chunk):
            stop = min(start + chunk, len(self.indices))
            block = self.reader.read(self.indices[start:stop])
            cache[start:stop] = self._normalize_target(block)
        self._cache = cache
        self.reader.close()

    def _target(self, positions: np.ndarray) -> np.ndarray:
        if self._cache is not None:
            return self._cache[positions]
        return self._normalize_target(self.reader.read(self.indices[positions]))

    # -- Dataset protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        position = np.asarray([int(item)], dtype=np.int64)
        target = self._target(position)[0]
        condition = self._condition[self.indices[int(item)]]
        return {
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "condition": torch.from_numpy(np.ascontiguousarray(condition)),
            "mask": self._mask_tensor,
            "index": int(self.indices[int(item)]),
        }

    def dates(self, items) -> list[str]:
        items = np.atleast_1d(np.asarray(items, dtype=np.int64))
        return date_keys(self.times[self.indices[items]]).tolist()

    def close(self) -> None:
        self.reader.close()


class AutoregressiveSuperResolutionDataset(SuperResolutionDataset):
    """One sample per consecutive day pair for single-step rollout training.

    Item ``i`` yields the high-resolution state ``y(t)`` as ``previous``, the
    coarse predictor ``x(t+1)`` as ``condition``, and ``y(t+1)`` as ``target``.
    Pairs never cross a date-range boundary, so no validation day can leak into
    training through the lag channel.
    """

    def __init__(
        self,
        config: dict,
        normalization: dict,
        date_ranges: list[list[str]],
        horizon: int = 1,
        derived: DerivedProduct | None = None,
        preload: bool = False,
    ):
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        super().__init__(
            config,
            normalization,
            date_ranges,
            derived=derived,
            preload=preload,
        )
        self.starts = consecutive_pair_starts(
            self.times, date_ranges, lag=self.horizon
        )
        self.source_to_position = np.full(len(self.times), -1, dtype=np.int64)
        self.source_to_position[self.indices] = np.arange(len(self.indices))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> dict:
        start = int(self.starts[int(item)])
        source = np.arange(start, start + self.horizon + 1, dtype=np.int64)
        positions = self.source_to_position[source]
        if np.any(positions < 0):
            raise RuntimeError(
                f"Window starting at source index {start} leaves the selected "
                "date ranges"
            )
        states = self._target(positions)
        condition = self._condition[source[1:]]
        return {
            "previous": torch.from_numpy(np.ascontiguousarray(states[0])),
            "target": torch.from_numpy(np.ascontiguousarray(states[1])),
            "targets": torch.from_numpy(np.ascontiguousarray(states[1:])),
            "condition": torch.from_numpy(np.ascontiguousarray(condition[0])),
            "conditions": torch.from_numpy(np.ascontiguousarray(condition)),
            "mask": self._mask_tensor,
            "index": int(start),
        }

    def dates(self, items) -> list[str]:
        items = np.atleast_1d(np.asarray(items, dtype=np.int64))
        starts = self.starts[items]
        return date_keys(self.times[starts]).tolist()

    def date_window(self, item: int) -> list[str]:
        start = int(self.starts[int(item)])
        return date_keys(
            self.times[start : start + self.horizon + 1]
        ).tolist()


def build_dataset(
    config: dict,
    normalization: dict,
    date_ranges: list[list[str]],
    kind: str,
    derived: DerivedProduct | None = None,
    preload: bool | None = None,
    horizon: int | None = None,
):
    """Factory used by every training entrypoint."""
    if preload is None:
        preload = bool(config.get("preload", False))
    if kind == "super_resolution":
        return SuperResolutionDataset(
            config, normalization, date_ranges, derived=derived, preload=preload
        )
    if kind == "autoregressive":
        return AutoregressiveSuperResolutionDataset(
            config,
            normalization,
            date_ranges,
            horizon=int(
                horizon if horizon is not None else config.get("horizon", 1)
            ),
            derived=derived,
            preload=preload,
        )
    raise ValueError(f"Unknown dataset kind {kind!r}")
