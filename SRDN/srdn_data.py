"""Streaming mask-aware data access for the TensorFlow SRDN experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np


SOURCE_VARIABLE = "temp"


def _date_keys(values) -> np.ndarray:
    return np.asarray([str(value)[:10] for value in values])


def _read_dates(dataset, variable_name: str) -> np.ndarray:
    variable = dataset.variables[variable_name]
    values = netCDF4.num2date(
        variable[:],
        units=variable.units,
        calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return _date_keys(values)


def _mask_sha256(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
    digest.update(np.asarray(lat, dtype=np.float64).tobytes())
    digest.update(np.asarray(lon, dtype=np.float64).tobytes())
    return digest.hexdigest()


def select_indices(dates: np.ndarray, ranges: list[list[str]]) -> np.ndarray:
    selected = np.zeros(len(dates), dtype=bool)
    for start, stop in ranges:
        selected |= (dates >= str(start)) & (dates <= str(stop))
    indices = np.flatnonzero(selected).astype(np.int64)
    if not len(indices):
        raise ValueError(f"date ranges selected no samples: {ranges}")
    return indices


class DerivedProduct:
    """Small derived product containing the coarse predictor and static masks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with netCDF4.Dataset(self.path, "r") as dataset:
            self.sst_lr = np.asarray(dataset.variables["sst_lr"][:], dtype=np.float32)
            self.ocean_mask = np.asarray(
                dataset.variables["ocean_mask"][:], dtype=bool
            )
            self.ocean_mask_lr = np.asarray(
                dataset.variables["ocean_mask_lr"][:], dtype=bool
            )
            self.valid_fraction_lr = np.asarray(
                dataset.variables["valid_fraction_lr"][:], dtype=np.float32
            )
            self.lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            self.lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
            self.lat_lr = np.asarray(dataset.variables["lat_lr"][:], dtype=np.float64)
            self.lon_lr = np.asarray(dataset.variables["lon_lr"][:], dtype=np.float64)
            self.dates = _read_dates(dataset, "time")
            self.source_mask_sha256 = str(
                getattr(dataset, "ocean_mask_sha256", "")
            )
            self.coarsen_factor = int(getattr(dataset, "coarsen_factor", 16))
        if self.sst_lr.shape[0] != len(self.dates):
            raise ValueError("derived time and sst_lr dimensions disagree")
        if self.ocean_mask.shape != tuple(self.sst_lr.shape[1:])[:0] + (
            len(self.lat),
            len(self.lon),
        ):
            raise ValueError("derived high-resolution mask shape is inconsistent")
        if self.source_mask_sha256:
            actual = _mask_sha256(self.ocean_mask, self.lat, self.lon)
            if actual != self.source_mask_sha256:
                raise ValueError(
                    f"derived mask fingerprint mismatch: {actual} != "
                    f"{self.source_mask_sha256}"
                )

    @property
    def fine_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.ocean_mask.shape)

    @property
    def coarse_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.ocean_mask_lr.shape)


class SRDNData:
    """One chronological split of the raw target and derived predictor."""

    def __init__(
        self,
        source_path: str | Path,
        derived: DerivedProduct,
        normalization_path: str | Path,
        date_ranges: list[list[str]],
    ):
        self.source_path = Path(source_path)
        self.derived = derived
        self.normalization = json.loads(Path(normalization_path).read_text())
        self.mean = float(self.normalization["sst_mean"])
        self.std = float(self.normalization["sst_std"])
        if self.std <= 0.0:
            raise ValueError("normalization standard deviation must be positive")
        self.indices = select_indices(derived.dates, date_ranges)
        self.dates = derived.dates[self.indices]
        self.ocean_mask = derived.ocean_mask
        self.ocean_mask_lr = derived.ocean_mask_lr
        self._source = None
        self._target_variable = None
        self._check_contract()

    def _check_contract(self):
        if self.derived.coarse_shape != (
            self.derived.fine_shape[0] // self.derived.coarsen_factor,
            self.derived.fine_shape[1] // self.derived.coarsen_factor,
        ):
            raise ValueError("fine/coarse shapes do not match coarsen_factor")
        expected_hash = self.normalization.get("ocean_mask_sha256")
        actual_hash = _mask_sha256(
            self.derived.ocean_mask, self.derived.lat, self.derived.lon
        )
        if expected_hash and expected_hash != actual_hash:
            raise ValueError("normalization and derived mask fingerprints disagree")

    def _open_source(self):
        if self._source is None:
            self._source = netCDF4.Dataset(self.source_path, "r")
            if SOURCE_VARIABLE not in self._source.variables:
                raise KeyError(f"source has no {SOURCE_VARIABLE!r} variable")
            self._target_variable = self._source.variables[SOURCE_VARIABLE]
            shape = self._target_variable.shape
            if tuple(shape[-2:]) != self.ocean_mask.shape:
                raise ValueError(
                    f"source target shape {shape[-2:]} != mask shape "
                    f"{self.ocean_mask.shape}"
                )

    def _read_target(self, source_index: int) -> np.ndarray:
        self._open_source()
        value = self._target_variable[int(source_index)]
        value = np.ma.filled(value, np.nan).astype(np.float32)
        while value.ndim > 2:
            value = value[0]
        normalized = (value - self.mean) / self.std
        return np.where(self.ocean_mask, normalized, 0.0).astype(np.float32)

    def batch(self, positions: np.ndarray | list[int]):
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        if np.any(positions < 0) or np.any(positions >= len(self.indices)):
            raise IndexError("batch position outside the selected split")
        source_indices = self.indices[positions]
        target = np.stack([self._read_target(int(index)) for index in source_indices])
        coarse = np.asarray(self.derived.sst_lr[source_indices], dtype=np.float32)
        coarse = (coarse - self.mean) / self.std
        coarse = np.where(self.ocean_mask_lr[None], coarse, 0.0).astype(np.float32)
        coarse_mask = np.broadcast_to(
            self.ocean_mask_lr[None, :, :, None],
            coarse.shape + (1,),
        ).astype(np.float32)
        fine_mask = np.broadcast_to(
            self.ocean_mask[None, :, :, None],
            target.shape + (1,),
        ).astype(np.float32)
        return {
            "coarse_sst": coarse[..., None],
            "coarse_mask": coarse_mask,
            "fine_mask": fine_mask,
        }, target[..., None]

    def random_positions(self, count: int, seed: int = 42) -> np.ndarray:
        count = min(int(count), len(self.indices))
        return np.random.default_rng(seed).choice(
            len(self.indices), size=count, replace=False
        ).astype(np.int64)

    def iter_epoch(self, batch_size: int, seed: int, epoch: int, shuffle=True):
        positions = np.arange(len(self.indices), dtype=np.int64)
        if shuffle:
            np.random.default_rng(int(seed) + int(epoch)).shuffle(positions)
        for start in range(0, len(positions), int(batch_size)):
            yield self.batch(positions[start : start + int(batch_size)])

    def close(self):
        if self._source is not None:
            self._source.close()
            self._source = None
            self._target_variable = None

    def __len__(self):
        return len(self.indices)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
