"""Worker-safe NOAA 0.05-degree dataset for hierarchical flow transfer."""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from common import LAND_FILL_VALUE, mask_sha256, selected_indices
from preprocess_noaa_5km import NOAA_VARIABLE, block_mean


class NOAATransferProduct:
    """Small predictor/mask product; high-resolution values remain in NOAA."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"{self.path} is missing; run preprocess_noaa_5km.py first"
            )
        with xr.open_dataset(self.path, engine="h5netcdf") as dataset:
            self.times = dataset.time.values
            self.source_index = dataset.source_index.values.astype(np.int64)
            self.sst_lr = dataset.sst_lr.values.astype(np.float32)
            self.target_mask = dataset.target_ocean_mask.values.astype(bool)
            self.base_mask = dataset.base_ocean_mask.values.astype(bool)
            self.coarse_mask = dataset.ocean_mask_lr.values.astype(bool)
            self.target_fraction_mid = dataset.target_valid_fraction_mid.values.astype(
                np.float32
            )
            self.target_fraction_lr = dataset.target_valid_fraction_lr.values.astype(
                np.float32
            )
            self.target_lat = dataset.lat_target.values.astype(np.float64)
            self.target_lon = dataset.lon_target.values.astype(np.float64)
            self.base_lat = dataset.lat.values.astype(np.float64)
            self.base_lon = dataset.lon.values.astype(np.float64)
            self.coarse_lat = dataset.lat_lr.values.astype(np.float64)
            self.coarse_lon = dataset.lon_lr.values.astype(np.float64)
            self.crop_y = slice(*json.loads(dataset.attrs["crop_y"]))
            self.crop_x = slice(*json.loads(dataset.attrs["crop_x"]))
            self.attrs = dict(dataset.attrs)

    @property
    def target_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.target_mask.shape)

    @property
    def base_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.base_mask.shape)

    @property
    def coarse_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.coarse_mask.shape)

    def verify(self, normalization: dict) -> None:
        if self.target_shape != (1024, 1024):
            raise ValueError(f"NOAA target grid must be 1024x1024, got {self.target_shape}")
        if self.base_shape != (512, 512) or self.coarse_shape != (32, 32):
            raise ValueError(
                f"Unexpected base/coarse grids {self.base_shape}/{self.coarse_shape}"
            )
        if len(self.times) != len(self.source_index) or len(self.times) != len(self.sst_lr):
            raise ValueError("NOAA time, source-index, and predictor lengths differ")
        if not np.all(np.diff(self.times.astype("datetime64[D]")) > np.timedelta64(0, "D")):
            raise ValueError("NOAA derived time axis is duplicated or unordered")
        if not np.isfinite(self.sst_lr[:, self.coarse_mask]).all():
            raise ValueError("NOAA predictor is non-finite in a fixed coarse-ocean cell")
        target_digest = mask_sha256(self.target_mask, self.target_lat, self.target_lon)
        if target_digest != normalization["target_ocean_mask_sha256"]:
            raise ValueError("NOAA target-mask fingerprint differs from normalization")
        base_digest = mask_sha256(self.base_mask, self.base_lat, self.base_lon)
        if base_digest != normalization["base_ocean_mask_sha256"]:
            raise ValueError("OFAM base-mask fingerprint differs from normalization")
        if np.any(self.target_fraction_mid[self.base_mask] <= 0):
            raise ValueError("An OFAM base-ocean cell contains no NOAA target ocean")


class _NOAAReader:
    """Open one independent NetCDF handle per DataLoader worker."""

    def __init__(self, path: str | Path, crop_y: slice, crop_x: slice):
        self.path = str(path)
        self.crop_y = crop_y
        self.crop_x = crop_x
        self._handle: netCDF4.Dataset | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _open(self) -> netCDF4.Dataset:
        if self._handle is None:
            self._handle = netCDF4.Dataset(self.path, "r")
        return self._handle

    def read(self, source_indices: np.ndarray) -> np.ndarray:
        variable = self._open().variables[NOAA_VARIABLE]
        fields = [
            np.ma.filled(
                variable[int(index), self.crop_y, self.crop_x].astype(np.float32),
                np.nan,
            )
            for index in np.atleast_1d(source_indices)
        ]
        return np.stack(fields)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class NOAATransferDataset(Dataset):
    """NOAA target, exact 0.1-degree block mean, and fixed 32x32 condition."""

    def __init__(
        self,
        config: dict,
        normalization: dict,
        date_ranges: list[list[str]],
        product: NOAATransferProduct | None = None,
    ):
        self.config = config
        self.normalization = normalization
        self.product = product or NOAATransferProduct(config["derived_path"])
        self.product.verify(normalization)
        self.indices = selected_indices(self.product.times, date_ranges)
        self.mean = float(normalization["sst_mean"])
        self.std = float(normalization["sst_std"])
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"Invalid SST standard deviation {self.std}")
        self.reader = _NOAAReader(
            config["source_path"], self.product.crop_y, self.product.crop_x
        )
        self._target_mask = torch.from_numpy(
            self.product.target_mask.astype(np.float32)[None]
        )
        self._base_mask = torch.from_numpy(
            self.product.base_mask.astype(np.float32)[None]
        )
        normalized = (self.product.sst_lr - self.mean) / self.std
        normalized = np.where(
            self.product.coarse_mask[None], normalized, LAND_FILL_VALUE
        )
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        coarse_mask = np.broadcast_to(
            self.product.coarse_mask[None].astype(np.float32), normalized.shape
        )
        self._condition = np.stack((normalized, coarse_mask), axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def _normalize_target(self, values: np.ndarray) -> np.ndarray:
        normalized = (values - self.mean) / self.std
        normalized = np.where(self.product.target_mask[None], normalized, 0.0)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

    def _base_target(self, values: np.ndarray) -> np.ndarray:
        means, _ = block_mean(values, self.product.target_mask, 2)
        normalized = (means - self.mean) / self.std
        normalized = np.where(self.product.base_mask[None], normalized, 0.0)
        if not np.isfinite(normalized[:, self.product.base_mask]).all():
            raise ValueError("NOAA 0.1-degree target is missing on OFAM ocean")
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

    def __getitem__(self, item: int) -> dict:
        derived_index = int(self.indices[int(item)])
        physical = self.reader.read(
            np.asarray([self.product.source_index[derived_index]], dtype=np.int64)
        )
        if not np.array_equal(np.isfinite(physical[0]), self.product.target_mask):
            raise ValueError(f"NOAA mask changed for derived index {derived_index}")
        return {
            "target": torch.from_numpy(self._normalize_target(physical)[0, None]),
            "base_target": torch.from_numpy(self._base_target(physical)[0, None]),
            "condition": torch.from_numpy(
                np.ascontiguousarray(self._condition[derived_index])
            ),
            "target_mask": self._target_mask,
            "base_mask": self._base_mask,
            "index": derived_index,
        }

    def dates(self, items) -> list[str]:
        values = np.atleast_1d(np.asarray(items, dtype=np.int64))
        return [str(value)[:10] for value in self.product.times[self.indices[values]]]

    def close(self) -> None:
        self.reader.close()
