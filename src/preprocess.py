#!/usr/bin/env python3
"""Build the derived super-resolution product from the raw OFAM NetCDF file.

The raw file (``sst_10km_OFAM_historical_Australia.nc``) is a 6.9 GB
NETCDF3_CLASSIC file holding ``temp(Time, st_ocean, yt_ocean, xt_ocean)`` as
int16 with a scale factor and an offset.  Random access is cheap (~9 ms per
day), so the *high-resolution* target is never copied: training reads it
straight from the source file.

This script produces one small derived NetCDF file containing

* ``ocean_mask``       - static ``(lat, lon)`` boolean high-resolution mask
* ``ocean_mask_lr``    - static ``(lat_lr, lon_lr)`` boolean coarse mask
* ``valid_fraction_lr``- ocean fraction inside each coarse cell
* ``sst_lr``           - ``(time, lat_lr, lon_lr)`` block-averaged predictor
* coordinates and provenance attributes

and one JSON file with the normalisation statistics computed **only** from the
training date ranges (a single global mean/standard deviation over every ocean
cell of every training day, as requested).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np

from coarsen import (
    block_valid_counts,
    coarse_coordinates,
    coarse_ocean_mask,
    coarsen,
)
from common import (
    SOURCE_LAT_DIM,
    SOURCE_LON_DIM,
    SOURCE_TIME_DIM,
    SOURCE_VARIABLE,
    atomic_json,
    date_keys,
    load_config,
    mask_sha256,
    selected_indices,
)


def open_source(path: str) -> netCDF4.Dataset:
    dataset = netCDF4.Dataset(path, "r")
    if SOURCE_VARIABLE not in dataset.variables:
        raise KeyError(
            f"{path} has no variable {SOURCE_VARIABLE!r}; "
            f"found {sorted(dataset.variables)}"
        )
    return dataset


def read_days(dataset: netCDF4.Dataset, start: int, stop: int) -> np.ndarray:
    """Read ``[start, stop)`` days as ``(n, lat, lon)`` float32 with NaN land."""
    variable = dataset.variables[SOURCE_VARIABLE]
    block = variable[start:stop]
    if block.ndim == 4:
        block = block[:, 0]
    return np.ma.filled(block.astype(np.float32), np.nan)


def source_times(dataset: netCDF4.Dataset) -> np.ndarray:
    variable = dataset.variables[SOURCE_TIME_DIM]
    return netCDF4.num2date(
        variable[:],
        units=variable.units,
        calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )


def build_masks(dataset: netCDF4.Dataset, config: dict, probe_days: int):
    """Derive the static ocean mask and verify that it never changes in time."""
    total_days = dataset.dimensions[SOURCE_TIME_DIM].size
    probe_indices = np.unique(
        np.linspace(0, total_days - 1, max(probe_days, 2)).astype(int)
    )
    reference = None
    for index in probe_indices:
        day = read_days(dataset, int(index), int(index) + 1)[0]
        mask = np.isfinite(day)
        if reference is None:
            reference = mask
        elif not np.array_equal(mask, reference):
            differing = int((mask != reference).sum())
            raise ValueError(
                f"Land/ocean mask changes at time index {index} "
                f"({differing} cells differ from index {probe_indices[0]}). "
                "The pipeline assumes a time-invariant mask."
            )
    ocean_mask = reference
    if ocean_mask is None or not ocean_mask.any():
        raise ValueError("No ocean cells found in the source file")

    factor = int(config["coarsen_factor"])
    fraction = float(config["min_valid_fraction"])
    counts = block_valid_counts(ocean_mask, factor)
    mask_lr = coarse_ocean_mask(ocean_mask, factor, fraction)
    if not mask_lr.any():
        raise ValueError(
            f"No coarse cell reaches min_valid_fraction={fraction}; "
            "lower it or reduce coarsen_factor"
        )
    valid_fraction = counts.astype(np.float32) / float(factor * factor)
    return ocean_mask, mask_lr, valid_fraction


def coarsen_all(
    dataset: netCDF4.Dataset,
    ocean_mask: np.ndarray,
    config: dict,
    chunk: int,
) -> np.ndarray:
    """Stream the whole record through the mask-aware block average."""
    total_days = dataset.dimensions[SOURCE_TIME_DIM].size
    factor = int(config["coarsen_factor"])
    fraction = float(config["min_valid_fraction"])
    height, width = ocean_mask.shape
    output = np.empty(
        (total_days, height // factor, width // factor), dtype=np.float32
    )
    for start in range(0, total_days, chunk):
        stop = min(start + chunk, total_days)
        block = read_days(dataset, start, stop)
        output[start:stop] = coarsen(
            block,
            ocean_mask,
            factor,
            min_valid_fraction=fraction,
            fill_value=np.nan,
        )
        print(f"[coarsen] {stop}/{total_days} days", flush=True)
    return output


def training_statistics(
    dataset: netCDF4.Dataset,
    ocean_mask: np.ndarray,
    indices: np.ndarray,
    chunk: int,
) -> tuple[float, float, int]:
    """Streaming global mean/std over every ocean cell of the training days."""
    total = squares = 0.0
    count = 0
    minimum, maximum = np.inf, -np.inf
    ocean_flat = ocean_mask.reshape(-1)
    for source_start, source_stop, _ in _runs(indices):
        for start in range(source_start, source_stop, chunk):
            stop = min(start + chunk, source_stop)
            block = read_days(dataset, start, stop).reshape(stop - start, -1)
            values = block[:, ocean_flat].astype(np.float64)
            if not np.isfinite(values).all():
                bad = int((~np.isfinite(values)).sum())
                raise ValueError(
                    f"{bad} non-finite ocean values near time index {start}; "
                    "the static mask does not describe this file"
                )
            total += float(values.sum())
            squares += float(np.square(values).sum())
            count += values.size
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
    if count == 0:
        raise ValueError("No finite ocean values in the training ranges")
    mean = total / count
    variance = max(squares / count - mean * mean, 1.0e-8)
    return mean, float(np.sqrt(variance)), count, minimum, maximum


def _runs(indices: np.ndarray):
    from common import contiguous_runs

    return contiguous_runs(indices)


def write_derived(
    path: Path,
    times,
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    mask_lr: np.ndarray,
    valid_fraction: np.ndarray,
    sst_lr: np.ndarray,
    config: dict,
) -> None:
    import xarray as xr

    factor = int(config["coarsen_factor"])
    lat_lr = coarse_coordinates(lat, factor)
    lon_lr = coarse_coordinates(lon, factor)
    dataset = xr.Dataset(
        data_vars={
            "sst_lr": (
                ("time", "lat_lr", "lon_lr"),
                np.where(mask_lr[None], sst_lr, np.nan).astype(np.float32),
            ),
            "ocean_mask": (("lat", "lon"), ocean_mask.astype(np.int8)),
            "ocean_mask_lr": (("lat_lr", "lon_lr"), mask_lr.astype(np.int8)),
            "valid_fraction_lr": (
                ("lat_lr", "lon_lr"),
                valid_fraction.astype(np.float32),
            ),
        },
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "lat": lat.astype(np.float64),
            "lon": lon.astype(np.float64),
            "lat_lr": lat_lr.astype(np.float64),
            "lon_lr": lon_lr.astype(np.float64),
        },
        attrs={
            "source_file": str(config["source_path"]),
            "coarsen_factor": factor,
            "min_valid_fraction": float(config["min_valid_fraction"]),
            "coarse_resolution_degrees": float(
                abs(float(lat[1] - lat[0])) * factor
            ),
            "fine_resolution_degrees": float(abs(float(lat[1] - lat[0]))),
            "ocean_cells": int(ocean_mask.sum()),
            "coarse_ocean_cells": int(mask_lr.sum()),
            "ocean_mask_sha256": mask_sha256(ocean_mask, lat, lon),
            "description": (
                "Mask-aware block-averaged low-resolution SST predictor and "
                "static land/ocean masks for super-resolution training."
            ),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.nc")
    encoding = {
        "sst_lr": {"dtype": "float32", "zlib": True, "complevel": 4},
        "ocean_mask": {"dtype": "int8", "zlib": True, "complevel": 4},
        "ocean_mask_lr": {"dtype": "int8", "zlib": True, "complevel": 4},
        "valid_fraction_lr": {"dtype": "float32", "zlib": True, "complevel": 4},
    }
    dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    temporary.replace(path)


def run(config: dict, chunk: int = 256, probe_days: int = 24) -> dict:
    source = open_source(config["source_path"])
    try:
        times = source_times(source)
        lat = np.asarray(source.variables[SOURCE_LAT_DIM][:], dtype=np.float64)
        lon = np.asarray(source.variables[SOURCE_LON_DIM][:], dtype=np.float64)
        ocean_mask, mask_lr, valid_fraction = build_masks(
            source, config, probe_days
        )
        print(
            f"[mask] ocean cells {int(ocean_mask.sum())}/{ocean_mask.size} "
            f"({ocean_mask.mean():.4f}); coarse ocean cells "
            f"{int(mask_lr.sum())}/{mask_lr.size}",
            flush=True,
        )
        sst_lr = coarsen_all(source, ocean_mask, config, chunk)

        train_indices = selected_indices(times, config["train_date_ranges"])
        mean, std, count, minimum, maximum = training_statistics(
            source, ocean_mask, train_indices, chunk
        )
    finally:
        source.close()

    derived_path = Path(config["derived_path"])
    write_derived(
        derived_path,
        times,
        lat,
        lon,
        ocean_mask,
        mask_lr,
        valid_fraction,
        sst_lr,
        config,
    )
    print(f"[write] {derived_path}", flush=True)

    lr_train = sst_lr[train_indices][:, mask_lr]
    if not np.isfinite(lr_train).all():
        raise ValueError("Non-finite coarse values inside the coarse ocean mask")

    normalization = {
        "variable": "sst",
        "units": "degrees C",
        "sst_mean": float(mean),
        "sst_std": float(std),
        "sst_min": float(minimum),
        "sst_max": float(maximum),
        "sst_lr_mean": float(lr_train.mean()),
        "sst_lr_std": float(lr_train.std()),
        "training_values": int(count),
        "training_days": int(len(train_indices)),
        "date_ranges": config["train_date_ranges"],
        "coarsen_factor": int(config["coarsen_factor"]),
        "min_valid_fraction": float(config["min_valid_fraction"]),
        "ocean_cells": int(ocean_mask.sum()),
        "coarse_ocean_cells": int(mask_lr.sum()),
        "ocean_mask_sha256": mask_sha256(ocean_mask, lat, lon),
        "grid_shape": [int(ocean_mask.shape[0]), int(ocean_mask.shape[1])],
        "coarse_grid_shape": [int(mask_lr.shape[0]), int(mask_lr.shape[1])],
        "first_date": str(date_keys(times)[0]),
        "last_date": str(date_keys(times)[-1]),
        "total_days": int(len(times)),
    }
    atomic_json(config["normalization_cache"], normalization)
    print(json.dumps(normalization, indent=2), flush=True)
    return normalization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument(
        "--probe-days",
        type=int,
        default=24,
        help="How many evenly spaced days to check for a time-varying mask",
    )
    arguments = parser.parse_args()
    run(
        load_config(arguments.config),
        chunk=arguments.chunk,
        probe_days=arguments.probe_days,
    )


if __name__ == "__main__":
    main()
