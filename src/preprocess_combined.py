#!/usr/bin/env python3
"""Build one coarse predictor and source map from contiguous OFAM experiments.

The immutable high-resolution targets remain in their original NetCDF files.
The derived product stores, for each unique day, the source-file id and local
time index needed by :class:`data.SuperResolutionDataset`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from coarsen import block_valid_counts, coarse_coordinates, coarse_ocean_mask, coarsen
from common import (
    SOURCE_LAT_DIM,
    SOURCE_LON_DIM,
    SOURCE_TIME_DIM,
    date_keys,
    load_config,
    mask_sha256,
)
from preprocess import build_masks, open_source, read_days, source_times


def _validate_sources(datasets, config: dict, probe_days: int):
    reference = datasets[0]
    lat = np.asarray(reference.variables[SOURCE_LAT_DIM][:], dtype=np.float64)
    lon = np.asarray(reference.variables[SOURCE_LON_DIM][:], dtype=np.float64)
    ocean_mask, mask_lr, valid_fraction = build_masks(reference, config, probe_days)
    for source, dataset in enumerate(datasets[1:], start=1):
        other_lat = np.asarray(dataset.variables[SOURCE_LAT_DIM][:], dtype=np.float64)
        other_lon = np.asarray(dataset.variables[SOURCE_LON_DIM][:], dtype=np.float64)
        if not np.array_equal(lat, other_lat) or not np.array_equal(lon, other_lon):
            raise ValueError(f"Source {source} grid differs from source 0")
        other_mask, other_lr, other_fraction = build_masks(dataset, config, probe_days)
        if not np.array_equal(ocean_mask, other_mask):
            raise ValueError(f"Source {source} fine ocean mask differs from source 0")
        if not np.array_equal(mask_lr, other_lr):
            raise ValueError(f"Source {source} coarse ocean mask differs from source 0")
        np.testing.assert_allclose(valid_fraction, other_fraction, rtol=0, atol=0)
    return lat, lon, ocean_mask, mask_lr, valid_fraction


def _segment_indices(times, date_range) -> np.ndarray:
    keys = date_keys(times)
    start, stop = date_range
    indices = np.flatnonzero((keys >= start) & (keys <= stop)).astype(np.int64)
    if not len(indices):
        raise ValueError(f"No source dates found in segment {start} to {stop}")
    return indices


def build_arrays(datasets, config: dict, ocean_mask: np.ndarray, chunk: int):
    ranges = config["source_date_ranges"]
    if len(ranges) != len(datasets):
        raise ValueError("source_paths and source_date_ranges must have equal length")
    factor = int(config["coarsen_factor"])
    fraction = float(config["min_valid_fraction"])
    coarse_shape = tuple(value // factor for value in ocean_mask.shape)
    times_all, ids_all, indices_all, coarse_all = [], [], [], []
    for source_id, (dataset, date_range) in enumerate(zip(datasets, ranges)):
        times = np.asarray(source_times(dataset))
        indices = _segment_indices(times, date_range)
        output = np.empty((len(indices), *coarse_shape), dtype=np.float32)
        for destination in range(0, len(indices), chunk):
            chosen = indices[destination : destination + chunk]
            # Configured segments are required to be contiguous within a source.
            if len(chosen) > 1 and not np.all(np.diff(chosen) == 1):
                raise ValueError(f"Source {source_id} segment is not contiguous")
            block = read_days(dataset, int(chosen[0]), int(chosen[-1]) + 1)
            output[destination : destination + len(chosen)] = coarsen(
                block,
                ocean_mask,
                factor,
                min_valid_fraction=fraction,
                fill_value=np.nan,
            )
        times_all.append(times[indices])
        ids_all.append(np.full(len(indices), source_id, dtype=np.int16))
        indices_all.append(indices)
        coarse_all.append(output)
        print(
            f"[combined] source={source_id} days={len(indices)} "
            f"{date_keys(times[indices])[[0, -1]].tolist()}",
            flush=True,
        )
    times = np.concatenate(times_all).astype("datetime64[ns]")
    day = times.astype("datetime64[D]")
    if len(np.unique(day)) != len(day):
        raise ValueError("Combined source segments contain duplicate dates")
    if len(day) > 1 and not np.all(np.diff(day) == np.timedelta64(1, "D")):
        raise ValueError("Combined source segments are not a gapless daily sequence")
    return (
        times,
        np.concatenate(ids_all),
        np.concatenate(indices_all),
        np.concatenate(coarse_all),
    )


def write_product(path: Path, config, lat, lon, ocean_mask, mask_lr,
                  valid_fraction, times, source_id, source_index, sst_lr):
    factor = int(config["coarsen_factor"])
    dataset = xr.Dataset(
        data_vars={
            "sst_lr": (
                ("time", "lat_lr", "lon_lr"),
                np.where(mask_lr[None], sst_lr, np.nan).astype(np.float32),
            ),
            "source_id": (("time",), source_id.astype(np.int16)),
            "source_index": (("time",), source_index.astype(np.int32)),
            "ocean_mask": (("lat", "lon"), ocean_mask.astype(np.int8)),
            "ocean_mask_lr": (("lat_lr", "lon_lr"), mask_lr.astype(np.int8)),
            "valid_fraction_lr": (
                ("lat_lr", "lon_lr"), valid_fraction.astype(np.float32)
            ),
        },
        coords={
            "time": times,
            "lat": lat,
            "lon": lon,
            "lat_lr": coarse_coordinates(lat, factor),
            "lon_lr": coarse_coordinates(lon, factor),
        },
        attrs={
            "source_files": json.dumps(config["source_paths"]),
            "source_date_ranges": json.dumps(config["source_date_ranges"]),
            "coarsen_factor": factor,
            "min_valid_fraction": float(config["min_valid_fraction"]),
            "ocean_mask_sha256": mask_sha256(ocean_mask, lat, lon),
            "description": "Concatenated mask-aware OFAM coarse SST with raw-source mapping",
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.nc")
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in dataset.data_vars
    }
    dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    os.replace(temporary, path)


def run(config: dict, chunk: int = 256, probe_days: int = 24):
    paths = config.get("source_paths")
    if not paths or len(paths) < 2:
        raise ValueError("Combined preprocessing requires at least two source_paths")
    datasets = [open_source(path) for path in paths]
    try:
        lat, lon, ocean_mask, mask_lr, valid_fraction = _validate_sources(
            datasets, config, probe_days
        )
        arrays = build_arrays(datasets, config, ocean_mask, chunk)
        write_product(
            Path(config["derived_path"]), config, lat, lon, ocean_mask, mask_lr,
            valid_fraction, *arrays,
        )
    finally:
        for dataset in datasets:
            dataset.close()
    summary = {
        "derived_path": config["derived_path"],
        "days": int(len(arrays[0])),
        "first_date": str(date_keys(arrays[0])[0]),
        "last_date": str(date_keys(arrays[0])[-1]),
        "source_counts": {
            str(source): int((arrays[1] == source).sum())
            for source in np.unique(arrays[1])
        },
        "ocean_cells": int(ocean_mask.sum()),
        "coarse_ocean_cells": int(mask_lr.sum()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--probe-days", type=int, default=24)
    args = parser.parse_args()
    run(load_config(args.config), args.chunk, args.probe_days)


if __name__ == "__main__":
    main()
