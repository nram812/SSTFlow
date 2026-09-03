#!/usr/bin/env python3
"""Prepare the NOAA 0.05 degree SST transfer-learning predictor.

The 8.3 GB NOAA source is immutable.  This program writes only a compact
derived product containing the 32 x 32 predictor, source indices, coordinates,
and static masks.  The 1024 x 1024 target continues to be read from the raw
file on demand.

The crop is not selected with rounded latitude/longitude bounds.  Instead, it
is proven to be an exactly nested, two-to-one refinement of the 512 x 512 OFAM
grid: every pair of NOAA cell centres must average to its OFAM cell centre.
The coarse predictor is then a 32 x 32 mask-aware block mean and is restricted
to the *existing* OFAM coarse mask.  This preserves the pretrained model's
condition contract while allowing the NOAA target to retain its own coastline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from common import REPOSITORY_ROOT, atomic_json, date_keys, load_config, load_json, mask_sha256

NOAA_VARIABLE = "analysed_sst"
NOAA_TIME = "time"
NOAA_LAT = "lat"
NOAA_LON = "lon"


def nested_crop_slice(
    satellite: np.ndarray,
    reference: np.ndarray,
    ratio: int = 2,
    atol: float = 5.0e-5,
) -> slice:
    """Return the exact satellite slice nested inside ``reference`` cells."""
    satellite = np.asarray(satellite, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if ratio < 1 or satellite.ndim != 1 or reference.ndim != 1:
        raise ValueError("Coordinates must be one-dimensional and ratio positive")
    if len(reference) < 2 or len(satellite) < len(reference) * ratio:
        raise ValueError("Coordinate vectors are too short for a nested crop")
    satellite_step = float(np.median(np.diff(satellite)))
    reference_step = float(np.median(np.diff(reference)))
    if satellite_step <= 0 or reference_step <= 0:
        raise ValueError("Latitude and longitude coordinates must be ascending")
    if not np.isclose(reference_step, ratio * satellite_step, rtol=0, atol=atol):
        raise ValueError(
            f"Grid spacings are not nested: {reference_step} versus "
            f"{ratio} * {satellite_step}"
        )
    expected_first = reference[0] - reference_step / 2 + satellite_step / 2
    start = int(np.argmin(np.abs(satellite - expected_first)))
    stop = start + ratio * len(reference)
    if stop > len(satellite):
        raise ValueError("Nested crop extends beyond the satellite coordinate")
    cropped = satellite[start:stop]
    centres = cropped.reshape(len(reference), ratio).mean(axis=1)
    maximum_error = float(np.max(np.abs(centres - reference)))
    if maximum_error > atol:
        raise ValueError(
            f"Satellite cell centres do not nest in reference grid "
            f"(maximum centre error {maximum_error:.3g} degrees)"
        )
    return slice(start, stop)


def block_mean(
    values: np.ndarray,
    ocean_mask: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ocean-only block means and finite fractions for 2-D or 3-D arrays."""
    values = np.asarray(values, dtype=np.float32)
    single = values.ndim == 2
    if single:
        values = values[None]
    if values.ndim != 3 or ocean_mask.shape != values.shape[-2:]:
        raise ValueError("Values and ocean mask have incompatible shapes")
    batch, height, width = values.shape
    if height % factor or width % factor:
        raise ValueError(f"Grid {(height, width)} is not divisible by {factor}")
    mask = np.asarray(ocean_mask, dtype=bool)
    if np.any(np.isfinite(values) != mask[None]):
        raise ValueError("NOAA finite-value mask changed within the input block")
    reshaped = np.where(mask[None], values, 0.0).reshape(
        batch, height // factor, factor, width // factor, factor
    )
    counts = mask.reshape(
        height // factor, factor, width // factor, factor
    ).sum(axis=(1, 3))
    sums = reshaped.sum(axis=(2, 4), dtype=np.float64)
    means = np.divide(
        sums,
        counts[None],
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts[None] > 0,
    ).astype(np.float32)
    fractions = counts.astype(np.float32) / float(factor * factor)
    return (means[0] if single else means), fractions


def source_times(dataset: netCDF4.Dataset) -> np.ndarray:
    variable = dataset.variables[NOAA_TIME]
    decoded = netCDF4.num2date(
        variable[:],
        units=variable.units,
        calendar=getattr(variable, "calendar", "standard"),
    )
    return np.asarray([np.datetime64(str(value)[:10], "D") for value in decoded])


def _load_ofam_contract(path: str | Path) -> dict:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        return {
            name: dataset[name].values
            for name in (
                "lat", "lon", "lat_lr", "lon_lr", "ocean_mask", "ocean_mask_lr"
            )
        }


def _selected_source_indices(times: np.ndarray, date_range: list[str]) -> np.ndarray:
    start, stop = date_range
    keys = date_keys(times)
    indices = np.flatnonzero((keys >= start) & (keys <= stop)).astype(np.int64)
    if not len(indices):
        raise ValueError(f"No NOAA dates found in {date_range}")
    if np.any(np.diff(times[indices]) <= np.timedelta64(0, "D")):
        raise ValueError("NOAA dates are duplicated or not ordered")
    return indices


def _training_selector(times: np.ndarray, ranges: list[list[str]]) -> np.ndarray:
    keys = date_keys(times)
    selected = np.zeros(len(times), dtype=bool)
    for start, stop in ranges:
        selected |= (keys >= start) & (keys <= stop)
    if not selected.any():
        raise ValueError("No NOAA dates selected for training diagnostics")
    return selected


def _write_product(
    path: Path,
    *,
    times: np.ndarray,
    source_index: np.ndarray,
    sst_lr: np.ndarray,
    target_mask: np.ndarray,
    target_fraction_lr: np.ndarray,
    target_fraction_mid: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    contract: dict,
    crop_y: slice,
    crop_x: slice,
    config: dict,
) -> None:
    dataset = xr.Dataset(
        data_vars={
            "sst_lr": (
                ("time", "lat_lr", "lon_lr"),
                np.where(contract["ocean_mask_lr"][None], sst_lr, np.nan).astype(
                    np.float32
                ),
            ),
            "source_index": (("time",), source_index.astype(np.int32)),
            "target_ocean_mask": (
                ("lat_target", "lon_target"), target_mask.astype(np.int8)
            ),
            "base_ocean_mask": (
                ("lat", "lon"), contract["ocean_mask"].astype(np.int8)
            ),
            "ocean_mask_lr": (
                ("lat_lr", "lon_lr"), contract["ocean_mask_lr"].astype(np.int8)
            ),
            "target_valid_fraction_mid": (
                ("lat", "lon"), target_fraction_mid.astype(np.float32)
            ),
            "target_valid_fraction_lr": (
                ("lat_lr", "lon_lr"), target_fraction_lr.astype(np.float32)
            ),
        },
        coords={
            "time": times.astype("datetime64[ns]"),
            "lat_target": target_lat.astype(np.float64),
            "lon_target": target_lon.astype(np.float64),
            "lat": np.asarray(contract["lat"], dtype=np.float64),
            "lon": np.asarray(contract["lon"], dtype=np.float64),
            "lat_lr": np.asarray(contract["lat_lr"], dtype=np.float64),
            "lon_lr": np.asarray(contract["lon_lr"], dtype=np.float64),
        },
        attrs={
            "source_file": str(config["source_path"]),
            "source_date_range": json.dumps(config["source_date_range"]),
            "crop_y": json.dumps([crop_y.start, crop_y.stop]),
            "crop_x": json.dumps([crop_x.start, crop_x.stop]),
            "target_to_base_ratio": 2,
            "target_to_coarse_factor": 32,
            "normalization_policy": "fixed pretrained OFAM train statistics",
            "target_ocean_mask_sha256": mask_sha256(
                target_mask, target_lat, target_lon
            ),
            "base_ocean_mask_sha256": mask_sha256(
                contract["ocean_mask"], contract["lat"], contract["lon"]
            ),
            "description": (
                "NOAA 0.05-degree target grid and 32x32 SST predictor aligned "
                "exactly to the pretrained OFAM grid contract"
            ),
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


def run(config: dict, chunk: int = 4, full_mask_check: bool = True) -> dict:
    """Create the compact NOAA predictor and provenance report."""
    config = dict(config)
    for key in ("source_path", "ofam_derived_path", "pretrained_normalization_cache"):
        path = Path(config[key])
        config[key] = str(path if path.is_absolute() else REPOSITORY_ROOT / path)
    source = netCDF4.Dataset(config["source_path"], "r")
    try:
        if NOAA_VARIABLE not in source.variables:
            raise KeyError(f"NOAA source has no {NOAA_VARIABLE!r} variable")
        contract = _load_ofam_contract(config["ofam_derived_path"])
        source_lat = np.asarray(source.variables[NOAA_LAT][:], dtype=np.float64)
        source_lon = np.asarray(source.variables[NOAA_LON][:], dtype=np.float64)
        crop_y = nested_crop_slice(source_lat, contract["lat"], ratio=2)
        crop_x = nested_crop_slice(source_lon, contract["lon"], ratio=2)
        target_lat = source_lat[crop_y]
        target_lon = source_lon[crop_x]

        all_times = source_times(source)
        source_indices = _selected_source_indices(all_times, config["source_date_range"])
        times = all_times[source_indices]
        variable = source.variables[NOAA_VARIABLE]
        first = np.ma.filled(
            variable[int(source_indices[0]), crop_y, crop_x].astype(np.float32), np.nan
        )
        target_mask = np.isfinite(first)
        if target_mask.shape != (1024, 1024):
            raise ValueError(f"Expected measured target shape 1024x1024, got {target_mask.shape}")
        _, target_fraction_mid = block_mean(first, target_mask, 2)
        _, target_fraction_lr = block_mean(first, target_mask, 32)
        fixed_coarse_mask = np.asarray(contract["ocean_mask_lr"], dtype=bool)
        if np.any(target_fraction_lr[fixed_coarse_mask] <= 0):
            raise ValueError("A fixed OFAM coarse-ocean cell has no NOAA ocean pixels")

        output = np.empty((len(source_indices), 32, 32), dtype=np.float32)
        train_selector = _training_selector(times, config["train_date_ranges"])
        total = squares = 0.0
        count = 0
        minimum, maximum = np.inf, -np.inf
        for destination in range(0, len(source_indices), chunk):
            chosen = source_indices[destination : destination + chunk]
            fields = []
            for index in chosen:
                field = np.ma.filled(
                    variable[int(index), crop_y, crop_x].astype(np.float32), np.nan
                )
                if full_mask_check and not np.array_equal(np.isfinite(field), target_mask):
                    differing = int((np.isfinite(field) != target_mask).sum())
                    raise ValueError(
                        f"NOAA target mask changed at source index {index}: "
                        f"{differing} cells"
                    )
                fields.append(field)
            block = np.stack(fields)
            coarse, _ = block_mean(block, target_mask, 32)
            if not np.isfinite(coarse[:, fixed_coarse_mask]).all():
                raise ValueError("Coarsened NOAA predictor is missing in an OFAM ocean cell")
            output[destination : destination + len(chosen)] = coarse

            local_train = train_selector[destination : destination + len(chosen)]
            if local_train.any():
                values = block[local_train][:, target_mask].astype(np.float64)
                total += float(values.sum())
                squares += float(np.square(values).sum())
                count += int(values.size)
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
            if destination == 0 or destination + chunk >= len(source_indices) or (
                destination // chunk
            ) % 250 == 0:
                print(
                    f"[noaa-preprocess] {min(destination + chunk, len(source_indices))}/"
                    f"{len(source_indices)} days",
                    flush=True,
                )

        output[:, ~fixed_coarse_mask] = np.nan
        derived_path = Path(config["derived_path"])
        _write_product(
            derived_path,
            times=times,
            source_index=source_indices,
            sst_lr=output,
            target_mask=target_mask,
            target_fraction_lr=target_fraction_lr,
            target_fraction_mid=target_fraction_mid,
            target_lat=target_lat,
            target_lon=target_lon,
            contract=contract,
            crop_y=crop_y,
            crop_x=crop_x,
            config=config,
        )

        pretrained = load_json(config["pretrained_normalization_cache"])
        noaa_mean = total / count
        noaa_std = float(np.sqrt(max(squares / count - noaa_mean * noaa_mean, 1.0e-8)))
        expected = np.arange(times[0], times[-1] + np.timedelta64(1, "D"), dtype="datetime64[D]")
        missing_dates = date_keys(np.setdiff1d(expected, times)).tolist()
        normalization = {
            "normalization_policy": "fixed_pretrained_ofam_statistics",
            "sst_mean": float(pretrained["sst_mean"]),
            "sst_std": float(pretrained["sst_std"]),
            "pretrained_normalization_cache": str(config["pretrained_normalization_cache"]),
            "noaa_training_diagnostic_mean": noaa_mean,
            "noaa_training_diagnostic_std": noaa_std,
            "noaa_training_diagnostic_min": minimum,
            "noaa_training_diagnostic_max": maximum,
            "noaa_training_value_count": count,
            "train_date_ranges": config["train_date_ranges"],
            "target_ocean_mask_sha256": mask_sha256(target_mask, target_lat, target_lon),
            "base_ocean_mask_sha256": mask_sha256(
                contract["ocean_mask"], contract["lat"], contract["lon"]
            ),
            "coarse_ocean_cells": int(fixed_coarse_mask.sum()),
            "target_ocean_cells": int(target_mask.sum()),
            "source_missing_dates": missing_dates,
        }
        atomic_json(config["normalization_cache"], normalization)
        summary = {
            "derived_path": str(derived_path),
            "normalization_cache": str(config["normalization_cache"]),
            "first_date": str(times[0]),
            "last_date": str(times[-1]),
            "days": int(len(times)),
            "missing_dates": missing_dates,
            "crop_y": [crop_y.start, crop_y.stop],
            "crop_x": [crop_x.start, crop_x.stop],
            "target_shape": list(target_mask.shape),
            "base_shape": list(contract["ocean_mask"].shape),
            "coarse_shape": list(fixed_coarse_mask.shape),
            "target_ocean_cells": int(target_mask.sum()),
            "base_ocean_cells": int(np.asarray(contract["ocean_mask"]).sum()),
            "coarse_ocean_cells": int(fixed_coarse_mask.sum()),
            "noaa_only_target_cells": int(
                (target_mask & ~np.repeat(np.repeat(contract["ocean_mask"].astype(bool), 2, 0), 2, 1)).sum()
            ),
            "ofam_only_target_cells": int(
                (~target_mask & np.repeat(np.repeat(contract["ocean_mask"].astype(bool), 2, 0), 2, 1)).sum()
            ),
            "normalization_policy": normalization["normalization_policy"],
        }
        atomic_json(derived_path.with_suffix(".summary.json"), summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument(
        "--probe-mask-only",
        action="store_true",
        help="Skip the all-days mask assertion (intended only for synthetic tests)",
    )
    args = parser.parse_args()
    run(
        load_config(args.config),
        chunk=args.chunk,
        full_mask_check=not args.probe_mask_only,
    )


if __name__ == "__main__":
    main()
