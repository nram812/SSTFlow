#!/usr/bin/env python3
"""Validate complete GAN test-set and converted ACCESS-CM2 products."""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np
import torch
import torch.nn.functional as F

from common import atomic_json, load_json, selected_indices
from data import DerivedProduct


def _expected_test_dates(config: dict, derived: DerivedProduct) -> np.ndarray:
    indices = selected_indices(derived.times, config["test_date_ranges"])
    return np.asarray(derived.times[indices]).astype("datetime64[D]")


def _dates(dataset: netCDF4.Dataset) -> np.ndarray:
    variable = dataset.variables["time"]
    values = netCDF4.num2date(
        variable[:], variable.units, getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return np.asarray([np.datetime64(value, "D") for value in values])


def _bilinear_baseline(
    coarse: np.ndarray,
    ocean_mask: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    valid = np.isfinite(coarse)
    normalized = np.where(valid, (coarse - mean) / std, 0.0).astype(np.float32)
    values = torch.from_numpy(normalized[:, None])
    resized = F.interpolate(
        values, size=ocean_mask.shape, mode="bilinear", align_corners=False
    ).numpy()[:, 0]
    return np.where(ocean_mask[None], resized * std + mean, np.nan)


def _coarse_error(
    generated: np.ndarray, coarse: np.ndarray, ocean_mask: np.ndarray
) -> float:
    batch, height, width = generated.shape
    coarse_height, coarse_width = coarse.shape[-2:]
    factor_y, factor_x = height // coarse_height, width // coarse_width
    valid = ocean_mask.reshape(
        coarse_height, factor_y, coarse_width, factor_x
    ).transpose(0, 2, 1, 3)
    values = np.where(ocean_mask[None], generated, 0.0).reshape(
        batch, coarse_height, factor_y, coarse_width, factor_x
    ).transpose(0, 1, 3, 2, 4)
    means = values.sum(axis=(-2, -1)) / np.maximum(valid.sum(axis=(-2, -1)), 1)
    wanted = np.isfinite(coarse)
    return float(np.max(np.abs(means[wanted] - coarse[wanted])))


def _field_checks(
    generated: np.ndarray,
    coarse: np.ndarray,
    ocean_mask: np.ndarray,
    mean: float,
    std: float,
) -> dict:
    if not np.isfinite(generated[:, ocean_mask]).all():
        raise ValueError("Generated SST contains a non-finite ocean cell")
    ocean_values = generated[:, ocean_mask]
    if float(ocean_values.min()) < -5.0 or float(ocean_values.max()) > 50.0:
        raise ValueError("Generated physical SST is outside the -5 to 50 C gate")
    baseline = _bilinear_baseline(coarse, ocean_mask, mean, std)
    residual = generated[:, ocean_mask] - baseline[:, ocean_mask]
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    if residual_rms <= 1.0e-4:
        raise ValueError("GAN output is indistinguishable from bilinear interpolation")
    return {
        "generated_ocean_min_c": float(ocean_values.min()),
        "generated_ocean_max_c": float(ocean_values.max()),
        "rms_difference_from_bilinear_c": residual_rms,
    }


def validate_test(run_dir: Path) -> dict:
    config = load_json(run_dir / "config_used.json")
    normalization = load_json(run_dir / "normalization.json")
    if config.get("model_kind") != "gan":
        raise ValueError(f"Not a GAN run: {run_dir}")
    derived = DerivedProduct(config["derived_path"])
    expected_dates = _expected_test_dates(config, derived)
    product = run_dir / "evaluation/full_test_samples.nc"
    with netCDF4.Dataset(product) as dataset:
        if not {"sst_generated", "sst_target", "sst_coarse"}.issubset(
            dataset.variables
        ):
            raise ValueError(f"Missing expected fields in {product}")
        actual_dates = _dates(dataset)
        if not np.array_equal(actual_dates, expected_dates):
            raise ValueError("Evaluation dates differ from configured test ranges")
        picks = np.unique(np.asarray([0, len(actual_dates) // 2, len(actual_dates) - 1]))
        generated = np.ma.filled(
            dataset.variables["sst_generated"][picks], np.nan
        )
        target = np.ma.filled(dataset.variables["sst_target"][picks], np.nan)
        coarse = np.ma.filled(dataset.variables["sst_coarse"][picks], np.nan)
        if not np.isfinite(target[:, derived.ocean_mask]).all():
            raise ValueError("Evaluation target contains a non-finite ocean cell")
        checks = _field_checks(
            generated, coarse, derived.ocean_mask,
            float(normalization["sst_mean"]), float(normalization["sst_std"]),
        )
        if config.get("enforce_coarse_consistency", False):
            error = _coarse_error(generated, coarse, derived.ocean_mask)
            if error > 5.0e-4:
                raise ValueError(f"Hard coarse constraint error is {error:.6g} C")
            checks["maximum_coarse_mean_error_c"] = error
    report = {
        "status": "passed",
        "product": str(product.resolve()),
        "samples": int(len(expected_dates)),
        "first_date": str(expected_dates[0]),
        "last_date": str(expected_dates[-1]),
        **checks,
    }
    atomic_json(run_dir / "evaluation/validation_report.json", report)
    return report


def validate_access(run_dir: Path, product: Path, start: str, end: str) -> dict:
    config = load_json(run_dir / "config_used.json")
    normalization = load_json(run_dir / "normalization.json")
    derived = DerivedProduct(config["derived_path"])
    expected_dates = np.arange(
        np.datetime64(start, "D"),
        np.datetime64(end, "D") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    with netCDF4.Dataset(product) as dataset:
        actual_dates = _dates(dataset)
        if not np.array_equal(actual_dates, expected_dates):
            raise ValueError("ACCESS output dates differ from the requested range")
        if int(dataset.getncattr("completed")) != len(expected_dates):
            raise ValueError("ACCESS output was not atomically completed")
        if dataset.getncattr("sampler") != "direct" or int(
            dataset.getncattr("sampler_steps")
        ) != 1:
            raise ValueError("GAN ACCESS product is not marked as direct inference")
        picks = np.unique(np.asarray([0, len(actual_dates) // 2, len(actual_dates) - 1]))
        generated = np.ma.filled(
            dataset.variables["sst_downscaled"][picks], np.nan
        )
        coarse = np.ma.filled(dataset.variables["sst_coarse"][picks], np.nan)
        checks = _field_checks(
            generated, coarse, derived.ocean_mask,
            float(normalization["sst_mean"]), float(normalization["sst_std"]),
        )
        if config.get("enforce_coarse_consistency", False):
            error = _coarse_error(generated, coarse, derived.ocean_mask)
            if error > 5.0e-4:
                raise ValueError(f"Hard coarse constraint error is {error:.6g} C")
            checks["maximum_coarse_mean_error_c"] = error
    report = {
        "status": "passed",
        "product": str(product.resolve()),
        "samples": int(len(expected_dates)),
        "first_date": str(expected_dates[0]),
        "last_date": str(expected_dates[-1]),
        **checks,
    }
    atomic_json(product.with_suffix(".validation.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--access-output", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    arguments = parser.parse_args()
    if arguments.test == (arguments.access_output is not None):
        parser.error("Choose exactly one of --test or --access-output")
    if arguments.test:
        report = validate_test(arguments.run)
    else:
        if not arguments.start or not arguments.end:
            parser.error("--access-output requires --start and --end")
        report = validate_access(
            arguments.run, arguments.access_output, arguments.start, arguments.end
        )
    print(report, flush=True)


if __name__ == "__main__":
    main()
