#!/usr/bin/env python3
"""Independently validate NOAA-grid test and ACCESS-CM2 inference products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from common import atomic_json, load_json
from data_noaa_5km import NOAATransferDataset, NOAATransferProduct
from infer_access_cm2 import select_time_indices
from infer_noaa_5km import _file_sha256, _settings
from train_flow_noaa_5km_v2 import configure_paths
from validate_flow_inference import scan_masked_field


def _coordinates(dataset: xr.Dataset, product: NOAATransferProduct) -> None:
    for name, wanted in (
        ("lat_target", product.target_lat),
        ("lon_target", product.target_lon),
        ("lat_lr", product.coarse_lat),
        ("lon_lr", product.coarse_lon),
    ):
        actual = np.asarray(dataset[name].values)
        if actual.shape != wanted.shape or not np.allclose(
            actual, wanted, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"{name} differs from the NOAA model grid")


def _range(stats: dict, name: str) -> None:
    if stats["minimum_c"] < -20.0 or stats["maximum_c"] > 60.0:
        raise ValueError(f"{name} has an implausible physical range: {stats}")


def _error_stats(generated, target, mask: np.ndarray, chunk_days: int = 8) -> dict:
    total = 0.0
    total_squared = 0.0
    target_total = 0.0
    generated_total = 0.0
    target_sq = 0.0
    generated_sq = 0.0
    cross = 0.0
    count = 0
    wanted = np.asarray(mask, dtype=bool)
    for start in range(0, generated.shape[0], chunk_days):
        stop = min(start + chunk_days, generated.shape[0])
        left = np.ma.filled(generated[start:stop], np.nan)[:, wanted].astype(np.float64)
        right = np.ma.filled(target[start:stop], np.nan)[:, wanted].astype(np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"non-finite ocean data in error chunk {start}:{stop}")
        error = left - right
        count += error.size
        total += float(error.sum())
        total_squared += float(np.square(error).sum())
        generated_total += float(left.sum())
        target_total += float(right.sum())
        generated_sq += float(np.square(left).sum())
        target_sq += float(np.square(right).sum())
        cross += float((left * right).sum())
    bias = total / count
    generated_mean = generated_total / count
    target_mean = target_total / count
    covariance = cross / count - generated_mean * target_mean
    generated_variance = max(generated_sq / count - generated_mean**2, 0.0)
    target_variance = max(target_sq / count - target_mean**2, 0.0)
    correlation = covariance / max((generated_variance * target_variance) ** 0.5, 1e-12)
    return {
        "bias_c": bias,
        "rmse_c": (total_squared / count) ** 0.5,
        "spatiotemporal_correlation": correlation,
        "ocean_values": count,
    }


def _load_contract(settings: dict):
    run_dir = Path(settings["run_dir"])
    config = configure_paths(load_json(run_dir / "config_used.json"))
    normalization = load_json(run_dir / "normalization.json")
    product = NOAATransferProduct(config["derived_path"])
    product.verify(normalization)
    weights = run_dir / "model_ema.pt"
    return run_dir, config, normalization, product, weights


def validate_test(settings: dict) -> dict:
    run_dir, config, normalization, product, weights = _load_contract(settings)
    truth = NOAATransferDataset(config, normalization, config["test_date_ranges"], product)
    expected_times = product.times[truth.indices].astype("datetime64[D]")
    output_path = Path(settings["output"])
    with xr.open_dataset(output_path, engine="h5netcdf") as output:
        if set(output.data_vars) != {
            "sst_generated",
            "sst_target",
            "sst_coarse",
            "ocean_mask",
            "ocean_mask_lr",
        }:
            raise ValueError(f"unexpected NOAA test variables: {set(output.data_vars)}")
        _coordinates(output, product)
        if not np.array_equal(output.time.values.astype("datetime64[D]"), expected_times):
            raise ValueError("NOAA test dates differ from the configured test split")
        if not np.array_equal(output.ocean_mask.values.astype(bool), product.target_mask):
            raise ValueError("NOAA test target mask differs from training")
        if not np.array_equal(output.ocean_mask_lr.values.astype(bool), product.coarse_mask):
            raise ValueError("NOAA test condition mask differs from training")
        if output.attrs.get("selection") != "full_test":
            raise ValueError("NOAA test product is not marked full_test")
        if output.attrs.get("sampler") != "ab3_pc" or int(
            output.attrs.get("sampler_steps", -1)
        ) != 75:
            raise ValueError("NOAA test product did not use 75-step ab3_pc")
        if int(output.attrs.get("batch_size", -1)) != 4 or "padded" not in str(
            output.attrs.get("fixed_batch_padding", "")
        ):
            raise ValueError("NOAA test product lacks the fixed batch-four contract")
        if output.attrs.get("weights_sha256") != _file_sha256(weights):
            raise ValueError("NOAA test weight checksum differs from final EMA")
        # Prove that the stored target is the raw NOAA field, not a generated or
        # coarsened surrogate, at the beginning, middle, and end of the split.
        for item in (0, len(truth) // 2, len(truth) - 1):
            expected = truth[item]["target"].numpy()[0]
            expected = expected * float(normalization["sst_std"]) + float(
                normalization["sst_mean"]
            )
            expected = np.where(product.target_mask, expected, np.nan)
            np.testing.assert_allclose(
                output.sst_target.isel(time=item).values,
                expected,
                rtol=0.0,
                atol=5.0e-5,
                equal_nan=True,
            )
    truth.close()

    with netCDF4.Dataset(output_path) as output:
        generated = scan_masked_field(output.variables["sst_generated"], product.target_mask, 4)
        target = scan_masked_field(output.variables["sst_target"], product.target_mask, 4)
        coarse = scan_masked_field(output.variables["sst_coarse"], product.coarse_mask, 64)
        metrics = _error_stats(
            output.variables["sst_generated"],
            output.variables["sst_target"],
            product.target_mask,
        )
    _range(generated, "generated NOAA test SST")
    _range(target, "target NOAA test SST")
    _range(coarse, "coarse NOAA test SST")
    metrics.update(
        {
            "selection": "full_test",
            "samples": len(expected_times),
            "date_ranges": config["test_date_ranges"],
            "sampler": "ab3_pc",
            "sampler_steps": 75,
        }
    )
    metrics_path = output_path.with_name(
        output_path.name.replace("samples", "metrics").replace(".nc", ".json")
    )
    atomic_json(metrics_path, metrics)
    report = {
        "status": "passed",
        "kind": "noaa_5km_full_test",
        "product": str(output_path.resolve()),
        "days": len(expected_times),
        "first_date": str(expected_times[0]),
        "last_date": str(expected_times[-1]),
        "weights_sha256": _file_sha256(weights),
        "generated": generated,
        "target": target,
        "coarse": coarse,
        "metrics": metrics,
    }
    atomic_json(output_path.with_suffix(".validation.json"), report)
    return report


def validate_access(settings: dict) -> dict:
    _, _, _, product, weights = _load_contract(settings)
    input_path = Path(settings["input_path"])
    variable = str(settings.get("variable", "sst_lr"))
    with xr.open_dataset(input_path, engine="h5netcdf") as source:
        indices = select_time_indices(
            source.time.values, start=settings["start"], end=settings["end"]
        )
        expected_times = source.time.values[indices].astype("datetime64[D]")
        expected_coarse = source[variable].isel(time=indices).values
    output_path = Path(settings["output"])
    with xr.open_dataset(output_path, engine="h5netcdf") as output:
        if set(output.data_vars) != {
            "sst_downscaled",
            "sst_coarse",
            "ocean_mask",
            "ocean_mask_lr",
        }:
            raise ValueError(f"unexpected NOAA ACCESS variables: {set(output.data_vars)}")
        _coordinates(output, product)
        if not np.array_equal(output.time.values.astype("datetime64[D]"), expected_times):
            raise ValueError("NOAA ACCESS dates differ from the requested period")
        if output.attrs.get("period_name") != settings["period_name"]:
            raise ValueError("NOAA ACCESS period label differs")
        if output.attrs.get("weights_sha256") != _file_sha256(weights):
            raise ValueError("NOAA ACCESS weight checksum differs from final EMA")
        if output.attrs.get("sampler") != "ab3_pc" or int(
            output.attrs.get("sampler_steps", -1)
        ) != 75:
            raise ValueError("NOAA ACCESS product did not use 75-step ab3_pc")
        if int(output.attrs.get("batch_size", -1)) != 4 or "padded" not in str(
            output.attrs.get("fixed_batch_padding", "")
        ):
            raise ValueError("NOAA ACCESS product lacks the fixed batch-four contract")
        if not np.array_equal(output.ocean_mask.values.astype(bool), product.target_mask):
            raise ValueError("NOAA ACCESS target mask differs from training")
        np.testing.assert_allclose(
            output.sst_coarse.values,
            expected_coarse,
            rtol=0.0,
            # Output is serialized as float32; this tolerance is just above
            # the observed two-ulp roundoff (1.91e-6 degC), not a scientific
            # relaxation of the boundary-field check.
            atol=2.0e-6,
            equal_nan=True,
        )
    with netCDF4.Dataset(output_path) as output:
        generated = scan_masked_field(output.variables["sst_downscaled"], product.target_mask, 4)
        coarse = scan_masked_field(output.variables["sst_coarse"], product.coarse_mask, 64)
    _range(generated, "NOAA-grid ACCESS SST")
    _range(coarse, "ACCESS condition SST")
    report = {
        "status": "passed",
        "kind": "noaa_5km_access_cm2",
        "period": settings["period_name"],
        "product": str(output_path.resolve()),
        "days": len(expected_times),
        "first_date": str(expected_times[0]),
        "last_date": str(expected_times[-1]),
        "weights_sha256": _file_sha256(weights),
        "generated": generated,
        "coarse": coarse,
    }
    atomic_json(output_path.with_suffix(".validation.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test", "access"))
    parser.add_argument("--config", type=Path, default="configs/noaa_5km_inference_150k.json")
    parser.add_argument("--period-name", choices=("historical", "future"))
    args = parser.parse_args()
    settings = _settings(args.config, args.period_name if args.mode == "access" else None)
    if args.mode == "test":
        report = validate_test(settings)
    else:
        if args.period_name is None:
            parser.error("access mode requires --period-name")
        report = validate_access(settings)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
