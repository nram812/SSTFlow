#!/usr/bin/env python3
"""Validate full-test and converted ACCESS products from a plain flow run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from common import atomic_json, load_json
from data import DerivedProduct, build_dataset
from infer_access_cm2 import load_settings, select_time_indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_finite_array(values) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values)


def scan_masked_field(
    variable,
    ocean_mask: np.ndarray,
    chunk_days: int = 16,
) -> dict:
    """Stream a time-varying field and require the exact static ocean mask."""
    if variable.ndim != 3 or tuple(variable.shape[1:]) != tuple(ocean_mask.shape):
        raise ValueError(
            f"{variable.name} shape {variable.shape} does not match mask {ocean_mask.shape}"
        )
    minimum = np.inf
    maximum = -np.inf
    count = 0
    total = 0.0
    total_squared = 0.0
    wanted = np.asarray(ocean_mask, dtype=bool)
    for start in range(0, variable.shape[0], chunk_days):
        values = _as_finite_array(variable[start : start + chunk_days])
        finite = np.isfinite(values)
        expected = np.broadcast_to(wanted, values.shape)
        if not np.array_equal(finite, expected):
            raise ValueError(
                f"{variable.name} mask mismatch in days {start}:{start + len(values)}: "
                f"missing_ocean={np.count_nonzero(expected & ~finite)}, "
                f"finite_land={np.count_nonzero(~expected & finite)}"
            )
        selected = values[:, wanted].astype(np.float64)
        minimum = min(minimum, float(selected.min()))
        maximum = max(maximum, float(selected.max()))
        count += selected.size
        total += float(selected.sum(dtype=np.float64))
        total_squared += float(np.square(selected).sum(dtype=np.float64))
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    return {
        "minimum_c": minimum,
        "maximum_c": maximum,
        "mean_c": mean,
        "std_c": variance**0.5,
        "finite_ocean_values": count,
    }


def _require_coordinates(dataset: xr.Dataset, derived: DerivedProduct) -> None:
    for name, wanted in (
        ("lat", derived.lat),
        ("lon", derived.lon),
        ("lat_lr", derived.lat_lr),
        ("lon_lr", derived.lon_lr),
    ):
        if name not in dataset.coords:
            raise ValueError(f"output has no {name} coordinate")
        actual = np.asarray(dataset[name].values)
        if actual.shape != wanted.shape or not np.allclose(
            actual, wanted, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"output {name} is not the training grid")


def _require_physical_range(stats: dict, name: str) -> None:
    if stats["minimum_c"] < -20.0 or stats["maximum_c"] > 60.0:
        raise ValueError(f"{name} has implausible SST range: {stats}")


def validate_test_product(run_dir: Path, product: Path) -> dict:
    config = load_json(run_dir / "config_used.json")
    normalization = load_json(run_dir / "normalization.json")
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    dataset = build_dataset(
        config,
        normalization,
        config["test_date_ranges"],
        "super_resolution",
        derived=derived,
        preload=False,
    )
    expected_dates = np.asarray(
        dataset.dates(np.arange(len(dataset), dtype=np.int64)), dtype="datetime64[D]"
    )
    dataset.close()
    with xr.open_dataset(product, engine="h5netcdf") as output:
        if set(output.data_vars) != {"sst_generated", "sst_target", "sst_coarse"}:
            raise ValueError(f"unexpected test variables: {set(output.data_vars)}")
        _require_coordinates(output, derived)
        dates = output.time.values.astype("datetime64[D]")
        if not np.array_equal(dates, expected_dates):
            raise ValueError("test output dates do not exactly match configured splits")
        if str(output.attrs.get("selection")) != "full_test":
            raise ValueError("test output is not marked as full_test")
        if str(output.attrs.get("sampler")) != "ab3_pc":
            raise ValueError("test output did not use ab3_pc")
        if int(output.attrs.get("sampler_steps", -1)) != 75:
            raise ValueError("test output did not use 75 sampler steps")

    with netCDF4.Dataset(product) as output:
        generated = scan_masked_field(output.variables["sst_generated"], derived.ocean_mask)
        target = scan_masked_field(output.variables["sst_target"], derived.ocean_mask)
        coarse = scan_masked_field(output.variables["sst_coarse"], derived.ocean_mask_lr)
    _require_physical_range(generated, "generated test SST")
    _require_physical_range(target, "target test SST")
    _require_physical_range(coarse, "coarse test SST")

    metrics_path = product.with_name(
        product.name.replace("samples", "metrics")
    ).with_suffix(".json")
    metrics = load_json(metrics_path)
    if metrics.get("selection") != "full_test" or int(metrics.get("samples", -1)) != len(
        expected_dates
    ):
        raise ValueError("test metrics do not describe the full configured test set")
    report = {
        "status": "passed",
        "kind": "combined_flow_full_test",
        "product": str(product.resolve()),
        "days": len(expected_dates),
        "date_ranges": config["test_date_ranges"],
        "first_date": str(expected_dates[0]),
        "last_date": str(expected_dates[-1]),
        "sampler": "ab3_pc",
        "sampler_steps": 75,
        "weights_sha256": _sha256(run_dir / "model_ema.pt"),
        "generated": generated,
        "target": target,
        "coarse": coarse,
    }
    atomic_json(product.with_suffix(".validation.json"), report)
    return report


def validate_access_product(
    run_dir: Path,
    access_config: Path,
    period_name: str,
) -> dict:
    settings = load_settings(access_config, period_name)
    product = Path(settings["output"])
    config = load_json(run_dir / "config_used.json")
    normalization = load_json(run_dir / "normalization.json")
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    with xr.open_dataset(settings["input_path"], engine="h5netcdf") as source:
        indices = select_time_indices(
            source.time.values, start=settings["start"], end=settings["end"]
        )
        expected_dates = source.time.values[indices].astype("datetime64[D]")
        expected_coarse = source[str(settings.get("variable", "sst_lr"))].isel(
            time=indices
        ).values
    if len(expected_dates) != int(settings["expected_days"]):
        raise ValueError("ACCESS source period has an unexpected day count")

    with xr.open_dataset(product, engine="h5netcdf") as output:
        wanted_variables = {
            "sst_downscaled",
            "sst_coarse",
            "ocean_mask",
            "ocean_mask_lr",
        }
        if set(output.data_vars) != wanted_variables:
            raise ValueError(f"unexpected ACCESS variables: {set(output.data_vars)}")
        _require_coordinates(output, derived)
        dates = output.time.values.astype("datetime64[D]")
        if not np.array_equal(dates, expected_dates):
            raise ValueError("ACCESS output dates do not exactly match the selected period")
        if str(output.attrs.get("period_name")) != period_name:
            raise ValueError("ACCESS period_name attribute is wrong")
        if str(output.attrs.get("sampler")) != "ab3_pc" or int(
            output.attrs.get("sampler_steps", -1)
        ) != 75:
            raise ValueError("ACCESS output did not use 75-step ab3_pc")
        if not np.array_equal(output.ocean_mask.values.astype(bool), derived.ocean_mask):
            raise ValueError("ACCESS output fine mask differs from training")
        if not np.array_equal(
            output.ocean_mask_lr.values.astype(bool), derived.ocean_mask_lr
        ):
            raise ValueError("ACCESS output coarse mask differs from training")
        actual_coarse = output.sst_coarse.values
        np.testing.assert_allclose(
            actual_coarse,
            expected_coarse,
            rtol=0.0,
            # Both arrays represent the same physical boundary.  The NetCDF
            # product is explicitly float32, so permit only its final two-ulp
            # serialization roundoff rather than requiring impossible
            # bitwise agreement with xarray's decoded source values.
            atol=2.0e-6,
            equal_nan=True,
        )

    with netCDF4.Dataset(product) as output:
        generated = scan_masked_field(output.variables["sst_downscaled"], derived.ocean_mask)
        coarse = scan_masked_field(output.variables["sst_coarse"], derived.ocean_mask_lr)
    _require_physical_range(generated, f"{period_name} ACCESS downscaled SST")
    _require_physical_range(coarse, f"{period_name} ACCESS coarse SST")
    report = {
        "status": "passed",
        "kind": "combined_flow_access_cm2",
        "period": period_name,
        "product": str(product.resolve()),
        "days": len(expected_dates),
        "first_date": str(expected_dates[0]),
        "last_date": str(expected_dates[-1]),
        "sampler": "ab3_pc",
        "sampler_steps": 75,
        "weights_sha256": _sha256(run_dir / "model_ema.pt"),
        "generated": generated,
        "coarse": coarse,
    }
    atomic_json(product.with_suffix(".validation.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--test-product", type=Path)
    parser.add_argument("--access-config", type=Path)
    parser.add_argument("--period-name", choices=("historical", "future"))
    arguments = parser.parse_args()
    if arguments.test_product is not None:
        report = validate_test_product(arguments.run, arguments.test_product)
    elif arguments.access_config is not None and arguments.period_name is not None:
        report = validate_access_product(
            arguments.run, arguments.access_config, arguments.period_name
        )
    else:
        parser.error(
            "provide --test-product, or both --access-config and --period-name"
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
