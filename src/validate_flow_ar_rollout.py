#!/usr/bin/env python3
"""Validate dates, masks, stability, provenance, and coarse authority of an AR run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from callbacks import write_metrics


def physical_block_means(field: np.ndarray, mask: np.ndarray, factor: int) -> np.ndarray:
    days, height, width = field.shape
    if height % factor or width % factor:
        raise ValueError("Fine grid is not divisible by the configured factor")
    values = np.nan_to_num(field, nan=0.0).reshape(
        days, height // factor, factor, width // factor, factor
    )
    weights = mask.reshape(
        height // factor, factor, width // factor, factor
    ).astype(np.float64)
    numerator = (values * weights[None]).sum(axis=(2, 4))
    denominator = weights.sum(axis=(1, 3))
    return numerator / np.maximum(denominator[None], 1.0)


def validate(
    path: Path,
    initial_date: str,
    end_date: str,
    sampler: str = "ab2_pc",
    sampler_steps: int = 75,
    factor: int = 16,
    expected_lag_conditioning: str = "full_state",
    expected_lag_guidance_scale: float = 1.0,
    expected_noise_correlation: float = 0.0,
) -> dict:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        required = {
            "sst_generated",
            "sst_target",
            "sst_coarse",
            "sst_initial_state",
            "ocean_mask",
            "coarse_ocean_mask",
        }
        if set(dataset.data_vars) != required:
            raise ValueError(f"Unexpected variables: {set(dataset.data_vars)}")
        expected_dates = np.arange(
            np.datetime64(initial_date, "D") + np.timedelta64(1, "D"),
            np.datetime64(end_date, "D") + np.timedelta64(1, "D"),
        )
        dates = dataset.time.values.astype("datetime64[D]")
        if not np.array_equal(dates, expected_dates):
            raise ValueError(
                f"Dates differ: got {dates[[0, -1]]}, expected "
                f"{expected_dates[[0, -1]]}"
            )
        if not np.array_equal(dataset.lead.values, np.arange(1, len(dates) + 1)):
            raise ValueError("Lead coordinate is not exactly 1..N")
        attrs = dict(dataset.attrs)
        expected_attrs = {
            "mode": "free_running_autoregressive",
            "initial_state_date": initial_date,
            "end_date": end_date,
            "truth_resets": 0,
            "sampler": sampler,
            "sampler_steps": sampler_steps,
            "lag_conditioning": expected_lag_conditioning,
            "lag_guidance_scale": expected_lag_guidance_scale,
            "noise_correlation": expected_noise_correlation,
            "coarse_consistency_projection": 1,
        }
        for name, wanted in expected_attrs.items():
            if attrs.get(name) != wanted:
                raise ValueError(f"Attribute {name}: {attrs.get(name)!r} != {wanted!r}")
        generated = dataset.sst_generated.values
        target = dataset.sst_target.values
        initial = dataset.sst_initial_state.values
        coarse = dataset.sst_coarse.values
        mask = dataset.ocean_mask.values.astype(bool)
        coarse_mask = dataset.coarse_ocean_mask.values.astype(bool)

    ocean = np.broadcast_to(mask[None], generated.shape)
    if not np.isfinite(generated[ocean]).all():
        raise ValueError("Generated SST contains non-finite ocean cells")
    if not np.isfinite(target[ocean]).all():
        raise ValueError("Target SST contains non-finite ocean cells")
    if not np.isfinite(initial[mask]).all():
        raise ValueError("Initial SST contains non-finite ocean cells")
    if not np.isnan(generated[~ocean]).all():
        raise ValueError("Generated land cells are not consistently NaN")
    coarse_ocean = np.broadcast_to(coarse_mask[None], coarse.shape)
    if not np.isfinite(coarse[coarse_ocean]).all():
        raise ValueError("Coarse SST contains non-finite valid-ocean cells")
    if np.nanmin(generated) < -5.0 or np.nanmax(generated) > 50.0:
        raise ValueError(
            f"Generated physical range is implausible: "
            f"[{np.nanmin(generated)}, {np.nanmax(generated)}] degC"
        )
    means = physical_block_means(generated, mask, factor)
    block_error = means - coarse
    max_block_error = float(np.nanmax(np.abs(block_error[:, coarse_mask])))
    if max_block_error > 5.0e-4:
        raise ValueError(f"Daily coarse constraint error is {max_block_error} degC")
    error = generated - target
    report = {
        "status": "passed",
        "path": str(path.resolve()),
        "days": int(len(dates)),
        "initial_state_date": initial_date,
        "first_generated_date": str(dates[0]),
        "last_generated_date": str(dates[-1]),
        "truth_resets": 0,
        "sampler": sampler,
        "sampler_steps": int(sampler_steps),
        "generated_min_c": float(np.nanmin(generated)),
        "generated_max_c": float(np.nanmax(generated)),
        "overall_rmse_c": float(np.sqrt(np.nanmean(np.square(error)))),
        "overall_bias_c": float(np.nanmean(error)),
        "max_coarse_block_error_c": max_block_error,
        "nonfinite_ocean_pixels": 0,
    }
    metrics_path = path.with_suffix(".metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    saved_metrics = json.loads(metrics_path.read_text())
    if saved_metrics.get("status") != "passed" or saved_metrics.get("truth_resets") != 0:
        raise ValueError("Companion metrics did not pass or contain truth resets")
    write_metrics(path.with_suffix(".validation.json"), report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--initial-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--sampler", default="ab2_pc")
    parser.add_argument("--sampler-steps", type=int, default=75)
    parser.add_argument("--factor", type=int, default=16)
    parser.add_argument("--expected-lag-conditioning", default="full_state")
    parser.add_argument("--expected-lag-guidance-scale", type=float, default=1.0)
    parser.add_argument("--expected-noise-correlation", type=float, default=0.0)
    arguments = parser.parse_args()
    validate(
        arguments.input,
        arguments.initial_date,
        arguments.end_date,
        arguments.sampler,
        arguments.sampler_steps,
        arguments.factor,
        arguments.expected_lag_conditioning,
        arguments.expected_lag_guidance_scale,
        arguments.expected_noise_correlation,
    )


if __name__ == "__main__":
    main()
