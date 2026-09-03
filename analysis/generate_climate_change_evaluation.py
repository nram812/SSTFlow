#!/usr/bin/env python3
"""Publication-oriented climate-change evaluation for SST downscaling.

This script keeps two scientifically different questions separate:

1. Perfect-framework extrapolation: does the combined historical/RCP8.5
   Flow-SR model reproduce the OFAM future-minus-historical signal where
   high-resolution truth is available?
2. Imperfect-framework signal preservation: after deployment on unseen
   ACCESS-CM2, does each high-resolution product preserve the climate-change
   signal supplied on the 32 x 32 conditioning grid?

Large NetCDF files are streamed a few days at a time.  The complete daily
1024 x 1024 products are never loaded into memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures/climate_change"
REPORT_DIR = ROOT / "reports/climate_change"

COMBINED_TEST = (
    ROOT
    / "runs/flow_sr_combined_hist_rcp85_continue_320k/evaluation/"
    "full_test_samples_ab3_pc_75step.nc"
)


@dataclass(frozen=True)
class AccessModel:
    key: str
    label: str
    run_directory: Path
    high_resolution_shape: tuple[int, int]
    filename_suffix: str = "ab3pc_75step"

    @property
    def historical(self) -> Path:
        return self.run_directory / (
            "access_cm2_converted/"
            f"historical_1980-01-01_1989-12-31_{self.filename_suffix}.nc"
        )

    @property
    def future(self) -> Path:
        return self.run_directory / (
            "access_cm2_converted/"
            f"future_2080-01-01_2089-12-31_{self.filename_suffix}.nc"
        )


ACCESS_MODELS = (
    AccessModel(
        "flow_sr_historical",
        "Flow-SR: historical-only training",
        ROOT / "runs/flow_sr",
        (512, 512),
    ),
    AccessModel(
        "flow_sr_combined",
        "Flow-SR: historical + future training (0.1 deg)",
        ROOT / "runs/flow_sr_combined_hist_rcp85_continue_320k",
        (512, 512),
    ),
    AccessModel(
        "gan_v2_historical",
        "GAN-v2: historical-only training",
        ROOT / "runs/gan_sr_v2",
        (512, 512),
        "direct_gan",
    ),
    AccessModel(
        "gan_v2_combined",
        "GAN-v2: historical + future training",
        ROOT / "runs/gan_sr_v2_hist_rcp85_continue_220k",
        (512, 512),
        "direct_gan",
    ),
    AccessModel(
        "gan_v2b_historical",
        "GAN-v2b: historical-only, image critic",
        ROOT / "runs/gan_sr_v2b_image_only_critic",
        (512, 512),
        "direct_gan",
    ),
    AccessModel(
        "gan_v2b_combined",
        "GAN-v2b: historical + future, image critic",
        ROOT / "runs/gan_sr_v2b_hist_rcp85_continue_220k",
        (512, 512),
        "direct_gan",
    ),
    AccessModel(
        "gan_v3_historical",
        "GAN-v3: historical-only, hard consistency",
        ROOT / "runs/gan_sr_v3_hard_consistency",
        (512, 512),
        "direct_gan",
    ),
    AccessModel(
        "gan_v3_combined",
        "GAN-v3: historical + future, hard consistency",
        ROOT / "runs/gan_sr_v3_hist_rcp85_continue_220k",
        (512, 512),
        "direct_gan",
    ),
    AccessModel(
        "flow_sr_noaa_5km",
        "Flow-SR: NOAA decoder fine-tune (0.05 deg)",
        ROOT / "runs/flow_sr_noaa_5km_decoder_continue_150k",
        (1024, 1024),
    ),
)

REQUESTED_ACCESS_KEYS = ("flow_sr_combined", "flow_sr_noaa_5km")

PERFECT_COMBINED_MODELS = {
    "flow_sr_combined": (
        "Flow-SR: historical + future",
        COMBINED_TEST,
    ),
    "gan_v2_combined": (
        "GAN-v2: historical + future",
        ROOT / "runs/gan_sr_v2_hist_rcp85_continue_220k/evaluation/full_test_samples.nc",
    ),
    "gan_v2b_combined": (
        "GAN-v2b: historical + future",
        ROOT / "runs/gan_sr_v2b_hist_rcp85_continue_220k/evaluation/full_test_samples.nc",
    ),
    "gan_v3_combined": (
        "GAN-v3: historical + future",
        ROOT / "runs/gan_sr_v3_hist_rcp85_continue_220k/evaluation/full_test_samples.nc",
    ),
}

SEASONS = ("DJF", "MAM", "JJA", "SON")
MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def decoded_dates(dataset: netCDF4.Dataset) -> list:
    time = dataset.variables["time"]
    return list(
        netCDF4.num2date(
            time[:],
            time.units,
            getattr(time, "calendar", "standard"),
            only_use_cftime_datetimes=True,
        )
    )


def coordinate_names(dataset: netCDF4.Dataset) -> tuple[str, str]:
    lat_name = "lat_target" if "lat_target" in dataset.variables else "lat"
    lon_name = "lon_target" if "lon_target" in dataset.variables else "lon"
    return lat_name, lon_name


def filled(values) -> np.ndarray:
    return np.ma.filled(values, np.nan).astype(np.float64, copy=False)


def native_time_chunk(variable, minimum: int) -> int:
    """Use at least one complete compressed time chunk per read."""
    chunking = variable.chunking()
    if isinstance(chunking, (list, tuple)) and chunking:
        return max(minimum, int(chunking[0]))
    return minimum


def stream_means(
    path: Path,
    variable_names: Iterable[str],
    period_by_year: dict[str, tuple[int, int]] | None = None,
    chunk: int = 4,
) -> dict:
    """Stream annual and seasonal means for one or more variables."""
    with netCDF4.Dataset(path) as dataset:
        dates = decoded_dates(dataset)
        lat_name, lon_name = coordinate_names(dataset)
        lat = np.asarray(dataset.variables[lat_name][:], dtype=np.float64)
        lon = np.asarray(dataset.variables[lon_name][:], dtype=np.float64)
        periods = period_by_year or {
            "all": (min(date.year for date in dates), max(date.year for date in dates))
        }
        output: dict = {
            "lat": lat,
            "lon": lon,
            "lat_lr": np.asarray(dataset.variables["lat_lr"][:], dtype=np.float64),
            "lon_lr": np.asarray(dataset.variables["lon_lr"][:], dtype=np.float64),
            "periods": {},
        }
        for period_name, (first_year, last_year) in periods.items():
            indices = np.asarray(
                [i for i, date in enumerate(dates) if first_year <= date.year <= last_year],
                dtype=np.int64,
            )
            if indices.size == 0:
                raise ValueError(f"{path}: no dates in {first_year}-{last_year}")
            expected = np.arange(indices[0], indices[-1] + 1)
            if not np.array_equal(indices, expected):
                raise ValueError(f"{path}: {period_name} indices are not contiguous")
            record = {
                "first_date": dates[int(indices[0])].strftime("%Y-%m-%d"),
                "last_date": dates[int(indices[-1])].strftime("%Y-%m-%d"),
                "days": int(indices.size),
                "variables": {},
            }
            for variable_name in variable_names:
                variable = dataset.variables[variable_name]
                read_chunk = native_time_chunk(variable, chunk)
                shape = variable.shape[-2:]
                total = np.zeros(shape, dtype=np.float64)
                count = np.zeros(shape, dtype=np.int64)
                season_total = {season: np.zeros(shape, dtype=np.float64) for season in SEASONS}
                season_count = {season: np.zeros(shape, dtype=np.int64) for season in SEASONS}
                for start in range(0, indices.size, read_chunk):
                    selection = indices[start : start + read_chunk]
                    values = filled(variable[selection])
                    finite = np.isfinite(values)
                    total += np.nansum(values, axis=0)
                    count += finite.sum(axis=0)
                    for local_index, source_index in enumerate(selection):
                        season = MONTH_TO_SEASON[dates[int(source_index)].month]
                        season_total[season] += np.nan_to_num(values[local_index], nan=0.0)
                        season_count[season] += finite[local_index]
                mean = total / np.maximum(count, 1)
                mean[count == 0] = np.nan
                seasonal = {}
                for season in SEASONS:
                    value = season_total[season] / np.maximum(season_count[season], 1)
                    value[season_count[season] == 0] = np.nan
                    seasonal[season] = value
                record["variables"][variable_name] = {
                    "annual": mean,
                    "seasonal": seasonal,
                }
            output["periods"][period_name] = record
        if "ocean_mask" in dataset.variables:
            output["ocean_mask"] = np.asarray(dataset.variables["ocean_mask"][:], dtype=bool)
        if "ocean_mask_lr" in dataset.variables:
            output["ocean_mask_lr"] = np.asarray(dataset.variables["ocean_mask_lr"][:], dtype=bool)
        output["attributes"] = {name: getattr(dataset, name) for name in dataset.ncattrs()}
    return output


def area_weights(lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, mask.shape[1]))
    return np.where(mask & np.isfinite(weights), weights, 0.0)


def weighted_mean(field: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(field)
    weights = area_weights(lat, valid)
    return float(np.sum(field[valid] * weights[valid]) / np.sum(weights[valid]))


def field_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    weights = area_weights(lat, valid)
    weight = weights[valid]
    pred = prediction[valid]
    truth = target[valid]
    error = pred - truth
    weight_sum = weight.sum()
    pred_mean = np.sum(weight * pred) / weight_sum
    truth_mean = np.sum(weight * truth) / weight_sum
    covariance = np.sum(weight * (pred - pred_mean) * (truth - truth_mean)) / weight_sum
    pred_variance = np.sum(weight * (pred - pred_mean) ** 2) / weight_sum
    truth_variance = np.sum(weight * (truth - truth_mean) ** 2) / weight_sum
    correlation = covariance / np.sqrt(max(pred_variance * truth_variance, 1.0e-20))
    return {
        "prediction_mean_c": float(pred_mean),
        "target_mean_c": float(truth_mean),
        "mean_bias_c": float(np.sum(weight * error) / weight_sum),
        "rmse_c": float(np.sqrt(np.sum(weight * error**2) / weight_sum)),
        "mae_c": float(np.sum(weight * np.abs(error)) / weight_sum),
        "spatial_correlation": float(correlation),
        "mean_signal_ratio": float(pred_mean / truth_mean),
        "pattern_std_ratio": float(np.sqrt(pred_variance / max(truth_variance, 1.0e-20))),
        "valid_cells": int(valid.sum()),
    }


def coarsen_ocean_mean(field: np.ndarray, mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Return exact non-overlapping ocean means on the 32 x 32 condition grid."""
    height, width = field.shape
    coarse_height, coarse_width = target_shape
    if height % coarse_height or width % coarse_width:
        raise ValueError(f"cannot coarsen {field.shape} to {target_shape}")
    fy, fx = height // coarse_height, width // coarse_width
    field_blocks = field.reshape(coarse_height, fy, coarse_width, fx)
    mask_blocks = mask.reshape(coarse_height, fy, coarse_width, fx)
    valid = mask_blocks & np.isfinite(field_blocks)
    numerator = np.where(valid, field_blocks, 0.0).sum(axis=(1, 3))
    denominator = valid.sum(axis=(1, 3))
    result = numerator / np.maximum(denominator, 1)
    result[denominator == 0] = np.nan
    return result


def expand_coarse(field: np.ndarray, high_shape: tuple[int, int]) -> np.ndarray:
    fy = high_shape[0] // field.shape[0]
    fx = high_shape[1] // field.shape[1]
    if field.shape[0] * fy != high_shape[0] or field.shape[1] * fx != high_shape[1]:
        raise ValueError(f"cannot expand {field.shape} to {high_shape}")
    return np.repeat(np.repeat(field, fy, axis=0), fx, axis=1)


def stream_period_skill(path: Path, years: tuple[int, int], chunk: int = 4) -> dict[str, float]:
    """Stream spatiotemporal skill for generated versus target SST."""
    with netCDF4.Dataset(path) as dataset:
        dates = decoded_dates(dataset)
        indices = np.asarray(
            [i for i, date in enumerate(dates) if years[0] <= date.year <= years[1]],
            dtype=np.int64,
        )
        generated = dataset.variables["sst_generated"]
        target = dataset.variables["sst_target"]
        read_chunk = max(native_time_chunk(generated, chunk), native_time_chunk(target, chunk))
        sums = {name: 0.0 for name in ("e2", "abs_e", "e", "x", "y", "x2", "y2", "xy")}
        count = 0
        for start in range(0, indices.size, read_chunk):
            selection = indices[start : start + read_chunk]
            x = filled(generated[selection])
            y = filled(target[selection])
            valid = np.isfinite(x) & np.isfinite(y)
            xv, yv = x[valid], y[valid]
            error = xv - yv
            sums["e2"] += float(np.sum(error**2))
            sums["abs_e"] += float(np.sum(np.abs(error)))
            sums["e"] += float(np.sum(error))
            sums["x"] += float(np.sum(xv))
            sums["y"] += float(np.sum(yv))
            sums["x2"] += float(np.sum(xv**2))
            sums["y2"] += float(np.sum(yv**2))
            sums["xy"] += float(np.sum(xv * yv))
            count += int(valid.sum())
        x_variance = sums["x2"] - sums["x"] ** 2 / count
        y_variance = sums["y2"] - sums["y"] ** 2 / count
        covariance = sums["xy"] - sums["x"] * sums["y"] / count
        return {
            "first_year": years[0],
            "last_year": years[1],
            "days": int(indices.size),
            "ocean_values": count,
            "rmse_c": float(np.sqrt(sums["e2"] / count)),
            "mae_c": float(sums["abs_e"] / count),
            "bias_c": float(sums["e"] / count),
            "spatiotemporal_correlation": float(
                covariance / np.sqrt(max(x_variance * y_variance, 1.0e-20))
            ),
        }


def perfect_framework_analysis(path: Path = COMBINED_TEST) -> tuple[dict, dict[str, np.ndarray]]:
    means = stream_means(
        path,
        ("sst_generated", "sst_target", "sst_coarse"),
        {"historical": (2011, 2014), "future": (2098, 2101)},
    )
    historical = means["periods"]["historical"]["variables"]
    future = means["periods"]["future"]["variables"]
    mask = np.isfinite(historical["sst_target"]["annual"])
    generated_signal = future["sst_generated"]["annual"] - historical["sst_generated"]["annual"]
    target_signal = future["sst_target"]["annual"] - historical["sst_target"]["annual"]
    signal_metrics = field_metrics(generated_signal, target_signal, means["lat"], mask)
    seasonal_metrics = {}
    seasonal_fields = {}
    for season in SEASONS:
        generated = (
            future["sst_generated"]["seasonal"][season]
            - historical["sst_generated"]["seasonal"][season]
        )
        target = (
            future["sst_target"]["seasonal"][season]
            - historical["sst_target"]["seasonal"][season]
        )
        seasonal_metrics[season] = field_metrics(generated, target, means["lat"], mask)
        seasonal_fields[season] = {"generated": generated, "target": target}
    report = {
        "question": "perfect-framework OFAM signal accuracy",
        "historical_skill": stream_period_skill(path, (2011, 2014)),
        "future_skill": stream_period_skill(path, (2098, 2101)),
        "climate_signal": signal_metrics,
        "seasonal_climate_signal": seasonal_metrics,
        "periods": {
            name: {
                key: value for key, value in record.items() if key != "variables"
            }
            for name, record in means["periods"].items()
        },
    }
    fields = {
        "lat": means["lat"],
        "lon": means["lon"],
        "mask": mask,
        "historical_target": historical["sst_target"]["annual"],
        "future_target": future["sst_target"]["annual"],
        "historical_generated": historical["sst_generated"]["annual"],
        "future_generated": future["sst_generated"]["annual"],
        "target_signal": target_signal,
        "generated_signal": generated_signal,
        "seasonal": seasonal_fields,
    }
    return report, fields


def access_model_analysis(model: AccessModel) -> tuple[dict, dict[str, np.ndarray]]:
    historical = stream_means(model.historical, ("sst_downscaled", "sst_coarse"))
    future = stream_means(model.future, ("sst_downscaled", "sst_coarse"))
    if not np.array_equal(historical["lat_lr"], future["lat_lr"]):
        raise ValueError(f"{model.key}: low-resolution latitude mismatch")
    if not np.array_equal(historical["lon_lr"], future["lon_lr"]):
        raise ValueError(f"{model.key}: low-resolution longitude mismatch")
    if not np.array_equal(historical["ocean_mask"], future["ocean_mask"]):
        raise ValueError(f"{model.key}: high-resolution mask mismatch")
    h_values = historical["periods"]["all"]["variables"]
    f_values = future["periods"]["all"]["variables"]
    high_signal = f_values["sst_downscaled"]["annual"] - h_values["sst_downscaled"]["annual"]
    coarse_signal = f_values["sst_coarse"]["annual"] - h_values["sst_coarse"]["annual"]
    mask = historical["ocean_mask"]
    coarse_mask = historical["ocean_mask_lr"]
    recoarsened_signal = coarsen_ocean_mean(high_signal, mask, coarse_signal.shape)
    annual_metrics = field_metrics(
        recoarsened_signal,
        coarse_signal,
        historical["lat_lr"],
        coarse_mask,
    )
    seasonal_metrics = {}
    seasonal_fields = {}
    for season in SEASONS:
        high = (
            f_values["sst_downscaled"]["seasonal"][season]
            - h_values["sst_downscaled"]["seasonal"][season]
        )
        coarse = (
            f_values["sst_coarse"]["seasonal"][season]
            - h_values["sst_coarse"]["seasonal"][season]
        )
        recoarsened = coarsen_ocean_mean(high, mask, coarse.shape)
        seasonal_metrics[season] = field_metrics(
            recoarsened, coarse, historical["lat_lr"], coarse_mask
        )
        seasonal_fields[season] = {
            "high": high,
            "coarse": coarse,
            "recoarsened": recoarsened,
        }
    report = {
        "question": "unseen ACCESS-CM2 driver-signal preservation; not high-resolution truth accuracy",
        "model": model.label,
        "historical_period": historical["periods"]["all"]["first_date"]
        + " to " + historical["periods"]["all"]["last_date"],
        "future_period": future["periods"]["all"]["first_date"]
        + " to " + future["periods"]["all"]["last_date"],
        "annual_signal_preservation": annual_metrics,
        "seasonal_signal_preservation": seasonal_metrics,
        "files": {
            "historical": str(model.historical),
            "future": str(model.future),
        },
        "weights_sha256": historical["attributes"].get("weights_sha256", "not recorded"),
    }
    fields = {
        "lat": historical["lat"],
        "lon": historical["lon"],
        "lat_lr": historical["lat_lr"],
        "lon_lr": historical["lon_lr"],
        "mask": mask,
        "coarse_mask": coarse_mask,
        "high_signal": high_signal,
        "coarse_signal": coarse_signal,
        "recoarsened_signal": recoarsened_signal,
        "seasonal": seasonal_fields,
    }
    return report, fields


def masked(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, field, np.nan)


def map_panel(axis, lon, lat, field, title, cmap, vmin, vmax):
    image = axis.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=10.5, weight="bold")
    axis.set_xlabel("Longitude (degE)")
    axis.set_ylabel("Latitude (degN)")
    return image


def plot_perfect_framework(report: dict, fields: dict[str, np.ndarray]) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    mask = fields["mask"]
    climatologies = [
        fields["historical_target"], fields["future_target"], fields["target_signal"],
        fields["historical_generated"], fields["future_generated"], fields["generated_signal"],
    ]
    valid_values = np.concatenate([field[mask] for field in climatologies[:2] + climatologies[3:5]])
    temp_min, temp_max = np.nanpercentile(valid_values, [1, 99])
    signal_values = np.concatenate([fields["target_signal"][mask], fields["generated_signal"][mask]])
    sig_min, sig_max = np.nanpercentile(signal_values, [1, 99])
    titles = (
        "OFAM truth: 2011-2014", "OFAM truth: 2098-2101", "OFAM true change",
        "Combined Flow-SR: 2011-2014", "Combined Flow-SR: 2098-2101", "Flow-SR predicted change",
    )
    for index, (axis, field, title) in enumerate(zip(axes.ravel(), climatologies, titles)):
        if index % 3 == 2:
            image_signal = map_panel(
                axis, fields["lon"], fields["lat"], masked(field, mask), title,
                "RdYlBu_r", sig_min, sig_max,
            )
        else:
            image_temp = map_panel(
                axis, fields["lon"], fields["lat"], masked(field, mask), title,
                "turbo", temp_min, temp_max,
            )
    figure.colorbar(image_temp, ax=axes[:, :2], label="SST (degC)", shrink=0.82)
    figure.colorbar(image_signal, ax=axes[:, 2], label="Future - historical SST (degC)", shrink=0.82)
    metrics = report["climate_signal"]
    figure.suptitle(
        "Perfect-framework future extrapolation: combined historical/RCP8.5 Flow-SR\n"
        f"signal RMSE {metrics['rmse_c']:.3f} degC | bias {metrics['mean_bias_c']:+.3f} degC | "
        f"spatial r {metrics['spatial_correlation']:.3f} | mean ratio {metrics['mean_signal_ratio']:.3f}",
        fontsize=14, weight="bold",
    )
    path = FIGURE_DIR / "flow_sr_combined_ofam_climate_change_signal.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_perfect_signal_error(report: dict, fields: dict[str, np.ndarray]) -> Path:
    error = fields["generated_signal"] - fields["target_signal"]
    limit = float(np.nanpercentile(np.abs(error[fields["mask"]]), 99))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    image = map_panel(
        axes[0], fields["lon"], fields["lat"], masked(error, fields["mask"]),
        "Predicted minus true climate-change signal", "RdBu_r", -limit, limit,
    )
    target = [report["seasonal_climate_signal"][season]["target_mean_c"] for season in SEASONS]
    predicted = [report["seasonal_climate_signal"][season]["prediction_mean_c"] for season in SEASONS]
    x = np.arange(len(SEASONS))
    axes[1].bar(x - 0.18, target, 0.36, label="OFAM truth", color="#173f5f")
    axes[1].bar(x + 0.18, predicted, 0.36, label="Combined Flow-SR", color="#ed553b")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, SEASONS)
    axes[1].set_ylabel("Area-weighted SST change (degC)")
    axes[1].set_title("Seasonal climate-change signal", weight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.colorbar(image, ax=axes[0], label="Signal error (degC)")
    path = FIGURE_DIR / "flow_sr_combined_ofam_signal_error_and_seasons.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_access_comparison(reports: dict, fields: dict) -> Path:
    selected_models = [model for model in ACCESS_MODELS if model.key in REQUESTED_ACCESS_KEYS]
    figure, axes = plt.subplots(2, len(selected_models), figsize=(15, 9), constrained_layout=True)
    all_signal = []
    all_error = []
    for model in selected_models:
        record = fields[model.key]
        all_signal.extend([
            record["high_signal"][record["mask"]],
            record["coarse_signal"][record["coarse_mask"]],
        ])
        error_lr = record["recoarsened_signal"] - record["coarse_signal"]
        all_error.append(error_lr[record["coarse_mask"]])
    signal_min, signal_max = np.nanpercentile(np.concatenate(all_signal), [1, 99])
    error_limit = float(np.nanpercentile(np.abs(np.concatenate(all_error)), 99))
    for column, model in enumerate(selected_models):
        record = fields[model.key]
        metrics = reports[model.key]["annual_signal_preservation"]
        image_signal = map_panel(
            axes[0, column], record["lon"], record["lat"],
            masked(record["high_signal"], record["mask"]),
            model.label + f"\nmean change {metrics['prediction_mean_c']:.3f} degC",
            "RdYlBu_r", signal_min, signal_max,
        )
        error_lr = record["recoarsened_signal"] - record["coarse_signal"]
        image_error = map_panel(
            axes[1, column], record["lon_lr"], record["lat_lr"],
            masked(error_lr, record["coarse_mask"]),
            "Re-coarsened output minus ACCESS driver\n"
            f"bias {metrics['mean_bias_c']:+.3f} degC | RMSE {metrics['rmse_c']:.3f} | "
            f"ratio {metrics['mean_signal_ratio']:.3f}",
            "RdBu_r", -error_limit, error_limit,
        )
    figure.colorbar(image_signal, ax=axes[0, :], label="2080s - 1980s SST (degC)", shrink=0.83)
    figure.colorbar(image_error, ax=axes[1, :], label="Signal discrepancy at 32 x 32 (degC)", shrink=0.83)
    figure.suptitle(
        "Imperfect-framework ACCESS-CM2 climate-signal preservation\n"
        "Accuracy against high-resolution truth is not available in this experiment",
        fontsize=14, weight="bold",
    )
    path = FIGURE_DIR / "access_cm2_signal_preservation_requested_models.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_access_seasons(reports: dict) -> Path:
    figure, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    x = np.arange(len(SEASONS))
    width = 0.25
    selected_models = [model for model in ACCESS_MODELS if model.key in REQUESTED_ACCESS_KEYS]
    first = selected_models[0]
    driver = [
        reports[first.key]["seasonal_signal_preservation"][season]["target_mean_c"]
        for season in SEASONS
    ]
    axis.bar(x - width, driver, width, label="ACCESS-CM2 driver", color="#173f5f")
    colors = ("#ed553b", "#7b2cbf")
    for offset, (model, color) in enumerate(zip(selected_models, colors)):
        prediction = [
            reports[model.key]["seasonal_signal_preservation"][season]["prediction_mean_c"]
            for season in SEASONS
        ]
        axis.bar(x + offset * width, prediction, width, label=model.label, color=color)
    axis.set_xticks(x, SEASONS)
    axis.set_ylabel("Area-weighted SST change (degC)")
    axis.set_title("Seasonal preservation of the ACCESS-CM2 climate-change signal", weight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=9)
    path = FIGURE_DIR / "access_cm2_signal_preservation_seasonal.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_perfect_model_comparison(reports: dict, fields: dict) -> Path:
    """Compare combined-training flow and GAN climate signals against OFAM."""
    first_fields = fields["flow_sr_combined"]
    mask = first_fields["mask"]
    values = [first_fields["target_signal"][mask]]
    values.extend(record["generated_signal"][record["mask"]] for record in fields.values())
    lower, upper = np.nanpercentile(np.concatenate(values), [1, 99])
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    panels = [("OFAM truth", first_fields["target_signal"], None)]
    for key, (label, _) in PERFECT_COMBINED_MODELS.items():
        panels.append((label, fields[key]["generated_signal"], reports[key]["climate_signal"]))
    for axis, (label, field, metrics) in zip(axes.ravel(), panels):
        title = label
        if metrics is not None:
            title += (
                f"\nratio {metrics['mean_signal_ratio']:.3f} | "
                f"RMSE {metrics['rmse_c']:.3f} degC | r {metrics['spatial_correlation']:.3f}"
            )
        image = map_panel(
            axis, first_fields["lon"], first_fields["lat"], masked(field, mask),
            title, "RdYlBu_r", lower, upper,
        )
    axes.ravel()[-1].axis("off")
    figure.colorbar(image, ax=axes, label="2098-2101 minus 2011-2014 SST (degC)", shrink=0.82)
    figure.suptitle(
        "Perfect-framework climate-change signal: combined-training flow and GAN models",
        fontsize=14, weight="bold",
    )
    path = FIGURE_DIR / "ofam_combined_flow_gan_climate_signal_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_access_model_comparison(reports: dict, fields: dict) -> Path:
    """Compare historical-only and combined 0.1-degree models on ACCESS-CM2."""
    selected = [model for model in ACCESS_MODELS if model.key != "flow_sr_noaa_5km"]
    reference = fields[selected[0].key]
    reference_expanded = expand_coarse(reference["coarse_signal"], reference["high_signal"].shape)
    values = [reference_expanded[reference["mask"]]]
    values.extend(fields[model.key]["high_signal"][fields[model.key]["mask"]] for model in selected)
    lower, upper = np.nanpercentile(np.concatenate(values), [1, 99])
    figure, axes = plt.subplots(3, 3, figsize=(17, 15), constrained_layout=True)
    panels = [("ACCESS-CM2 32x32 driver", reference_expanded, None)]
    for model in selected:
        panels.append((model.label, fields[model.key]["high_signal"], reports[model.key]["annual_signal_preservation"]))
    for axis, (label, field, metrics) in zip(axes.ravel(), panels):
        title = label
        if metrics is not None:
            title += (
                f"\nmean ratio {metrics['mean_signal_ratio']:.3f} | "
                f"32x32 RMSE {metrics['rmse_c']:.3f} degC"
            )
        image = map_panel(
            axis, reference["lon"], reference["lat"], masked(field, reference["mask"]),
            title, "RdYlBu_r", lower, upper,
        )
    figure.colorbar(image, ax=axes, label="2080s minus 1980s SST (degC)", shrink=0.80)
    figure.suptitle(
        "ACCESS-CM2 climate signal: historical-only versus historical+future training",
        fontsize=14, weight="bold",
    )
    path = FIGURE_DIR / "access_cm2_flow_gan_training_period_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_access_signal_ratios(reports: dict) -> Path:
    selected = [model for model in ACCESS_MODELS if model.key != "flow_sr_noaa_5km"]
    labels = [model.label.replace(": ", "\n") for model in selected]
    ratios = [reports[model.key]["annual_signal_preservation"]["mean_signal_ratio"] for model in selected]
    colors = ["#3b82f6" if "historical-only" in model.label else "#16a34a" for model in selected]
    figure, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    bars = axis.bar(np.arange(len(selected)), ratios, color=colors)
    axis.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Perfect mean preservation")
    axis.set_xticks(np.arange(len(selected)), labels, rotation=22, ha="right")
    axis.set_ylabel("Mean climate-signal ratio")
    axis.set_ylim(min(0.75, min(ratios) - 0.03), max(1.04, max(ratios) + 0.03))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for bar, ratio in zip(bars, ratios):
        axis.text(bar.get_x() + bar.get_width() / 2, ratio + 0.006, f"{ratio:.3f}", ha="center", fontsize=9)
    axis.set_title(
        "ACCESS-CM2 mean warming preserved after exact re-coarsening\n"
        "Blue: historical-only training; green: historical + future training",
        weight="bold",
    )
    path = FIGURE_DIR / "access_cm2_flow_gan_signal_ratio_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_tables(report: dict) -> dict[str, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    perfect = report["perfect_framework"]
    perfect_models = report["perfect_framework_models"]
    test_rows = []
    for key, (model_label, _) in PERFECT_COMBINED_MODELS.items():
        model_report = perfect_models[key]
        for climate, label in (("historical_skill", "Historical OFAM test"), ("future_skill", "Future OFAM test")):
            metrics = model_report[climate]
            test_rows.append({
                "Model": model_label,
                "Evaluation": label,
                "Period": f"{metrics['first_year']}-{metrics['last_year']}",
                "Daily RMSE (degC)": metrics["rmse_c"],
                "Daily MAE (degC)": metrics["mae_c"],
                "Daily bias (degC)": metrics["bias_c"],
                "Spatiotemporal correlation": metrics["spatiotemporal_correlation"],
            })
    test_csv = FIGURE_DIR / "requested_models_test_period_metrics.csv"
    with test_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)

    signal_rows = []
    for key, (model_label, _) in PERFECT_COMBINED_MODELS.items():
        perfect_signal = perfect_models[key]["climate_signal"]
        signal_rows.append({
            "Framework": "Perfect OFAM",
            "Model": model_label,
            "Historical": "2011-2014",
            "Future": "2098-2101",
            "Reference": "OFAM high-resolution truth",
            "Reference mean change (degC)": perfect_signal["target_mean_c"],
            "Predicted mean change (degC)": perfect_signal["prediction_mean_c"],
            "Mean signal ratio": perfect_signal["mean_signal_ratio"],
            "Signal bias (degC)": perfect_signal["mean_bias_c"],
            "Signal RMSE (degC)": perfect_signal["rmse_c"],
            "Signal spatial correlation": perfect_signal["spatial_correlation"],
            "Pattern std ratio": perfect_signal["pattern_std_ratio"],
        })
    for model in ACCESS_MODELS:
        metrics = report["access_cm2"][model.key]["annual_signal_preservation"]
        signal_rows.append({
            "Framework": "Imperfect ACCESS-CM2",
            "Model": model.label,
            "Historical": "1980-1989",
            "Future": "2080-2089",
            "Reference": "ACCESS-CM2 32x32 driving signal",
            "Reference mean change (degC)": metrics["target_mean_c"],
            "Predicted mean change (degC)": metrics["prediction_mean_c"],
            "Mean signal ratio": metrics["mean_signal_ratio"],
            "Signal bias (degC)": metrics["mean_bias_c"],
            "Signal RMSE (degC)": metrics["rmse_c"],
            "Signal spatial correlation": metrics["spatial_correlation"],
            "Pattern std ratio": metrics["pattern_std_ratio"],
        })
    signal_csv = FIGURE_DIR / "requested_models_climate_signal_metrics.csv"
    with signal_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(signal_rows[0]))
        writer.writeheader()
        writer.writerows(signal_rows)

    seasonal_rows = []
    for season in SEASONS:
        for key, (model_label, _) in PERFECT_COMBINED_MODELS.items():
            metrics = perfect_models[key]["seasonal_climate_signal"][season]
            seasonal_rows.append({
                "Framework": "Perfect OFAM",
                "Model": model_label,
                "Season": season,
                "Reference mean change (degC)": metrics["target_mean_c"],
                "Predicted mean change (degC)": metrics["prediction_mean_c"],
                "Mean signal ratio": metrics["mean_signal_ratio"],
                "Signal bias (degC)": metrics["mean_bias_c"],
                "Signal RMSE (degC)": metrics["rmse_c"],
                "Signal spatial correlation": metrics["spatial_correlation"],
            })
        for model in ACCESS_MODELS:
            metrics = report["access_cm2"][model.key]["seasonal_signal_preservation"][season]
            seasonal_rows.append({
                "Framework": "Imperfect ACCESS-CM2",
                "Model": model.label,
                "Season": season,
                "Reference mean change (degC)": metrics["target_mean_c"],
                "Predicted mean change (degC)": metrics["prediction_mean_c"],
                "Mean signal ratio": metrics["mean_signal_ratio"],
                "Signal bias (degC)": metrics["mean_bias_c"],
                "Signal RMSE (degC)": metrics["rmse_c"],
                "Signal spatial correlation": metrics["spatial_correlation"],
            })
    seasonal_csv = FIGURE_DIR / "requested_models_seasonal_signal_metrics.csv"
    with seasonal_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seasonal_rows[0]))
        writer.writeheader()
        writer.writerows(seasonal_rows)
    return {"test_period_metrics": str(test_csv), "climate_signal_metrics": str(signal_csv), "seasonal_signal_metrics": str(seasonal_csv)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checksums", action="store_true", help="Skip expensive full-file SHA-256 checks.")
    arguments = parser.parse_args()
    required = [path for _, path in PERFECT_COMBINED_MODELS.values()]
    for model in ACCESS_MODELS:
        required.extend((model.historical, model.future))
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required products:\n" + "\n".join(map(str, missing)))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Perfect-framework OFAM historical/future evaluation", flush=True)
    perfect_reports = {}
    perfect_fields_by_model = {}
    for key, (label, path) in PERFECT_COMBINED_MODELS.items():
        print(f"  streaming {label}", flush=True)
        perfect_reports[key], perfect_fields_by_model[key] = perfect_framework_analysis(path)
    perfect_report = perfect_reports["flow_sr_combined"]
    perfect_fields = perfect_fields_by_model["flow_sr_combined"]
    access_reports = {}
    access_fields = {}
    print("[2/3] ACCESS-CM2 signal-preservation evaluation", flush=True)
    for model in ACCESS_MODELS:
        print(f"  streaming {model.label}", flush=True)
        access_reports[model.key], access_fields[model.key] = access_model_analysis(model)

    report = {
        "status": "passed",
        "interpretation": {
            "perfect_framework": "Prediction is compared with paired high-resolution OFAM truth.",
            "imperfect_framework": "Prediction is compared with the supplied ACCESS-CM2 large-scale signal after exact mask-aware re-coarsening; no high-resolution ACCESS truth exists.",
            "weighting": "All scalar map statistics use cosine-latitude ocean-area weights.",
        },
        "perfect_framework": perfect_report,
        "perfect_framework_models": perfect_reports,
        "access_cm2": access_reports,
        "source_files": {str(path): {"bytes": path.stat().st_size} for path in required},
    }
    if not arguments.skip_checksums:
        for path in required:
            report["source_files"][str(path)]["sha256"] = sha256(path)

    print("[3/3] Figures and tables", flush=True)
    figure_paths = [
        plot_perfect_framework(perfect_report, perfect_fields),
        plot_perfect_signal_error(perfect_report, perfect_fields),
        plot_access_comparison(access_reports, access_fields),
        plot_access_seasons(access_reports),
        plot_perfect_model_comparison(perfect_reports, perfect_fields_by_model),
        plot_access_model_comparison(access_reports, access_fields),
        plot_access_signal_ratios(access_reports),
    ]
    report["figures"] = [str(path) for path in figure_paths]
    report["tables"] = write_tables(report)
    report_path = REPORT_DIR / "requested_models_climate_change_evaluation.json"
    atomic_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "figures": report["figures"],
        "tables": report["tables"],
        "perfect_signal": perfect_report["climate_signal"],
        "access_signals": {
            key: value["annual_signal_preservation"] for key, value in access_reports.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
