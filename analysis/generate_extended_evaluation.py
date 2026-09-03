#!/usr/bin/env python3
"""Generate the AR and NOAA-transfer analyses added after the main comparison.

The original comparison notebook mixes models sharing the 0.1-degree OFAM
target.  This module deliberately keeps two scientifically different checks
separate:

* time-aligned 2011 rollouts on the OFAM grid; and
* the 0.05-degree NOAA transfer model on its own satellite target grid.

Large NetCDF products are streamed in small time chunks.  No multi-gigabyte
prediction cube is loaded into memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures/extended_evaluation"
REPORTS = ROOT / "reports/extended_evaluation"

AR_PRODUCTS = {
    "Flow-SR (no memory)": ROOT / "runs/flow_sr/evaluation/full_test_samples.nc",
    "Legacy Flow-AR": ROOT / "runs/flow_ar/evaluation/test_2011_free_rollout_ab2pc_75step.nc",
    "Residual-memory Flow-AR": ROOT / "runs/flow_ar_residual_memory/evaluation/test_2011_common_noise_free_rollout_ab2pc_75step.nc",
    "Coarse-balanced legacy Flow-AR": ROOT / "runs/flow_ar_legacy_coarse_balanced/evaluation/test_2011_common_noise_free_rollout_ab2pc_75step.nc",
}
GAN_PRODUCTS = {
    "GAN-v2 (historical)": ROOT / "runs/gan_sr_v2/evaluation/full_test_samples.nc",
    "GAN-v2b (historical)": ROOT / "runs/gan_sr_v2b_image_only_critic/evaluation/full_test_samples.nc",
    "GAN-v3 (historical)": ROOT / "runs/gan_sr_v3_hard_consistency/evaluation/full_test_samples.nc",
    "GAN-v2 (historical + future)": ROOT / "runs/gan_sr_v2_hist_rcp85_continue_220k/evaluation/full_test_samples.nc",
    "GAN-v2b (historical + future)": ROOT / "runs/gan_sr_v2b_hist_rcp85_continue_220k/evaluation/full_test_samples.nc",
    "GAN-v3 (historical + future)": ROOT / "runs/gan_sr_v3_hist_rcp85_continue_220k/evaluation/full_test_samples.nc",
}
NOAA_TEST = ROOT / "runs/flow_sr_noaa_5km_decoder_continue_150k/evaluation/full_test_samples_ab3_pc_75step.nc"
NOAA_HISTORICAL = ROOT / "runs/flow_sr_noaa_5km_decoder_continue_150k/access_cm2_converted/historical_1980-01-01_1989-12-31_ab3pc_75step.nc"
NOAA_FUTURE = ROOT / "runs/flow_sr_noaa_5km_decoder_continue_150k/access_cm2_converted/future_2080-01-01_2089-12-31_ab3pc_75step.nc"


def _dates(dataset: netCDF4.Dataset) -> np.ndarray:
    variable = dataset.variables["time"]
    values = netCDF4.num2date(
        variable[:], variable.units, getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return np.asarray([np.datetime64(value.strftime("%Y-%m-%d")) for value in values])


def _coordinates(dataset: netCDF4.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lat_name = "lat_target" if "lat_target" in dataset.variables else "lat"
    lon_name = "lon_target" if "lon_target" in dataset.variables else "lon"
    return np.asarray(dataset.variables[lat_name][:]), np.asarray(dataset.variables[lon_name][:])


def _index_for_dates(dataset: netCDF4.Dataset, wanted: np.ndarray) -> np.ndarray:
    available = _dates(dataset)
    lookup = {date: index for index, date in enumerate(available)}
    missing = [str(date) for date in wanted if date not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} requested dates are missing; first={missing[0]}")
    return np.asarray([lookup[date] for date in wanted], dtype=np.int64)


def stream_pair(
    path: Path,
    indices: np.ndarray | None = None,
    generated_name: str = "sst_generated",
    target_name: str = "sst_target",
    chunk: int = 4,
) -> dict:
    """Stream scalar, climatology, temporal, and two-point diagnostics."""
    with netCDF4.Dataset(path) as dataset:
        if generated_name not in dataset.variables or target_name not in dataset.variables:
            raise ValueError(f"{path} lacks {generated_name}/{target_name}")
        generated_variable = dataset.variables[generated_name]
        target_variable = dataset.variables[target_name]
        if indices is None:
            indices = np.arange(generated_variable.shape[0], dtype=np.int64)
        lat, lon = _coordinates(dataset)
        points = {
            "EAC": (int(np.abs(lat + 32.0).argmin()), int(np.abs(lon - 154.0).argmin())),
            "Ningaloo": (int(np.abs(lat + 22.0).argmin()), int(np.abs(lon - 113.0).argmin())),
        }
        shape = generated_variable.shape[-2:]
        sum_generated = np.zeros(shape, np.float64)
        sum_target = np.zeros(shape, np.float64)
        count = np.zeros(shape, np.int64)
        square_error = 0.0
        absolute_error = 0.0
        signed_error = 0.0
        valid_values = 0
        rmse_by_day = []
        point_series = {name: {"generated": [], "target": []} for name in points}
        generated_change = target_change = 0.0
        change_values = 0
        previous_generated = previous_target = None
        for start in range(0, len(indices), chunk):
            selection = indices[start : start + chunk]
            generated = np.ma.filled(generated_variable[selection], np.nan).astype(np.float32)
            target = np.ma.filled(target_variable[selection], np.nan).astype(np.float32)
            valid = np.isfinite(generated) & np.isfinite(target)
            error = generated - target
            square_error += float(np.nansum(error.astype(np.float64) ** 2))
            absolute_error += float(np.nansum(np.abs(error.astype(np.float64))))
            signed_error += float(np.nansum(error.astype(np.float64)))
            valid_values += int(valid.sum())
            rmse_by_day.extend(
                np.sqrt(np.nanmean(error.astype(np.float64) ** 2, axis=(1, 2))).tolist()
            )
            sum_generated += np.nansum(generated, axis=0)
            sum_target += np.nansum(target, axis=0)
            count += np.isfinite(target).sum(axis=0)
            for name, (iy, ix) in points.items():
                point_series[name]["generated"].extend(generated[:, iy, ix].tolist())
                point_series[name]["target"].extend(target[:, iy, ix].tolist())
            sequence_generated = generated
            sequence_target = target
            if previous_generated is not None:
                sequence_generated = np.concatenate((previous_generated[None], generated))
                sequence_target = np.concatenate((previous_target[None], target))
            if len(sequence_generated) > 1:
                valid_change = (
                    np.isfinite(sequence_generated[1:])
                    & np.isfinite(sequence_generated[:-1])
                    & np.isfinite(sequence_target[1:])
                    & np.isfinite(sequence_target[:-1])
                )
                generated_change += float(
                    np.abs(np.diff(sequence_generated, axis=0))[valid_change].sum()
                )
                target_change += float(
                    np.abs(np.diff(sequence_target, axis=0))[valid_change].sum()
                )
                change_values += int(valid_change.sum())
            previous_generated = generated[-1]
            previous_target = target[-1]
        generated_climatology = sum_generated / np.maximum(count, 1)
        target_climatology = sum_target / np.maximum(count, 1)
        generated_climatology[count == 0] = np.nan
        target_climatology[count == 0] = np.nan
        point_correlations = {}
        for name, series in point_series.items():
            generated = np.asarray(series["generated"])
            target = np.asarray(series["target"])
            finite = np.isfinite(generated) & np.isfinite(target)
            point_correlations[name] = float(np.corrcoef(generated[finite], target[finite])[0, 1])
        dates = _dates(dataset)[indices]
    metrics = {
        "days": int(len(indices)),
        "rmse_c": float(np.sqrt(square_error / valid_values)),
        "mae_c": float(absolute_error / valid_values),
        "bias_c": float(signed_error / valid_values),
        "climatology_rmse_c": float(
            np.sqrt(np.nanmean((generated_climatology - target_climatology) ** 2))
        ),
        "generated_mean_abs_daily_change_c": generated_change / change_values,
        "target_mean_abs_daily_change_c": target_change / change_values,
        "evolution_ratio": generated_change / max(target_change, 1.0e-12),
        "point_correlations": point_correlations,
    }
    return {
        "metrics": metrics,
        "dates": dates,
        "lat": lat,
        "lon": lon,
        "rmse_by_day": np.asarray(rmse_by_day),
        "point_series": point_series,
        "generated_climatology": generated_climatology,
        "target_climatology": target_climatology,
    }


def _stream_mean(path: Path, name: str, chunk: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netCDF4.Dataset(path) as dataset:
        variable = dataset.variables[name]
        lat, lon = _coordinates(dataset)
        total = np.zeros(variable.shape[-2:], np.float64)
        count = np.zeros(variable.shape[-2:], np.int64)
        for start in range(0, variable.shape[0], chunk):
            values = np.ma.filled(variable[start : start + chunk], np.nan)
            total += np.nansum(values, axis=0)
            count += np.isfinite(values).sum(axis=0)
    mean = total / np.maximum(count, 1)
    mean[count == 0] = np.nan
    return mean, lat, lon


def analyse_ar() -> dict:
    with netCDF4.Dataset(AR_PRODUCTS["Legacy Flow-AR"]) as reference:
        dates = _dates(reference)
    results = {}
    for name, path in AR_PRODUCTS.items():
        with netCDF4.Dataset(path) as dataset:
            indices = _index_for_dates(dataset, dates)
        results[name] = stream_pair(path, indices)

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for name, result in results.items():
        rolling = np.convolve(result["rmse_by_day"], np.ones(15) / 15, mode="same")
        axes[0].plot(result["dates"], rolling, label=name, linewidth=1.4)
    axes[0].set_ylabel("15-day running RMSE (degC)")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.25)
    point = "EAC"
    truth = np.asarray(results["Legacy Flow-AR"]["point_series"][point]["target"])
    axes[1].plot(dates, truth, color="black", linewidth=1.3, label="OFAM truth")
    for name, result in results.items():
        axes[1].plot(
            dates,
            result["point_series"][point]["generated"],
            linewidth=0.8,
            alpha=0.85,
            label=name,
        )
    axes[1].set_ylabel("EAC SST (degC)")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.25)
    figure.suptitle("2011 time-aligned autoregressive comparison")
    figure.savefig(FIGURES / "autoregressive_2011_skill_and_eac_timeseries.png", dpi=180)
    plt.close(figure)

    columns = 2
    rows_count = int(np.ceil(len(results) / columns))
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(16, 5 * rows_count), constrained_layout=True,
        squeeze=False,
    )
    limits = max(
        float(np.nanpercentile(np.abs(result["generated_climatology"] - result["target_climatology"]), 99))
        for result in results.values()
    )
    flat_axes = axes.ravel()
    for axis, (name, result) in zip(flat_axes, results.items()):
        error = result["generated_climatology"] - result["target_climatology"]
        image = axis.pcolormesh(result["lon"], result["lat"], error, cmap="RdBu_r", vmin=-limits, vmax=limits)
        axis.set_title(f"{name}\nRMSE {result['metrics']['rmse_c']:.3f} degC")
        axis.set_xlabel("Longitude (degE)")
    for axis in flat_axes[len(results):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("Latitude (degN)")
    figure.colorbar(image, ax=axes, label="mean generated - target (degC)", shrink=0.78)
    figure.savefig(FIGURES / "autoregressive_2011_mean_bias_maps.png", dpi=180)
    plt.close(figure)
    return {name: result["metrics"] for name, result in results.items()}


def analyse_gans(reference_dates: np.ndarray) -> dict:
    """Evaluate direct GAN samples on the exact 2011 AR comparison dates.

    The evolution ratio is a frame-to-frame diagnostic here, not an
    autoregressive-rollout metric: each GAN day is generated directly from that
    day's coarse SST and its saved latent-noise draw.
    """
    results = {}
    for name, path in GAN_PRODUCTS.items():
        with netCDF4.Dataset(path) as dataset:
            indices = _index_for_dates(dataset, reference_dates)
        results[name] = stream_pair(path, indices)["metrics"]
    return results


def analyse_noaa() -> dict:
    result = stream_pair(NOAA_TEST)
    bias = result["generated_climatology"] - result["target_climatology"]
    mask = np.isfinite(result["target_climatology"])
    coastal_distance = distance_transform_edt(mask)
    coastal_metrics = {}
    with netCDF4.Dataset(NOAA_TEST) as dataset:
        generated = dataset.variables["sst_generated"]
        target = dataset.variables["sst_target"]
        bands = {"coast_1px": coastal_distance <= 1, "coast_4px": coastal_distance <= 4, "interior": coastal_distance > 8}
        sums = {name: [0.0, 0] for name in bands}
        for start in range(0, generated.shape[0], 4):
            error = np.ma.filled(generated[start : start + 4], np.nan) - np.ma.filled(target[start : start + 4], np.nan)
            for name, band in bands.items():
                valid = np.isfinite(error) & band[None]
                sums[name][0] += float(np.nansum(error[valid].astype(np.float64) ** 2))
                sums[name][1] += int(valid.sum())
        coastal_metrics = {name + "_rmse_c": float(np.sqrt(total / count)) for name, (total, count) in sums.items()}

    historical, lat, lon = _stream_mean(NOAA_HISTORICAL, "sst_downscaled")
    future, future_lat, future_lon = _stream_mean(NOAA_FUTURE, "sst_downscaled")
    if not np.array_equal(lat, future_lat) or not np.array_equal(lon, future_lon):
        raise ValueError("historical/future NOAA-grid coordinates differ")
    warming = future - historical

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    lower, upper = np.nanpercentile(result["target_climatology"], [1, 99])
    panels = (
        (result["target_climatology"], "NOAA test climatology", "turbo", lower, upper),
        (result["generated_climatology"], "150k generated climatology", "turbo", lower, upper),
        (bias, f"bias; RMSE {result['metrics']['rmse_c']:.3f} degC", "RdBu_r", -0.5, 0.5),
    )
    for axis, (field, title, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.pcolormesh(result["lon"], result["lat"], field, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Longitude (degE)")
        figure.colorbar(image, ax=axis, shrink=0.77)
    axes[0].set_ylabel("Latitude (degN)")
    figure.savefig(FIGURES / "noaa_5km_test_climatology_and_bias.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    limit = float(np.nanpercentile(np.abs(warming), 99))
    image = axis.pcolormesh(lon, lat, warming, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set(title="NOAA-grid model: ACCESS-CM2 2080s minus 1980s", xlabel="Longitude (degE)", ylabel="Latitude (degN)")
    figure.colorbar(image, ax=axis, label="SST change (degC)")
    figure.savefig(FIGURES / "noaa_5km_access_cm2_warming_signal.png", dpi=180)
    plt.close(figure)
    return {
        **result["metrics"],
        **coastal_metrics,
        "access_warming_mean_c": float(np.nanmean(warming)),
        "access_warming_p01_c": float(np.nanpercentile(warming, 1)),
        "access_warming_p99_c": float(np.nanpercentile(warming, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing", action="store_true")
    arguments = parser.parse_args()
    required = [
        *AR_PRODUCTS.values(),
        *GAN_PRODUCTS.values(),
        NOAA_TEST,
        NOAA_HISTORICAL,
        NOAA_FUTURE,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing and not arguments.allow_missing:
        raise FileNotFoundError("Required products are missing:\n" + "\n".join(map(str, missing)))
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "passed",
        "ar": None,
        "gan": None,
        "noaa": None,
        "missing": [str(path) for path in missing],
    }
    if all(path.is_file() for path in AR_PRODUCTS.values()):
        report["ar"] = analyse_ar()
        with netCDF4.Dataset(AR_PRODUCTS["Legacy Flow-AR"]) as reference:
            reference_dates = _dates(reference)
        if all(path.is_file() for path in GAN_PRODUCTS.values()):
            report["gan"] = analyse_gans(reference_dates)
    if all(path.is_file() for path in (NOAA_TEST, NOAA_HISTORICAL, NOAA_FUTURE)):
        report["noaa"] = analyse_noaa()
    (REPORTS / "extended_model_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
