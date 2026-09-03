"""Evaluate an SRDN run on the chronological test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np
import tensorflow as tf

from model_srdn_advanced import CoarseConsistencyProjection
from srdn_data import DerivedProduct, SRDNData
from srdn_metrics import (
    coast_distance_masks,
    daily_mse,
    daily_spatial_correlation,
    denormalize,
    masked_field_metrics,
    paired_bootstrap_delta,
    spectral_log_power_error,
    write_json,
)
from train_srdn import build_model, load_config


def restore_model(config: dict, run_dir: Path):
    model = build_model(config)
    derived = DerivedProduct(config["derived_path"])
    data = SRDNData(
        config["source_path"], derived, config["normalization_path"],
        config["test_date_ranges"],
    )
    inputs, _ = data.batch([0])
    model(inputs, training=False)
    checkpoint = tf.train.latest_checkpoint(str(run_dir / "checkpoints"))
    if checkpoint is None:
        raise FileNotFoundError(f"no TensorFlow checkpoint found in {run_dir}")
    tf.train.Checkpoint(model=model).restore(checkpoint).expect_partial()
    return model, data, derived, checkpoint


def bilinear_prediction(inputs, derived, consistent: bool):
    coarse = tf.convert_to_tensor(inputs["coarse_sst"])
    result = tf.image.resize(
        coarse,
        [derived.fine_shape[0], derived.fine_shape[1]],
        method="bilinear",
        antialias=False,
    )
    fine_mask = tf.convert_to_tensor(inputs["fine_mask"])
    result = result * fine_mask
    if consistent:
        result = CoarseConsistencyProjection(derived.coarsen_factor)(
            [result,
             tf.convert_to_tensor(inputs["coarse_sst"]),
             tf.convert_to_tensor(inputs["coarse_mask"]),
             fine_mask]
        )
    return result.numpy()


def _accumulate(total, prediction, target, mask):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    finite = mask & np.isfinite(prediction) & np.isfinite(target)
    error = prediction - target
    total["abs"] += float(np.abs(error[finite]).sum())
    total["sq"] += float(np.square(error[finite]).sum())
    total["bias"] += float(error[finite].sum())
    total["count"] += int(finite.sum())
    total["target_sum"] += float(target[finite].sum())
    total["prediction_sum"] += float(prediction[finite].sum())
    total["target_sq"] += float(np.square(target[finite]).sum())
    total["prediction_sq"] += float(np.square(prediction[finite]).sum())
    total["nonfinite"] += int((mask & ~np.isfinite(prediction)).sum())


def _finish(total):
    count = max(total["count"], 1)
    return {
        "mae_c": total["abs"] / count,
        "rmse_c": float(np.sqrt(total["sq"] / count)),
        "bias_c": total["bias"] / count,
        "target_mean_c": total["target_sum"] / count,
        "prediction_mean_c": total["prediction_sum"] / count,
        "target_std_c": float(np.sqrt(max(total["target_sq"] / count - (total["target_sum"] / count) ** 2, 0.0))),
        "prediction_std_c": float(np.sqrt(max(total["prediction_sq"] / count - (total["prediction_sum"] / count) ** 2, 0.0))),
        "nonfinite_ocean_pixels": total["nonfinite"],
    }


def _new_total():
    return {key: 0.0 for key in (
        "abs", "sq", "bias", "count", "target_sum", "prediction_sum",
        "target_sq", "prediction_sq", "nonfinite",
    )}


def save_samples(path, dates, prediction, target, bilinear, coarse, derived):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.nc")
    with netCDF4.Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("time", len(dates))
        output.createDimension("lat", len(derived.lat))
        output.createDimension("lon", len(derived.lon))
        output.createDimension("lat_lr", len(derived.lat_lr))
        output.createDimension("lon_lr", len(derived.lon_lr))
        time_var = output.createVariable("time", "i8", ("time",))
        time_var.units = "days since 1979-01-01 00:00:00"
        time_var.calendar = "proleptic_gregorian"
        time_var[:] = netCDF4.date2num(
            [netCDF4.datetime.strptime(date, "%Y-%m-%d") for date in dates],
            time_var.units,
            calendar=time_var.calendar,
        )
        for name, values, dims in (
            ("sst_generated", prediction, ("time", "lat", "lon")),
            ("sst_target", target, ("time", "lat", "lon")),
            ("sst_bilinear", bilinear, ("time", "lat", "lon")),
            ("sst_coarse", coarse, ("time", "lat_lr", "lon_lr")),
        ):
            variable = output.createVariable(name, "f4", dims, zlib=True, complevel=4)
            variable[:] = values.astype(np.float32)
        output.createVariable("ocean_mask", "i1", ("lat", "lon"))[:] = derived.ocean_mask
        output.createVariable("ocean_mask_lr", "i1", ("lat_lr", "lon_lr"))[:] = derived.ocean_mask_lr
        output.createVariable("lat", "f8", ("lat",))[:] = derived.lat
        output.createVariable("lon", "f8", ("lon",))[:] = derived.lon
        output.createVariable("lat_lr", "f8", ("lat_lr",))[:] = derived.lat_lr
        output.createVariable("lon_lr", "f8", ("lon_lr",))[:] = derived.lon_lr
        output.setncattr("units", "degrees C")
        output.setncattr("land_value", "NaN")
    temporary.replace(path)


def evaluate(config, run_dir: Path, batch_size=2, max_days=None):
    model, data, derived, checkpoint = restore_model(config, run_dir)
    positions = np.arange(len(data), dtype=np.int64)
    if max_days is not None:
        positions = positions[: int(max_days)]
    coast_masks = coast_distance_masks(derived.ocean_mask)
    totals = _new_total()
    bilinear_total = _new_total()
    daily_model, daily_bilinear, daily_corr = [], [], []
    regional = {name: _new_total() for name in coast_masks}
    sample_positions = np.unique(np.linspace(0, len(positions) - 1, min(8, len(positions))).astype(int))
    sample_set = set(int(value) for value in sample_positions)
    sample_prediction = {}; sample_target = {}; sample_bilinear = {}; sample_coarse = {}; sample_dates = {}
    spectral_predictions, spectral_targets = [], []

    for start in range(0, len(positions), int(batch_size)):
        current = positions[start : start + int(batch_size)]
        inputs, target_normalized = data.batch(current)
        prediction_normalized = model(inputs, training=False).numpy()
        bilinear_normalized = bilinear_prediction(inputs, derived, consistent=True)
        mask = inputs["fine_mask"][..., 0].astype(bool)
        prediction = denormalize(prediction_normalized[..., 0], data.mean, data.std)
        target = denormalize(target_normalized[..., 0], data.mean, data.std)
        bilinear = denormalize(bilinear_normalized[..., 0], data.mean, data.std)
        prediction = np.where(mask, prediction, np.nan)
        target = np.where(mask, target, np.nan)
        bilinear = np.where(mask, bilinear, np.nan)
        _accumulate(totals, prediction, target, mask)
        _accumulate(bilinear_total, bilinear, target, mask)
        daily_model.extend(daily_mse(prediction, target, derived.ocean_mask))
        daily_bilinear.extend(daily_mse(bilinear, target, derived.ocean_mask))
        daily_corr.extend(daily_spatial_correlation(prediction, target, derived.ocean_mask))
        for name, band in coast_masks.items():
            _accumulate(regional[name], prediction, target, band)
        if len(spectral_predictions) < 8:
            spectral_predictions.extend(prediction[: 8 - len(spectral_predictions)])
            spectral_targets.extend(target[: 8 - len(spectral_targets)])
        for offset, position in enumerate(current):
            if int(start + offset) in sample_set:
                sample_prediction[int(position)] = prediction[offset]
                sample_target[int(position)] = target[offset]
                sample_bilinear[int(position)] = bilinear[offset]
                coarse = inputs["coarse_sst"][offset, ..., 0]
                sample_coarse[int(position)] = denormalize(coarse, data.mean, data.std)
                sample_dates[int(position)] = data.dates[int(position)]

    metrics = _finish(totals)
    bilinear_metrics = _finish(bilinear_total)
    model_daily = np.asarray(daily_model, dtype=np.float64)
    bilinear_daily = np.asarray(daily_bilinear, dtype=np.float64)
    metrics["skill_vs_bilinear"] = float(1.0 - model_daily.mean() / bilinear_daily.mean())
    metrics["daily_spatial_correlation_mean"] = float(np.mean(daily_corr))
    metrics["paired_bootstrap_vs_bilinear"] = paired_bootstrap_delta(
        model_daily, bilinear_daily
    )
    metrics["spectral_log_power_error"] = spectral_log_power_error(
        np.asarray(spectral_predictions), np.asarray(spectral_targets), derived.ocean_mask
    )
    metrics["regions"] = {name: _finish(value) for name, value in regional.items()}
    metrics["bilinear_reference"] = bilinear_metrics
    metrics["checkpoint"] = checkpoint
    metrics["days"] = int(len(positions))
    metrics["model_variant"] = config["model_variant"]
    write_json(run_dir / "evaluation" / "metrics.json", metrics)
    np.savez_compressed(
        run_dir / "evaluation" / "daily_metrics.npz",
        model_mse=model_daily,
        bilinear_mse=bilinear_daily,
        spatial_correlation=np.asarray(daily_corr),
    )
    if sample_dates:
        ordered = sorted(sample_dates)
        save_samples(
            run_dir / "evaluation" / "test_samples.nc",
            [sample_dates[index] for index in ordered],
            np.stack([sample_prediction[index] for index in ordered]),
            np.stack([sample_target[index] for index in ordered]),
            np.stack([sample_bilinear[index] for index in ordered]),
            np.stack([sample_coarse[index] for index in ordered]),
            derived,
        )
    data.close()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-days", type=int)
    args = parser.parse_args()
    config_path = args.config or args.run / "config_used.json"
    evaluate(load_config(config_path), args.run, args.batch_size, args.max_days)


if __name__ == "__main__":
    main()
