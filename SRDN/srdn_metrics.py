"""Physical-unit metrics and diagnostics for SRDN experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def denormalize(values, mean: float, std: float):
    return np.asarray(values, dtype=np.float32) * float(std) + float(mean)


def masked_field_metrics(prediction, target, mask):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim == 2:
        prediction = prediction[None]
        target = target[None]
    if mask.ndim == 2:
        mask = np.broadcast_to(mask[None], target.shape)
    else:
        mask = np.broadcast_to(mask, target.shape)
    finite = mask & np.isfinite(prediction) & np.isfinite(target)
    if not finite.any():
        raise ValueError("no finite ocean pixels available for scoring")
    error = prediction - target
    values = error[finite]
    return {
        "mae_c": float(np.mean(np.abs(values))),
        "rmse_c": float(np.sqrt(np.mean(np.square(values)))),
        "bias_c": float(np.mean(values)),
        "target_mean_c": float(np.mean(target[finite])),
        "prediction_mean_c": float(np.mean(prediction[finite])),
        "target_std_c": float(np.std(target[finite])),
        "prediction_std_c": float(np.std(prediction[finite])),
        "nonfinite_ocean_pixels": int((mask & ~np.isfinite(prediction)).sum()),
    }


def daily_mse(prediction, target, mask):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.ndim != 3:
        raise ValueError("daily metrics expect (time, lat, lon) fields")
    if mask.ndim == 2:
        mask = np.broadcast_to(mask[None], prediction.shape)
    finite = mask & np.isfinite(prediction) & np.isfinite(target)
    counts = finite.sum(axis=(1, 2))
    if np.any(counts == 0):
        raise ValueError("at least one day has no finite ocean pixels")
    error = prediction - target
    return np.sum(np.where(finite, error * error, 0.0), axis=(1, 2)) / counts


def daily_spatial_correlation(prediction, target, mask):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2:
        mask = np.broadcast_to(mask[None], prediction.shape)
    correlations = []
    for pred, truth, valid in zip(prediction, target, mask):
        valid = np.array(valid, dtype=bool, copy=True)
        valid &= np.isfinite(pred) & np.isfinite(truth)
        x, y = pred[valid], truth[valid]
        x = x - x.mean()
        y = y - y.mean()
        denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
        correlations.append(float(np.sum(x * y) / denominator) if denominator else 0.0)
    return np.asarray(correlations, dtype=np.float64)


def radial_power_profile(field, mask):
    field = np.asarray(field, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    values = np.where(mask, field, 0.0)
    values = values - values[mask].mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(values))) ** 2
    height, width = values.shape
    yy, xx = np.indices(values.shape)
    radius = np.hypot(yy - height / 2.0, xx - width / 2.0).astype(np.int64)
    counts = np.bincount(radius.ravel())
    totals = np.bincount(radius.ravel(), weights=power.ravel())
    limit = min(height, width) // 2
    return totals[:limit] / np.maximum(counts[:limit], 1.0)


def spectral_log_power_error(prediction, target, mask):
    errors = []
    for pred, truth in zip(prediction, target):
        pred_power = radial_power_profile(pred, mask)
        target_power = radial_power_profile(truth, mask)
        errors.append(float(np.mean(np.abs(
            np.log1p(pred_power) - np.log1p(target_power)
        ))))
    return float(np.mean(errors))


def coarse_block_differences(values, coarse, coarse_mask, fine_mask, shrink):
    """Return fine valid-block mean minus coarse values for valid blocks."""
    values = np.asarray(values, dtype=np.float64)
    coarse = np.asarray(coarse, dtype=np.float64)
    coarse_mask = np.asarray(coarse_mask, dtype=bool)
    fine_mask = np.asarray(fine_mask, dtype=bool)
    if values.ndim == 2:
        values = values[None]
        fine_mask = fine_mask[None]
    if coarse.ndim == 2:
        coarse = coarse[None]
        coarse_mask = coarse_mask[None]
    batch, height, width = values.shape
    if height % shrink or width % shrink:
        raise ValueError("fine dimensions must be divisible by shrink")
    coarse_height, coarse_width = height // shrink, width // shrink
    masks = fine_mask.reshape(batch, coarse_height, shrink, coarse_width, shrink)
    # Avoid NaN*0 propagation when callers represent land as NaN.
    values = np.where(fine_mask, values, 0.0)
    blocks = values.reshape(batch, coarse_height, shrink, coarse_width, shrink)
    counts = masks.sum(axis=(2, 4))
    means = (blocks * masks).sum(axis=(2, 4)) / np.maximum(counts, 1.0)
    valid = coarse_mask & (counts > 0) & np.isfinite(means) & np.isfinite(coarse)
    return (means - coarse)[valid]


def coarse_consistency_error(values, coarse, coarse_mask, fine_mask, shrink):
    differences = coarse_block_differences(
        values, coarse, coarse_mask, fine_mask, shrink
    )
    if not len(differences):
        return {"mae_c": float("nan"), "rmse_c": float("nan"), "max_abs_c": float("nan")}
    return {
        "mae_c": float(np.mean(np.abs(differences))),
        "rmse_c": float(np.sqrt(np.mean(np.square(differences)))),
        "max_abs_c": float(np.max(np.abs(differences))),
    }


def coast_distance_masks(ocean_mask: np.ndarray):
    """Return ocean bands at 1 pixel, 4 pixels, and beyond 4 pixels."""
    ocean = np.asarray(ocean_mask, dtype=bool)
    north = np.zeros_like(ocean); north[1:] = ocean[:-1]
    south = np.zeros_like(ocean); south[:-1] = ocean[1:]
    west = np.zeros_like(ocean); west[:, 1:] = ocean[:, :-1]
    east = np.zeros_like(ocean); east[:, :-1] = ocean[:, 1:]
    adjacent_to_land = ocean & ~(north & south & west & east)
    band1 = adjacent_to_land
    grown = band1.copy()
    for _ in range(3):
        n = np.zeros_like(grown); n[1:] = grown[:-1]
        s = np.zeros_like(grown); s[:-1] = grown[1:]
        w = np.zeros_like(grown); w[:, 1:] = grown[:, :-1]
        e = np.zeros_like(grown); e[:, :-1] = grown[:, 1:]
        grown = ocean & (grown | n | s | w | e)
    band4 = grown & ~band1
    interior = ocean & ~grown
    return {"coast_1px": band1, "coast_2_4px": band4, "interior_gt4px": interior}


def paired_bootstrap_delta(model_mse, reference_mse, seed=42, samples=2000):
    model_mse = np.asarray(model_mse, dtype=np.float64)
    reference_mse = np.asarray(reference_mse, dtype=np.float64)
    if model_mse.shape != reference_mse.shape or model_mse.ndim != 1:
        raise ValueError("paired bootstrap inputs must be equal-length vectors")
    rng = np.random.default_rng(int(seed))
    differences = model_mse - reference_mse
    draws = rng.integers(0, len(differences), size=(int(samples), len(differences)))
    means = differences[draws].mean(axis=1)
    return {
        "delta_mse_c2": float(differences.mean()),
        "ci95_low_c2": float(np.quantile(means, 0.025)),
        "ci95_high_c2": float(np.quantile(means, 0.975)),
        "win_at_95pct": bool(np.quantile(means, 0.975) < 0.0),
    }


def write_json(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
