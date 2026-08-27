"""Periodic NetCDF products, preview figures, and loss-curve plots.

All three training entrypoints call the same helpers so the artefacts are
directly comparable between experiments:

``predictions/preview_step_XXXXXX.png``   coarse input / truth / generated / error
``predictions/loss_curve_step_XXXXXX.png``  raw and smoothed training loss
``netcdf/sample_step_XXXXXX.nc``          generated + target fields in degrees C
``netcdf/rollout_step_XXXXXX.nc``         free-running rollout (autoregressive)

Land is written as NaN so the products open identically to the source data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

from coarsen import upsample_nearest  # noqa: E402
from common import denormalize  # noqa: E402


def to_physical(
    values: torch.Tensor, normalization: dict, ocean_mask: np.ndarray
) -> np.ndarray:
    """Convert normalised tensors to degrees Celsius with NaN over land."""
    array = values.detach().float().cpu().numpy()
    array = denormalize(
        array, float(normalization["sst_mean"]), float(normalization["sst_std"])
    )
    mask = np.asarray(ocean_mask, dtype=bool)
    while mask.ndim < array.ndim:
        mask = mask[None]
    return np.where(np.broadcast_to(mask, array.shape), array, np.nan)


def coarse_to_physical(
    condition: torch.Tensor, normalization: dict, coarse_mask: np.ndarray
) -> np.ndarray:
    """Physical coarse predictor from the two-channel condition tensor."""
    array = condition.detach().float().cpu().numpy()[:, 0]
    array = denormalize(
        array, float(normalization["sst_mean"]), float(normalization["sst_std"])
    )
    mask = np.broadcast_to(np.asarray(coarse_mask, dtype=bool)[None], array.shape)
    return np.where(mask, array, np.nan)


def field_metrics(generated: np.ndarray, target: np.ndarray) -> dict:
    """NaN-aware error statistics; land is NaN so it is excluded everywhere."""
    error = generated - target
    ocean = np.isfinite(target)
    finite = ocean & np.isfinite(generated)
    if not finite.any():
        raise ValueError("No finite ocean pixels to score")
    return {
        "mae": float(np.nanmean(np.abs(error))),
        "rmse": float(np.sqrt(np.nanmean(np.square(error)))),
        "bias": float(np.nanmean(error)),
        "target_mean": float(np.nanmean(target)),
        "generated_mean": float(np.nanmean(generated)),
        "target_std": float(np.nanstd(target)),
        "generated_std": float(np.nanstd(generated)),
        "generated_min": float(np.nanmin(generated)),
        "generated_max": float(np.nanmax(generated)),
        "nonfinite_ocean_pixels": int((ocean & ~np.isfinite(generated)).sum()),
    }


def radial_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a land-zeroed 2-D field."""
    values = np.nan_to_num(field, nan=0.0)
    values = values - values.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(values))) ** 2
    height, width = values.shape
    y, x = np.indices((height, width))
    radius = np.hypot(y - height / 2.0, x - width / 2.0).astype(int)
    counts = np.bincount(radius.ravel())
    totals = np.bincount(radius.ravel(), weights=power.ravel())
    limit = min(height, width) // 2
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = totals[:limit] / np.maximum(counts[:limit], 1)
    return np.arange(limit), profile


def save_preview(
    coarse: np.ndarray,
    target: np.ndarray,
    generated: np.ndarray,
    output_path: Path,
    title: str,
    coarsen_factor: int,
) -> None:
    """Four-panel comparison plus a power-spectrum panel for the first sample."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite_target = target[np.isfinite(target)]
    vmin = float(np.percentile(finite_target, 1)) if finite_target.size else 0.0
    vmax = float(np.percentile(finite_target, 99)) if finite_target.size else 1.0
    error = generated - target
    scale = float(np.nanpercentile(np.abs(error), 99)) if np.isfinite(error).any() else 1.0
    scale = max(scale, 1.0e-6)

    figure, axes = plt.subplots(1, 5, figsize=(26, 5.2), constrained_layout=True)
    panels = (
        (upsample_nearest(coarse, coarsen_factor), "coarse input (nearest)", "viridis", (vmin, vmax)),
        (target, "high-resolution truth", "viridis", (vmin, vmax)),
        (generated, "generated", "viridis", (vmin, vmax)),
        (error, "generated - truth", "RdBu_r", (-scale, scale)),
    )
    for axis, (field, label, cmap, limits) in zip(axes, panels):
        image = axis.imshow(
            field, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1]
        )
        axis.set_title(label)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)

    wavenumber, target_spectrum = radial_spectrum(target)
    _, generated_spectrum = radial_spectrum(generated)
    axes[4].loglog(wavenumber[1:], target_spectrum[1:], label="truth")
    axes[4].loglog(wavenumber[1:], generated_spectrum[1:], label="generated")
    axes[4].set_title("radial power spectrum")
    axes[4].set_xlabel("wavenumber")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    figure.suptitle(title)
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def save_loss_curve(history: list[dict], output_path: Path, keys=("total",)) -> None:
    """Raw and rolling-mean loss curves on a log scale."""
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([record["step"] for record in history])
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for key in keys:
        values = np.asarray(
            [float(record.get(key, np.nan)) for record in history], dtype=np.float64
        )
        if not np.isfinite(values).any():
            continue
        axis.plot(steps, values, alpha=0.25, linewidth=0.8)
        window = max(1, min(101, len(values) // 10))
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(values, kernel, mode="valid")
            axis.plot(steps[window - 1 :], smoothed, linewidth=1.8, label=key)
        else:
            axis.plot(steps, values, linewidth=1.8, label=key)
    axis.set_yscale("log")
    axis.set_xlabel("optimiser step")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def _write(dataset: xr.Dataset, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.nc")
    encoding = {
        name: {"dtype": "float32", "zlib": True, "complevel": 4}
        for name in dataset.data_vars
    }
    dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    os.replace(temporary, output_path)
    return output_path


def save_netcdf(
    generated: np.ndarray,
    target: np.ndarray,
    coarse: np.ndarray,
    dates: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    lat_lr: np.ndarray,
    lon_lr: np.ndarray,
    output_path: Path,
    attrs: dict,
) -> Path:
    """Write generated/target/coarse fields for a batch of days."""
    times = np.asarray(dates, dtype="datetime64[ns]")
    dataset = xr.Dataset(
        data_vars={
            "sst_generated": (("time", "lat", "lon"), generated.astype(np.float32)),
            "sst_target": (("time", "lat", "lon"), target.astype(np.float32)),
            "sst_coarse": (("time", "lat_lr", "lon_lr"), coarse.astype(np.float32)),
        },
        coords={
            "time": times,
            "lat": lat,
            "lon": lon,
            "lat_lr": lat_lr,
            "lon_lr": lon_lr,
        },
        attrs={"units": "degrees C", **attrs},
    )
    return _write(dataset, output_path)


def save_rollout_netcdf(
    generated: np.ndarray,
    target: np.ndarray,
    dates: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    output_path: Path,
    attrs: dict,
    coarse: np.ndarray | None = None,
    lat_lr: np.ndarray | None = None,
    lon_lr: np.ndarray | None = None,
) -> Path:
    """Write a free-running autoregressive rollout with a lead-time axis."""
    data_vars = {
        "sst_generated": (("lead", "lat", "lon"), generated.astype(np.float32)),
        "sst_target": (("lead", "lat", "lon"), target.astype(np.float32)),
    }
    coords = {
        "lead": np.arange(1, generated.shape[0] + 1, dtype=np.int32),
        "time": ("lead", np.asarray(dates, dtype="datetime64[ns]")),
        "lat": lat,
        "lon": lon,
    }
    if coarse is not None:
        if lat_lr is None or lon_lr is None:
            raise ValueError("lat_lr and lon_lr are required with rollout coarse data")
        data_vars["sst_coarse"] = (
            ("lead", "lat_lr", "lon_lr"), coarse.astype(np.float32)
        )
        coords.update(lat_lr=lat_lr, lon_lr=lon_lr)
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={"units": "degrees C", **attrs},
    )
    return _write(dataset, output_path)


def save_rollout_skill_plot(
    generated: np.ndarray, target: np.ndarray, output_path: Path, title: str
) -> None:
    """RMSE and bias as a function of forecast lead time."""
    leads = np.arange(1, generated.shape[0] + 1)
    error = generated - target
    rmse = np.sqrt(np.nanmean(np.square(error), axis=(1, 2)))
    bias = np.nanmean(error, axis=(1, 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(leads, rmse, marker="o", label="RMSE")
    axis.plot(leads, bias, marker="s", label="bias")
    axis.axhline(0.0, color="grey", linewidth=0.8)
    axis.set_xlabel("lead time (days)")
    axis.set_ylabel("degrees C")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)
