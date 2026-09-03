"""Small, deterministic prediction figures written during SRDN training."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_prediction_preview(
    output: str | Path,
    target: np.ndarray,
    bilinear: np.ndarray,
    prediction: np.ndarray,
    ocean_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
) -> None:
    """Save one masked target/reference/prediction comparison atomically."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(ocean_mask, dtype=bool)
    target = np.where(mask, np.asarray(target, dtype=np.float32), np.nan)
    bilinear = np.where(mask, np.asarray(bilinear, dtype=np.float32), np.nan)
    prediction = np.where(mask, np.asarray(prediction, dtype=np.float32), np.nan)

    fields = [target, bilinear, prediction]
    values = np.concatenate([field[np.isfinite(field)] for field in fields])
    vmin, vmax = np.quantile(values, [0.01, 0.99])
    errors = [bilinear - target, prediction - target, prediction - bilinear]
    error_values = np.concatenate([field[np.isfinite(field)] for field in errors])
    error_limit = max(float(np.quantile(np.abs(error_values), 0.99)), 0.05)

    extent = [
        float(np.min(lon)),
        float(np.max(lon)),
        float(np.min(lat)),
        float(np.max(lat)),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    names = ["Target", "Bilinear", "Model prediction"]
    image = None
    for axis, field, name in zip(axes[0], fields, names):
        image = axis.imshow(
            np.ma.masked_invalid(field),
            origin="lower",
            extent=extent,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(name)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
    figure.colorbar(image, ax=axes[0].tolist(), label="SST (°C)", shrink=0.82)

    error_names = ["Bilinear − target", "Model − target", "Model − bilinear"]
    image = None
    for axis, field, name in zip(axes[1], errors, error_names):
        image = axis.imshow(
            np.ma.masked_invalid(field),
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-error_limit,
            vmax=error_limit,
        )
        axis.set_title(name)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
    figure.colorbar(image, ax=axes[1].tolist(), label="difference (°C)", shrink=0.82)
    figure.suptitle(title)

    temporary = output.with_suffix(".partial.png")
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    temporary.replace(output)
