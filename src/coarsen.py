"""Mask-aware block coarsening from the 0.1 degree grid to the coarse grid.

The high-resolution OFAM field has permanently missing land cells.  A naive
``reshape(...).mean(...)`` would propagate a single NaN across an entire coarse
cell, so every routine here works on *valid counts* instead:

``coarse = sum(value over ocean cells) / count(ocean cells)``

A coarse cell is declared valid when at least ``min_valid_fraction`` of the
fine cells beneath it are ocean (default 0.5, i.e. the "over 50% of the domain
is NaN" rule).  Invalid coarse cells never enter a mean and are replaced with
the fill value, so no NaN can reach the network.
"""

from __future__ import annotations

import numpy as np


def check_divisible(shape: tuple[int, int], factor: int) -> None:
    height, width = shape
    if factor < 1:
        raise ValueError(f"Coarsening factor must be positive, got {factor}")
    if height % factor or width % factor:
        raise ValueError(
            f"Grid {height}x{width} is not divisible by coarsening factor {factor}"
        )


def block_valid_counts(ocean_mask: np.ndarray, factor: int) -> np.ndarray:
    """Number of ocean cells inside each coarse block."""
    ocean_mask = np.asarray(ocean_mask, dtype=bool)
    if ocean_mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {ocean_mask.shape}")
    check_divisible(ocean_mask.shape, factor)
    height, width = ocean_mask.shape
    blocks = ocean_mask.reshape(
        height // factor, factor, width // factor, factor
    )
    return blocks.sum(axis=(1, 3)).astype(np.int64)


def coarse_ocean_mask(
    ocean_mask: np.ndarray, factor: int, min_valid_fraction: float = 0.5
) -> np.ndarray:
    """Boolean mask of coarse cells that carry a trustworthy block average."""
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError(
            f"min_valid_fraction must lie in (0, 1], got {min_valid_fraction}"
        )
    counts = block_valid_counts(ocean_mask, factor)
    required = min_valid_fraction * factor * factor
    return counts >= required


def coarsen(
    values: np.ndarray,
    ocean_mask: np.ndarray,
    factor: int,
    min_valid_fraction: float = 0.5,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Block-average ``values`` over ocean cells only.

    Parameters
    ----------
    values:
        ``(lat, lon)`` or ``(sample, lat, lon)`` array of physical values.  Land
        entries may be NaN; they are ignored rather than propagated.
    ocean_mask:
        Static ``(lat, lon)`` boolean mask, ``True`` over ocean.
    factor:
        Integer coarsening factor; both grid dimensions must be divisible by it.
    min_valid_fraction:
        Minimum fraction of ocean cells required for a coarse cell to be valid.
    fill_value:
        Value written into coarse cells that fail the validity test.

    Returns
    -------
    ``(lat // factor, lon // factor)`` or ``(sample, ...)`` array.
    """
    values = np.asarray(values, dtype=np.float64)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D values, got shape {values.shape}")

    ocean_mask = np.asarray(ocean_mask, dtype=bool)
    if values.shape[-2:] != ocean_mask.shape:
        raise ValueError(
            f"Value grid {values.shape[-2:]} does not match mask {ocean_mask.shape}"
        )
    check_divisible(ocean_mask.shape, factor)

    samples, height, width = values.shape
    coarse_height, coarse_width = height // factor, width // factor

    # Zero out land before summing so NaN never participates in the arithmetic.
    numeric = np.where(ocean_mask[None], np.nan_to_num(values, nan=0.0), 0.0)
    totals = numeric.reshape(
        samples, coarse_height, factor, coarse_width, factor
    ).sum(axis=(2, 4))
    counts = block_valid_counts(ocean_mask, factor)

    valid = coarse_ocean_mask(ocean_mask, factor, min_valid_fraction)
    safe_counts = np.where(counts > 0, counts, 1).astype(np.float64)
    coarse = totals / safe_counts[None]
    coarse = np.where(valid[None], coarse, fill_value)

    if squeeze:
        coarse = coarse[0]
    return coarse.astype(np.float32)


def coarse_coordinates(coordinate: np.ndarray, factor: int) -> np.ndarray:
    """Block-average a 1-D coordinate vector to the coarse grid."""
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if coordinate.ndim != 1:
        raise ValueError(f"Expected a 1-D coordinate, got {coordinate.shape}")
    if coordinate.size % factor:
        raise ValueError(
            f"Coordinate length {coordinate.size} is not divisible by {factor}"
        )
    return coordinate.reshape(-1, factor).mean(axis=1)


def upsample_nearest(values: np.ndarray, factor: int) -> np.ndarray:
    """Replicate each coarse cell into a ``factor x factor`` block.

    Only used for diagnostics and for the "coarse input" panel of preview
    figures; the networks perform their own learnable/bilinear upsampling.
    """
    values = np.asarray(values)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None]
    output = np.repeat(np.repeat(values, factor, axis=-2), factor, axis=-1)
    return output[0] if squeeze else output
