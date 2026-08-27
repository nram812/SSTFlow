"""Coarse-grid authority for high-resolution SST generation.

The coarse SST channel is an ocean-only block mean.  These helpers separate a
fine field into its block mean and within-block anomaly, and can project a
generated field back onto the supplied coarse means.  The projection adds one
constant per valid block, so it does not damp or otherwise alter fine-scale
anomalies within that block.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPSILON = 1.0e-8


def _factor(field: torch.Tensor, coarse: torch.Tensor) -> int:
    fine_height, fine_width = field.shape[-2:]
    coarse_height, coarse_width = coarse.shape[-2:]
    if fine_height % coarse_height or fine_width % coarse_width:
        raise ValueError(
            f"Fine grid {fine_height}x{fine_width} is not divisible by coarse "
            f"grid {coarse_height}x{coarse_width}"
        )
    factor_height = fine_height // coarse_height
    factor_width = fine_width // coarse_width
    if factor_height != factor_width:
        raise ValueError(
            f"Expected a square coarsening factor, got {factor_height}x{factor_width}"
        )
    return factor_height


def masked_block_mean(
    field: torch.Tensor, mask: torch.Tensor, coarse_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ocean-only block means and the corresponding ocean fractions."""
    reference = field.new_empty((*field.shape[:-2], *coarse_shape))
    factor = _factor(field, reference)
    expanded_mask = mask.expand_as(field).to(field.dtype)
    ocean_fraction = F.avg_pool2d(expanded_mask, factor)
    block_mean = F.avg_pool2d(field * expanded_mask, factor) / ocean_fraction.clamp_min(
        EPSILON
    )
    return block_mean, ocean_fraction


def within_block_anomaly(
    field: torch.Tensor, mask: torch.Tensor, coarse_shape: tuple[int, int]
) -> torch.Tensor:
    """Remove every ocean block mean, retaining only fine-scale lag guidance."""
    block_mean, ocean_fraction = masked_block_mean(field, mask, coarse_shape)
    block_mean = torch.where(ocean_fraction > 0, block_mean, torch.zeros_like(block_mean))
    expanded = F.interpolate(block_mean, size=field.shape[-2:], mode="nearest")
    return (field - expanded) * mask.expand_as(field).to(field.dtype)


def project_to_coarse(
    field: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Make valid ocean-block means exactly equal condition channel zero.

    Condition channel one is the authoritative coarse validity mask.  Invalid
    coastal/land blocks are left unchanged except for the static fine-grid land
    mask.  The operation is differentiable but is intended for the boundary of
    sampling, keeping the rectified-flow velocity objective untouched.
    """
    if condition.shape[1] < 2:
        raise ValueError("Coarse consistency requires SST and validity channels")
    coarse_target = condition[:, :1]
    block_mean, ocean_fraction = masked_block_mean(
        field, mask, coarse_target.shape[-2:]
    )
    valid = (condition[:, 1:2] > 0.5) & (ocean_fraction > 0)
    correction = torch.where(valid, coarse_target - block_mean, 0.0)
    correction = F.interpolate(correction, size=field.shape[-2:], mode="nearest")
    expanded_mask = mask.expand_as(field).to(field.dtype)
    return (field + correction) * expanded_mask


def coarse_consistency_mse(
    field: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared block-mean error over valid coarse ocean cells."""
    if condition.shape[1] < 2:
        raise ValueError("Coarse consistency requires SST and validity channels")
    block_mean, ocean_fraction = masked_block_mean(
        field, mask, condition.shape[-2:]
    )
    valid = (condition[:, 1:2] > 0.5) & (ocean_fraction > 0)
    squared = (block_mean - condition[:, :1]).square()
    return (squared * valid).sum() / valid.sum().clamp_min(1)
