"""Masked losses.

Every loss in this module restricts the reduction to ocean pixels.  Land is
represented by zeros in the tensors, which would otherwise dominate a plain
``mse_loss`` (34% of the grid) and teach the model to reproduce a constant.

All reductions divide by ``mask.sum()`` guarded with ``clamp_min(1)`` so an
empty mask can never produce a NaN.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Guard used in every masked denominator.
EPSILON = 1.0e-8


def expand_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast an ocean mask to the shape of ``reference``."""
    if mask.dim() != reference.dim():
        raise ValueError(
            f"Mask rank {mask.dim()} does not match tensor rank {reference.dim()}"
        )
    return mask.expand_as(reference).to(dtype=reference.dtype)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the entries where ``mask`` is one."""
    mask = expand_mask(mask, values)
    total = (values * mask).sum()
    count = mask.sum().clamp_min(EPSILON)
    return total / count


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Ocean-only mean squared error."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    return masked_mean((prediction - target).square(), mask)


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    return masked_mean((prediction - target).abs(), mask)


def masked_rmse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mse(prediction, target, mask).clamp_min(0.0).sqrt()


def masked_bias(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mean(prediction - target, mask)


def apply_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out land pixels; used before spectral losses and before sampling."""
    return values * expand_mask(mask, values)


def spectral_amplitude_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Log-amplitude rFFT loss on the land-zeroed fields.

    The FFT is global so it cannot be masked pixel-wise; both fields are given
    the *same* zero-filled land, so the land contribution cancels in the
    difference of spectra and only the ocean structure is scored.
    """
    prediction = apply_mask(prediction.float(), mask)
    target = apply_mask(target.float(), mask)
    predicted_spectrum = torch.fft.rfft2(prediction, norm="ortho").abs()
    target_spectrum = torch.fft.rfft2(target, norm="ortho").abs()
    difference = torch.log1p(predicted_spectrum) - torch.log1p(target_spectrum)
    return difference.square().mean()


def conservation_loss(
    prediction: torch.Tensor,
    coarse_target: torch.Tensor,
    mask: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """Optional soft constraint that the block average matches the predictor.

    Disabled by default (``lambda_conservation = 0``) because the first round of
    experiments is deliberately unconstrained, but kept here so the ablation can
    be run without touching the training loops.
    """
    mask = expand_mask(mask, prediction)
    pooled_values = F.avg_pool2d(prediction * mask, factor)
    pooled_counts = F.avg_pool2d(mask, factor).clamp_min(EPSILON)
    block_mean = pooled_values / pooled_counts
    valid = (F.avg_pool2d(mask, factor) >= 0.5).to(prediction.dtype)
    difference = (block_mean - coarse_target).square() * valid
    return difference.sum() / valid.sum().clamp_min(EPSILON)


def hinge_discriminator_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """Standard hinge loss for the critic; bounded and stable without a scaler."""
    real_term = F.relu(1.0 - real_logits).mean()
    fake_term = F.relu(1.0 + fake_logits).mean()
    return real_term + fake_term


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()
