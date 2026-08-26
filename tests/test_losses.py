"""C4: every loss must ignore land."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from losses import (
    apply_mask,
    conservation_loss,
    hinge_discriminator_loss,
    hinge_generator_loss,
    masked_bias,
    masked_l1,
    masked_mean,
    masked_mse,
    masked_rmse,
    spectral_amplitude_loss,
)


@pytest.fixture
def fields():
    torch.manual_seed(0)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :, :4] = 1.0
    prediction = torch.randn(2, 1, 8, 8)
    target = torch.randn(2, 1, 8, 8)
    return prediction, target, mask


def test_masked_mse_ignores_land(fields):
    prediction, target, mask = fields
    reference = masked_mse(prediction, target, mask)
    polluted = prediction.clone()
    polluted[:, :, :, 4:] = 1.0e6
    assert masked_mse(polluted, target, mask) == pytest.approx(float(reference))


def test_masked_mse_matches_manual(fields):
    prediction, target, mask = fields
    manual = ((prediction - target) ** 2)[mask.bool()].mean()
    assert masked_mse(prediction, target, mask) == pytest.approx(float(manual), rel=1e-6)


def test_masked_mse_full_mask_equals_mse(fields):
    prediction, target, _ = fields
    ones = torch.ones_like(prediction)
    assert masked_mse(prediction, target, ones) == pytest.approx(
        float(F.mse_loss(prediction, target)), rel=1e-6
    )


def test_empty_mask_is_finite(fields):
    prediction, target, mask = fields
    value = masked_mse(prediction, target, torch.zeros_like(mask))
    assert torch.isfinite(value) and float(value) == 0.0


def test_masked_bias_rmse_and_l1(fields):
    _, target, mask = fields
    shifted = target + 2.0
    assert masked_bias(shifted, target, mask) == pytest.approx(2.0, rel=1e-6)
    assert masked_rmse(shifted, target, mask) == pytest.approx(2.0, rel=1e-6)
    assert masked_l1(shifted, target, mask) == pytest.approx(2.0, rel=1e-6)


def test_masked_mean(fields):
    _, _, mask = fields
    assert masked_mean(torch.full_like(mask, 3.0), mask) == pytest.approx(3.0)


def test_gradients_are_zero_over_land(fields):
    prediction, target, mask = fields
    prediction = prediction.clone().requires_grad_(True)
    masked_mse(prediction, target, mask).backward()
    assert float(prediction.grad[mask == 0].abs().max()) == 0.0
    assert float(prediction.grad[mask > 0].abs().max()) > 0.0


def test_apply_mask_zeroes_land(fields):
    prediction, _, mask = fields
    assert float(apply_mask(prediction, mask)[mask == 0].abs().max()) == 0.0


def test_spectral_loss_zero_for_identical(fields):
    prediction, _, mask = fields
    assert spectral_amplitude_loss(prediction, prediction, mask) == pytest.approx(
        0.0, abs=1e-10
    )


def test_spectral_loss_positive_for_different(fields):
    prediction, target, mask = fields
    assert float(spectral_amplitude_loss(prediction, target, mask)) > 0.0


def test_conservation_loss_zero_for_consistent_pair():
    mask = torch.ones(1, 1, 8, 8)
    prediction = torch.randn(1, 1, 8, 8)
    coarse = F.avg_pool2d(prediction, 4)
    assert conservation_loss(prediction, coarse, mask, 4) == pytest.approx(0.0, abs=1e-10)


def test_conservation_loss_positive_when_inconsistent():
    mask = torch.ones(1, 1, 8, 8)
    prediction = torch.randn(1, 1, 8, 8)
    coarse = F.avg_pool2d(prediction, 4) + 1.0
    assert float(conservation_loss(prediction, coarse, mask, 4)) > 0.5


def test_hinge_losses_signs():
    confident = hinge_discriminator_loss(
        torch.full((4, 1, 2, 2), 3.0), torch.full((4, 1, 2, 2), -3.0)
    )
    confused = hinge_discriminator_loss(torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2))
    assert float(confident) == pytest.approx(0.0)
    assert float(confused) == pytest.approx(2.0)
    assert float(hinge_generator_loss(torch.full((4, 1, 2, 2), 2.0))) == pytest.approx(-2.0)


def test_shape_mismatch_raises(fields):
    prediction, _, mask = fields
    with pytest.raises(ValueError, match="Shape mismatch"):
        masked_mse(prediction, torch.zeros(2, 1, 4, 4), mask)
