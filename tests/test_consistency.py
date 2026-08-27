"""Coarse-authority and fine-scale lag-guidance regression tests."""

import torch

from consistency import (
    coarse_consistency_mse,
    masked_block_mean,
    project_to_coarse,
    within_block_anomaly,
)
from flow import rollout


def fields():
    torch.manual_seed(3)
    fine = torch.randn(2, 1, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    mask[:, :, :2, :4] = 0
    condition = torch.randn(2, 2, 4, 4)
    condition[:, 1] = 1
    condition[:, 1, 0, 0] = 0
    return fine * mask, condition, mask


def test_projection_matches_valid_coarse_ocean_means_exactly():
    fine, condition, mask = fields()
    projected = project_to_coarse(fine, condition, mask)
    means, _ = masked_block_mean(projected, mask, condition.shape[-2:])
    valid = condition[:, 1:2].bool()
    torch.testing.assert_close(means[valid], condition[:, :1][valid], atol=2e-6, rtol=0)
    assert float(coarse_consistency_mse(projected, condition, mask)) < 1e-12
    assert torch.count_nonzero(projected[mask == 0]) == 0


def test_projection_preserves_within_block_anomalies_and_gradients():
    fine, condition, mask = fields()
    fine.requires_grad_()
    before = within_block_anomaly(fine, mask, condition.shape[-2:])
    projected = project_to_coarse(fine, condition, mask)
    after = within_block_anomaly(projected, mask, condition.shape[-2:])
    torch.testing.assert_close(after, before, atol=2e-6, rtol=0)
    projected.square().mean().backward()
    assert fine.grad is not None and torch.isfinite(fine.grad).all()


def test_lag_guidance_has_zero_ocean_mean_in_every_block():
    fine, condition, mask = fields()
    guidance = within_block_anomaly(fine, mask, condition.shape[-2:])
    means, fractions = masked_block_mean(guidance, mask, condition.shape[-2:])
    torch.testing.assert_close(
        means[fractions > 0], torch.zeros_like(means[fractions > 0]), atol=2e-6, rtol=0
    )


class ZeroVelocityAR(torch.nn.Module):
    def forward(self, state, condition, mask, previous_state, flow_time):
        return torch.zeros_like(state)


def test_rollout_tracks_evolving_coarse_input_despite_static_dynamics():
    _, condition, mask = fields()
    condition = condition[:1, None].repeat(1, 3, 1, 1, 1)
    condition[:, 0, 0] = -1.0
    condition[:, 1, 0] = 0.0
    condition[:, 2, 0] = 1.0
    initial = torch.zeros(1, 1, 16, 16)
    generated = rollout(
        ZeroVelocityAR(),
        initial,
        condition,
        mask[:1],
        steps=1,
        generator=torch.Generator().manual_seed(4),
        enforce_coarse_consistency=True,
    )
    for lead in range(3):
        means, _ = masked_block_mean(
            generated[:, lead], mask[:1], condition.shape[-2:]
        )
        valid = condition[:, lead, 1:2].bool()
        torch.testing.assert_close(
            means[valid], condition[:, lead, :1][valid], atol=2e-6, rtol=0
        )
