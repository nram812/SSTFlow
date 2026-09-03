"""Focused tests for production free-running SST autoregressive inference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flow import masked_noise
from run_flow_ar_rollout import (
    checkpoint_semantics,
    generate_rollout,
    lead_count,
    require_inside_one_range,
    rollout_metrics,
)


class PreviousStateVelocity(torch.nn.Module):
    def forward(self, state, condition, mask, previous_state, flow_time):
        return previous_state * mask


def test_date_window_is_one_initial_state_followed_by_generated_leads():
    assert lead_count("2011-01-01", "2011-12-31") == 364
    require_inside_one_range(
        "2011-01-01", "2011-12-31", [["2011-01-01", "2014-12-31"]]
    )
    with pytest.raises(ValueError, match="not contained"):
        require_inside_one_range(
            "2014-12-31", "2015-01-02", [["2011-01-01", "2014-12-31"]]
        )


def test_legacy_semantics_are_never_inferred_as_new_model():
    assert checkpoint_semantics({}) == {
        "lag_conditioning": "full_state",
        "lag_guidance_scale": 1.0,
        "coarse_consistency_projection": False,
    }
    assert checkpoint_semantics(
        {
            "lag_conditioning": "within_block_anomaly",
            "lag_guidance_scale": 0.25,
            "enforce_coarse_consistency": True,
        }
    )["lag_conditioning"] == "within_block_anomaly"


def test_generate_rollout_chains_generated_state_without_truth_reset():
    previous = torch.full((1, 1, 4, 4), 0.5)
    conditions = torch.zeros(2, 2, 2, 2)
    conditions[:, 1] = 1.0
    mask = torch.ones(1, 1, 4, 4)
    seed = 17
    generated, diagnostics = generate_rollout(
        PreviousStateVelocity(),
        previous,
        conditions,
        mask,
        "ab2_pc",
        1,
        torch.Generator().manual_seed(seed),
        False,
        20.0,
        progress_every=10,
    )
    noise_generator = torch.Generator().manual_seed(seed)
    noise1 = masked_noise(previous, mask, noise_generator)
    expected1 = noise1 + previous
    noise2 = masked_noise(expected1, mask, noise_generator)
    expected2 = noise2 + expected1
    torch.testing.assert_close(generated[0], expected1[0])
    torch.testing.assert_close(generated[1], expected2[0])
    assert len(diagnostics["coarse_rmse_normalized_by_lead"]) == 2


def test_daily_projection_enforces_each_current_boundary():
    previous = torch.zeros(1, 1, 4, 4)
    conditions = torch.zeros(2, 2, 2, 2)
    conditions[:, 1] = 1.0
    conditions[0, 0] = 2.0
    conditions[1, 0] = -1.0
    mask = torch.ones(1, 1, 4, 4)
    generated, diagnostics = generate_rollout(
        PreviousStateVelocity(),
        previous,
        conditions,
        mask,
        "ab2_pc",
        1,
        torch.Generator().manual_seed(3),
        True,
        20.0,
        progress_every=10,
    )
    means = generated.reshape(2, 1, 2, 2, 2, 2).mean(dim=(3, 5))
    torch.testing.assert_close(means, conditions[:, :1], atol=1e-6, rtol=0)
    assert max(diagnostics["coarse_max_abs_error_normalized_by_lead"]) < 1e-6


def test_rollout_metrics_are_finite_and_lead_resolved():
    initial = np.zeros((3, 4), np.float32)
    target = np.stack((np.ones_like(initial), np.full_like(initial, 2.0)))
    generated = target + 0.5
    result = rollout_metrics(generated, target, initial)
    assert result["days"] == 2
    assert result["rmse_c_by_lead"] == pytest.approx([0.5, 0.5])
    assert np.isfinite(result["evolution_ratio"])
