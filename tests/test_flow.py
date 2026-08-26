import pytest
import torch

from flow import (ab2_sample, euler_sample, flow_matching_loss, get_sampler,
                  heun_sample, masked_noise, rollout, sample,
                  single_step_rollout_loss)
from model import AutoregressiveSuperResolutionFlowUNet, SuperResolutionFlowUNet


class ConstantVelocity(torch.nn.Module):
    def forward(self, state, condition, mask, flow_time, *args):
        return torch.ones_like(state) * mask


def inputs(batch=1):
    target = torch.randn(batch, 1, 16, 16); condition = torch.randn(batch, 2, 4, 4)
    mask = torch.ones(batch, 1, 16, 16); mask[..., :3, :3] = 0; target *= mask
    return target, condition, mask


def tiny(ar=False):
    cls = AutoregressiveSuperResolutionFlowUNet if ar else SuperResolutionFlowUNet
    kw = dict(base_channels=4, levels=2, attention=False, attention_heads=1)
    if ar: kw.update(lag_base_channels=2, lag_dropout=0.0, lag_path_dropout=0.0)
    return cls(**kw)


def test_masked_noise_is_zero_on_land():
    target, _, mask = inputs(); assert torch.count_nonzero(masked_noise(target, mask)[mask == 0]) == 0


@pytest.mark.parametrize("ar", [False, True])
def test_loss_is_finite_and_gradient_flows(ar):
    target, condition, mask = inputs(); model = tiny(ar)
    loss = flow_matching_loss(model, target, condition, mask, target if ar else None)
    assert torch.isfinite(loss) and loss > 0; loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("sampler", [euler_sample, heun_sample, ab2_sample])
@pytest.mark.parametrize("steps", [1, 2, 5])
def test_samplers_shape_finite_and_land(sampler, steps):
    target, condition, mask = inputs(); result = sampler(ConstantVelocity(), target, condition, mask, steps)
    assert result.shape == target.shape and torch.isfinite(result).all()
    assert torch.count_nonzero(result[mask == 0]) == 0


def test_sampler_determinism_with_seed():
    target, condition, mask = inputs(); model = tiny()
    first = sample(model, condition, mask, target.shape, 2, generator=torch.Generator().manual_seed(2))
    second = sample(model, condition, mask, target.shape, 2, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_heun_matches_constant_velocity():
    target, condition, mask = inputs(); initial = torch.zeros_like(target)
    torch.testing.assert_close(heun_sample(ConstantVelocity(), initial, condition, mask, 5), mask)


def test_invalid_sampler_arguments_raise():
    target, condition, mask = inputs()
    with pytest.raises(ValueError, match="at least one"): euler_sample(ConstantVelocity(), target, condition, mask, 0)
    with pytest.raises(ValueError, match="Unknown sampler"): get_sampler("bad")


def test_rollout_shape_and_chaining():
    target, condition, mask = inputs(); conditions = condition[:, None].repeat(1, 3, 1, 1, 1)
    result = rollout(tiny(True), target, conditions, mask, steps=1)
    assert result.shape == (1, 3, 1, 16, 16) and torch.isfinite(result).all()


def test_single_step_rollout_loss_backward():
    target, condition, mask = inputs(); model = tiny(True)
    loss, prediction = single_step_rollout_loss(model, target, condition, target, mask, steps=1)
    assert torch.isfinite(loss) and prediction.shape == target.shape
    loss.backward(); assert any(p.grad is not None for p in model.parameters())
