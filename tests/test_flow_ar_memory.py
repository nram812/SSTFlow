"""Regression tests for the plain-flow-anchored residual-memory model."""

import hashlib

import pytest
import torch

from flow import flow_matching_loss, rollout
from model import AutoregressiveSuperResolutionFlowUNet, SuperResolutionFlowUNet
from train_flow_ar import configure_trainable_policy, initialize_from_plain_flow


def models():
    plain = SuperResolutionFlowUNet(
        base_channels=8,
        levels=2,
        attention=False,
    )
    autoregressive = AutoregressiveSuperResolutionFlowUNet(
        base_channels=8,
        levels=2,
        attention=False,
        lag_base_channels=4,
        lag_path_dropout=0.5,
        lag_guidance_scale=0.15,
        lag_conditioning="within_block_anomaly",
    )
    return plain, autoregressive


def initialize(tmp_path):
    plain, autoregressive = models()
    # Production initialization comes from a trained plain-flow EMA.  Open the
    # synthetic model's zero-initialized output path to reproduce that state.
    with torch.no_grad():
        plain.output.weight.normal_(mean=0.0, std=0.02)
    source = tmp_path / "plain_ema.pt"
    torch.save(plain.state_dict(), source)
    config = {
        "pretrained_flow_ema_path": str(source),
        "pretrained_flow_ema_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "trainable_policy": "lag_only",
    }
    provenance = initialize_from_plain_flow(autoregressive, config, tmp_path / "run")
    policy = configure_trainable_policy(autoregressive, config)
    return plain, autoregressive, provenance, policy


def test_residual_memory_starts_exactly_as_plain_flow(tmp_path):
    torch.manual_seed(20)
    plain, autoregressive, provenance, policy = initialize(tmp_path)
    plain.eval()
    autoregressive.eval()
    state = torch.randn(2, 1, 32, 32)
    condition = torch.randn(2, 2, 2, 2)
    mask = torch.ones(2, 1, 32, 32)
    previous = torch.randn_like(state)
    time = torch.tensor([0.25, 0.75])
    expected = plain(state, condition, mask, time)
    actual = autoregressive(state, condition, mask, previous, time)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert provenance["mode"] == "plain_flow_ema_backbone"
    assert policy["trainable_policy"] == "lag_only"


def test_lag_only_update_cannot_change_plain_flow_backbone(tmp_path):
    torch.manual_seed(21)
    _, model, _, policy = initialize(tmp_path)
    assert policy["trainable_parameters"] > 0
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith(("lag_encoder.", "fusion.")))
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith(("lag_encoder.", "fusion.")))
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    target = torch.randn(2, 1, 32, 32)
    condition = torch.randn(2, 2, 2, 2)
    mask = torch.ones(2, 1, 32, 32)
    previous = torch.randn_like(target)
    loss = flow_matching_loss(
        model,
        target,
        condition,
        mask,
        previous_state=previous,
        generator=torch.Generator().manual_seed(22),
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.fusion.parameters()
    )
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0.0, atol=0.0)


def test_plain_flow_initialization_refuses_wrong_checksum(tmp_path):
    plain, autoregressive = models()
    source = tmp_path / "plain.pt"
    torch.save(plain.state_dict(), source)
    with pytest.raises(ValueError, match="checksum"):
        initialize_from_plain_flow(
            autoregressive,
            {
                "pretrained_flow_ema_path": str(source),
                "pretrained_flow_ema_sha256": "0" * 64,
            },
            tmp_path / "run",
        )


def test_complete_ar_initialization_is_strict_and_retains_new_controls(tmp_path):
    _, source_model = models()
    source = tmp_path / "legacy_ar_ema.pt"
    torch.save(source_model.state_dict(), source)
    _, target_model = models()
    target_model.fusion[0].guidance_scale = 0.35
    provenance = initialize_from_plain_flow(
        target_model,
        {
            "pretrained_ar_ema_path": str(source),
            "pretrained_ar_ema_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        tmp_path / "new_run",
    )
    assert provenance["mode"] == "complete_autoregressive_ema"
    assert target_model.fusion[0].guidance_scale == 0.35
    for expected, actual in zip(source_model.parameters(), target_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_ar_and_plain_initializers_are_mutually_exclusive(tmp_path):
    _, model = models()
    with pytest.raises(ValueError, match="only one"):
        initialize_from_plain_flow(
            model,
            {
                "pretrained_ar_ema_path": "ar.pt",
                "pretrained_flow_ema_path": "flow.pt",
            },
            tmp_path / "run",
        )


class ZeroVelocity(torch.nn.Module):
    def forward(self, state, condition, mask, previous, flow_time):
        return torch.zeros_like(state)


def test_correlated_rollout_noise_preserves_marginals_and_removes_flicker():
    previous = torch.zeros(1, 1, 8, 8)
    conditions = torch.zeros(1, 4, 2, 1, 1)
    mask = torch.ones_like(previous)
    common = dict(
        model=ZeroVelocity(),
        initial_state=previous,
        conditions=conditions,
        mask=mask,
        steps=1,
        sampler="euler",
        enforce_coarse_consistency=False,
    )
    coupled = rollout(
        **common,
        generator=torch.Generator().manual_seed(25),
        noise_correlation=1.0,
    )
    independent = rollout(
        **common,
        generator=torch.Generator().manual_seed(25),
        noise_correlation=0.0,
    )
    torch.testing.assert_close(coupled[:, 1:], coupled[:, :1].expand_as(coupled[:, 1:]))
    assert not torch.equal(independent[:, 0], independent[:, 1])
    with pytest.raises(ValueError, match="noise_correlation"):
        rollout(**common, noise_correlation=1.01)
