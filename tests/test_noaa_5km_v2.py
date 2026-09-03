"""Focused tests for the frozen-trunk, learned-1024 NOAA transfer."""

from __future__ import annotations

import hashlib

import torch

import engine
from flow import sample
from model import SuperResolutionFlowUNet
from model_noaa_5km_v2 import (
    NOAAFrozenTrunkFlow,
    coastline_ocean_mask,
    high_resolution_flow_losses,
    ocean_block_mean,
)
from train_flow_noaa_5km_v2 import initialize_stage, optimizer_parameter_groups


def small_model(policy: str = "head_only") -> NOAAFrozenTrunkFlow:
    base = SuperResolutionFlowUNet(
        base_channels=8,
        levels=2,
        condition_channels=2,
        target_channels=1,
        attention=False,
    )
    base_mask = torch.ones(1, 1, 16, 16)
    base_mask[:, :, 4:8, 4:8] = 0
    return NOAAFrozenTrunkFlow(
        base,
        base_mask,
        head_width=8,
        head_blocks=1,
        trainable_policy=policy,
    )


def test_masked_downsampling_handles_partial_coast_blocks():
    values = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    mask = torch.ones_like(values)
    mask[:, :, 0, 0] = 0
    means, valid = ocean_block_mean(values, mask)
    assert means[0, 0, 0, 0] == torch.tensor((1 + 8 + 9) / 3)
    assert torch.all(valid == 1)
    mask[:, :, 2:4, 2:4] = 0
    _, valid = ocean_block_mean(values, mask)
    assert valid[0, 0, 1, 1] == 0


def test_coast_mask_is_ocean_only_and_excludes_interior():
    mask = torch.ones(1, 1, 16, 16)
    mask[:, :, 7:9, 7:9] = 0
    coast = coastline_ocean_mask(mask, radius=2)
    assert torch.count_nonzero(coast * (1 - mask)) == 0
    assert coast[0, 0, 6, 6] == 1
    assert coast[0, 0, 0, 0] == 0


def test_pretrained_output_layers_are_bypassed_and_all_base_weights_frozen():
    torch.manual_seed(1)
    model = small_model().eval()
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    state = torch.randn(1, 1, 32, 32)
    condition = torch.randn(1, 2, 2, 2)
    mask = torch.ones(1, 1, 32, 32)
    time = torch.tensor([0.4])
    first = model(state, condition, mask, time)
    with torch.no_grad():
        model.base.output.weight.fill_(1.0e6)
        model.base.output.bias.fill_(-1.0e6)
        model.base.output_norm.weight.fill_(1.0e6)
    second = model(state, condition, mask, time)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_two_updates_train_head_but_not_frozen_trunk():
    torch.manual_seed(2)
    model = small_model().train()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=2.0e-3)
    target = torch.randn(1, 1, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    mask[:, :, 10:14, 10:14] = 0
    condition = torch.randn(1, 2, 2, 2)
    head_before = model.head.learned_upsample[0].weight.detach().clone()
    base_before = {
        name: value.detach().clone() for name, value in model.base.named_parameters()
    }
    for seed in (3, 4):
        full, coast = high_resolution_flow_losses(
            model,
            target * mask,
            condition,
            mask,
            coast_radius=2,
            generator=torch.Generator().manual_seed(seed),
        )
        loss = full + 0.5 * coast
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is None
            for parameter in model.base.parameters()
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert torch.linalg.vector_norm(
        model.head.learned_upsample[0].weight.detach() - head_before
    ) > 0
    for name, value in model.base.named_parameters():
        torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)


def test_decoder_policy_trains_complete_decoder_and_keeps_encoder_frozen():
    torch.manual_seed(12)
    model = small_model("decoder_and_head").train()
    assert all(parameter.requires_grad for parameter in model.decoder_parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder_parameters())
    assert not any(parameter.requires_grad for parameter in model.base.output.parameters())
    assert not any(parameter.requires_grad for parameter in model.base.output_norm.parameters())

    groups = optimizer_parameter_groups(
        model, {"learning_rate": 2.0e-3, "decoder_learning_rate": 4.0e-4}
    )
    assert [group["name"] for group in groups] == ["head", "decoder"]
    assert [group["lr"] for group in groups] == [2.0e-3, 4.0e-4]
    optimizer = torch.optim.AdamW(groups)
    encoder_before = [parameter.detach().clone() for parameter in model.encoder_parameters()]
    # Residual blocks initialize conv2 to zero, so conv2 is the guaranteed
    # first-update probe; gradients reach conv1 after that path opens.
    decoder_probe = model.base.up[-1].conv2.weight
    decoder_before = decoder_probe.detach().clone()
    head_probe = model.head.learned_upsample[0].weight
    head_before = head_probe.detach().clone()
    target = torch.randn(1, 1, 32, 32)
    condition = torch.randn(1, 2, 2, 2)
    mask = torch.ones(1, 1, 32, 32)
    mask[:, :, 10:14, 10:14] = 0
    full, coast = high_resolution_flow_losses(
        model,
        target * mask,
        condition,
        mask,
        coast_radius=2,
        generator=torch.Generator().manual_seed(13),
    )
    (full + 0.5 * coast).backward()
    assert decoder_probe.grad is not None and torch.count_nonzero(decoder_probe.grad)
    optimizer.step()
    assert torch.linalg.vector_norm(decoder_probe.detach() - decoder_before) > 0
    assert torch.linalg.vector_norm(head_probe.detach() - head_before) > 0
    for parameter, before in zip(model.encoder_parameters(), encoder_before):
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)


def test_trainable_policy_rejects_unknown_value():
    base = SuperResolutionFlowUNet(base_channels=8, levels=2, attention=False)
    try:
        NOAAFrozenTrunkFlow(
            base,
            torch.ones(1, 1, 16, 16),
            head_width=8,
            head_blocks=1,
            trainable_policy="everything",
        )
    except ValueError as error:
        assert "trainable_policy" in str(error)
    else:
        raise AssertionError("unknown trainable policy was accepted")


def test_decoder_stage_fork_requires_exact_step_and_starts_fresh_optimizer(tmp_path):
    source_model = small_model("head_only")
    source_ema = engine.ExponentialMovingAverage(source_model, 0.9)
    source = tmp_path / "source.pt"
    torch.save(
        {
            "step": 38000,
            "module_model": source_model.state_dict(),
            "module_model_ema": source_ema.state_dict(),
        },
        source,
    )
    model = small_model("decoder_and_head")
    ema = engine.ExponentialMovingAverage(model, 0.9)
    config = {
        "initial_checkpoint": str(source),
        "initial_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "initial_step_required": 38000,
        "learning_rate": 2.0e-3,
        "decoder_learning_rate": 4.0e-4,
        "max_steps": 42000,
    }
    optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, config))
    scheduler = engine.build_scheduler(optimizer, config)
    step, history, validation, provenance = initialize_stage(
        tmp_path / "new_run",
        {"model": model, "model_ema": ema},
        {"model": optimizer},
        {"model": scheduler},
        torch.device("cpu"),
        config,
    )
    assert step == 38000
    assert history == [] and validation == {}
    assert provenance["optimizer_state_restored"] is False
    assert optimizer.state == {}
    for wanted, loaded in zip(source_model.parameters(), model.parameters()):
        torch.testing.assert_close(wanted, loaded, rtol=0, atol=0)

    config["initial_step_required"] = 37999
    with torch.no_grad():
        try:
            initialize_stage(
                tmp_path / "other_run",
                {"model": model, "model_ema": ema},
                {"model": optimizer},
                {"model": scheduler},
                torch.device("cpu"),
                config,
            )
        except ValueError as error:
            assert "required step" in str(error)
        else:
            raise AssertionError("incorrect source step was accepted")

    config["initial_step_required"] = 38000
    config["initial_checkpoint_sha256"] = "0" * 64
    try:
        initialize_stage(
            tmp_path / "checksum_mismatch",
            {"model": model, "model_ema": ema},
            {"model": optimizer},
            {"model": scheduler},
            torch.device("cpu"),
            config,
        )
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("incorrect source checksum was accepted")


def test_direct_high_resolution_heun_sample_is_finite_and_not_block_projected():
    torch.manual_seed(5)
    model = small_model().eval()
    condition = torch.randn(1, 2, 2, 2)
    mask = torch.ones(1, 1, 32, 32)
    mask[:, :, 10:14, 10:14] = 0
    generated = sample(
        model,
        condition,
        mask,
        (1, 1, 32, 32),
        steps=2,
        sampler="heun",
        generator=torch.Generator().manual_seed(6),
    )
    assert generated.shape == (1, 1, 32, 32)
    assert torch.isfinite(generated).all()
    assert torch.count_nonzero(generated * (1 - mask)) == 0
    block_means, valid = ocean_block_mean(generated, mask)
    assert torch.count_nonzero(block_means * valid) > 0
