"""C5: architecture shape, stability, and conditioning tests."""

from __future__ import annotations

import pytest
import torch

from model import (
    AutoregressiveSuperResolutionFlowUNet,
    SuperResolutionFlowUNet,
    build_model,
    parameter_count,
)

GRID = 64
FACTOR = 8
COARSE = GRID // FACTOR


def make_inputs(batch=2, grid=GRID, coarse=COARSE):
    torch.manual_seed(0)
    state = torch.randn(batch, 1, grid, grid)
    condition = torch.randn(batch, 2, coarse, coarse)
    mask = torch.ones(batch, 1, grid, grid)
    mask[:, :, : grid // 8, : grid // 8] = 0.0
    flow_time = torch.rand(batch)
    return state, condition, mask, flow_time


def make_sr(levels=3, base=8):
    return SuperResolutionFlowUNet(
        base_channels=base, levels=levels, attention=True, attention_heads=2
    )


def make_ar(levels=3, base=8, path_dropout=0.5):
    return AutoregressiveSuperResolutionFlowUNet(
        base_channels=base,
        levels=levels,
        attention=True,
        attention_heads=2,
        lag_base_channels=4,
        lag_path_dropout=path_dropout,
    )


def test_sr_forward_shape():
    state, condition, mask, t = make_inputs()
    assert make_sr()(state, condition, mask, t).shape == state.shape


def test_ar_forward_shape():
    state, condition, mask, t = make_inputs()
    output = make_ar()(state, condition, mask, torch.randn_like(state), t)
    assert output.shape == state.shape


def test_zero_init_output_at_step_zero():
    state, condition, mask, t = make_inputs()
    assert float(make_sr()(state, condition, mask, t).abs().max()) == 0.0
    assert float(
        make_ar()(state, condition, mask, torch.randn_like(state), t).abs().max()
    ) == 0.0


def test_forward_is_finite_with_extreme_inputs():
    state, condition, mask, t = make_inputs()
    model = make_sr()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    output = model(state * 10.0, condition * 10.0, mask, t)
    assert torch.isfinite(output).all()


def test_backward_produces_finite_grads():
    state, condition, mask, t = make_inputs()
    model = make_sr()
    model(state, condition, mask, t).square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def _train_one_step(model, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss = model(*args).square().mean() + sum(
        p.square().sum() for p in model.parameters()
    ) * 0.0
    # A zero-init model has zero output; perturb the parameters so the model
    # becomes sensitive to its inputs.
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    optimizer.zero_grad()
    return model


def test_condition_actually_matters():
    state, condition, mask, t = make_inputs()
    model = _train_one_step(make_sr(), (state, condition, mask, t))
    a = model(state, condition, mask, t)
    b = model(state, condition + 3.0, mask, t)
    assert float((a - b).abs().max()) > 1e-6


def test_previous_state_matters():
    state, condition, mask, t = make_inputs()
    previous = torch.randn_like(state)
    model = _train_one_step(make_ar(path_dropout=0.0), (state, condition, mask, previous, t))
    model.eval()
    a = model(state, condition, mask, previous, t)
    b = model(state, condition, mask, previous + 3.0, t)
    assert float((a - b).abs().max()) > 1e-6


def test_lag_path_dropout_is_active_in_train_only():
    state, condition, mask, t = make_inputs()
    previous = torch.randn_like(state)
    model = _train_one_step(make_ar(path_dropout=0.5), (state, condition, mask, previous, t))
    model.eval()
    first = model(state, condition, mask, previous, t)
    second = model(state, condition, mask, previous, t)
    torch.testing.assert_close(first, second)
    model.train()
    torch.manual_seed(1)
    a = model(state, condition, mask, previous, t)
    torch.manual_seed(2)
    b = model(state, condition, mask, previous, t)
    assert float((a - b).abs().max()) > 0.0


def test_wrong_channel_count_raises():
    state, condition, mask, t = make_inputs()
    model = make_sr()
    with pytest.raises(ValueError, match="condition channels"):
        model(state, condition[:, :1], mask, t)
    with pytest.raises(ValueError, match="state channels"):
        model(torch.randn(2, 3, GRID, GRID), condition, mask, t)


def test_wrong_mask_grid_raises():
    state, condition, _, t = make_inputs()
    with pytest.raises(ValueError, match="Ocean mask grid"):
        make_sr()(state, condition, torch.ones(2, 1, 32, 32), t)


def test_previous_shape_mismatch_raises():
    state, condition, mask, t = make_inputs()
    with pytest.raises(ValueError, match="Previous state"):
        make_ar()(state, condition, mask, torch.randn(2, 1, 32, 32), t)


@pytest.mark.parametrize("grid", [32, 64, 128])
def test_multiple_grid_sizes(grid):
    state, condition, mask, t = make_inputs(batch=1, grid=grid, coarse=grid // FACTOR)
    assert make_sr()(state, condition, mask, t).shape == state.shape


def test_build_model_factory():
    assert isinstance(
        build_model({"model_kind": "super_resolution", "base_channels": 8, "levels": 3}),
        SuperResolutionFlowUNet,
    )
    assert isinstance(
        build_model({"model_kind": "autoregressive", "base_channels": 8, "levels": 3}),
        AutoregressiveSuperResolutionFlowUNet,
    )
    with pytest.raises(ValueError, match="Unknown model_kind"):
        build_model({"model_kind": "nope"})


def test_parameter_count_reasonable():
    production = build_model(
        {"model_kind": "super_resolution", "base_channels": 32, "levels": 4}
    )
    count = parameter_count(production)
    assert 1e6 < count < 1e8, count


def test_time_embedding_odd_dimension_raises():
    with pytest.raises(ValueError, match="even"):
        from model import TimeEmbedding

        TimeEmbedding(7)
