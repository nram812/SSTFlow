import random

import numpy as np
import pytest
import torch

import engine
from model import SuperResolutionFlowUNet


def small_model(): return torch.nn.Linear(2, 1)


def test_ema_tracks_parameters():
    model = small_model(); ema = engine.ExponentialMovingAverage(model, 0.5)
    before = [p.clone() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters(): p.add_(2)
    ema.update(model)
    for old, current in zip(before, ema.module.parameters()): torch.testing.assert_close(current, old + 1)


def test_scheduler_warmup_and_decay():
    model = small_model(); optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = engine.build_scheduler(optimizer, {"warmup_steps": 2, "max_steps": 6, "min_learning_rate_factor": 0.1})
    values = [scheduler.get_last_lr()[0]]
    for _ in range(6): optimizer.step(); scheduler.step(); values.append(scheduler.get_last_lr()[0])
    assert values[0] < values[1] <= values[2] and values[-1] == pytest.approx(0.1)


def test_constant_scheduler_keeps_continuation_lr():
    model = small_model(); optimizer = torch.optim.SGD(model.parameters(), lr=5e-6)
    scheduler = engine.build_scheduler(optimizer, {"scheduler_kind": "constant"})
    for _ in range(3): optimizer.step(); scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(5e-6)


def test_check_finite_and_clip_reject_nan():
    with pytest.raises(FloatingPointError, match="Non-finite"): engine.check_finite(torch.tensor(float("nan")), 3)
    model = small_model(); optimizer = torch.optim.SGD(model.parameters(), lr=1)
    model(torch.ones(1, 2)).sum().backward(); next(model.parameters()).grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="gradient"): engine.clip_and_step(model, optimizer, 3)


def test_checkpoint_roundtrip_and_rng(tmp_path):
    model = small_model(); optimizer = torch.optim.Adam(model.parameters()); scheduler = engine.build_scheduler(optimizer, {"max_steps": 3})
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    engine.save_training_state(tmp_path, 2, {"model": model}, {"model": optimizer}, {"model": scheduler}, [{"step": 2}], {}, {"name": "x"}, {})
    expected = (random.random(), np.random.rand(), torch.rand(1)); restored = small_model()
    step, history, _ = engine.restore_training_state(tmp_path, {"model": restored}, {"model": torch.optim.Adam(restored.parameters())}, {}, torch.device("cpu"))
    assert step == 2 and history[0]["step"] == 2
    torch.testing.assert_close(model(torch.ones(1, 2)), restored(torch.ones(1, 2)))
    actual = (random.random(), np.random.rand(), torch.rand(1)); assert expected[:2] == actual[:2]; torch.testing.assert_close(expected[2], actual[2])


def test_external_checkpoint_fork_preserves_source_and_resets_scheduler(tmp_path):
    source = tmp_path / "source"; fork = tmp_path / "fork"
    model = small_model(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = engine.build_scheduler(optimizer, {"max_steps": 2, "min_learning_rate_factor": 0.05})
    optimizer.step(); scheduler.step(); optimizer.step(); scheduler.step()
    engine.save_training_state(source, 120000, {"model": model}, {"model": optimizer}, {"model": scheduler}, [], {}, {"name": "source"}, {})
    source_bytes = (source / "checkpoint.pt").read_bytes()

    restored = small_model(); fork_optimizer = torch.optim.Adam(restored.parameters(), lr=5e-6)
    fork_scheduler = engine.build_scheduler(fork_optimizer, {"scheduler_kind": "constant"})
    step, _, _ = engine.restore_training_state(
        fork, {"model": restored}, {"model": fork_optimizer}, {"model": fork_scheduler},
        torch.device("cpu"), source / "checkpoint.pt", 5e-6,
    )
    assert step == 120000
    assert fork_optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)
    fork_optimizer.step(); fork_scheduler.step()
    assert fork_scheduler.get_last_lr()[0] == pytest.approx(5e-6)
    assert (source / "checkpoint.pt").read_bytes() == source_bytes


def test_step_weight_snapshots_start_at_configured_step(tmp_path):
    model = small_model(); optimizer = torch.optim.Adam(model.parameters())
    config = {"name": "fork", "checkpoint_every": 10, "keep_step_weights": True, "snapshot_min_step": 130}
    for step in (120, 130, 131):
        engine.save_training_state(tmp_path, step, {"model": model}, {"model": optimizer}, {}, [], {}, config, {})
    assert not (tmp_path / "weights" / "model_step_000120.pt").exists()
    assert (tmp_path / "weights" / "model_step_000130.pt").is_file()
    assert not (tmp_path / "weights" / "model_step_000131.pt").exists()


def test_should_run_cadence_and_fixed_indices():
    assert not engine.should_run(10, 0) and engine.should_run(10, 5) and not engine.should_run(11, 5)
    first = engine.fixed_indices(list(range(20)), 4); np.testing.assert_array_equal(first, engine.fixed_indices(list(range(20)), 4))
