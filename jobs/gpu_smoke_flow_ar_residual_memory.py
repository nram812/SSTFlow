#!/usr/bin/env python3
"""H200 gate for the plain-flow-anchored residual-memory AR model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import engine
from common import atomic_json, load_config
from consistency import coarse_consistency_mse
from flow import flow_matching_loss, rollout
from model import build_model
from train_flow_ar import configure_trainable_policy, initialize_from_plain_flow


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/flow_ar_residual_memory.json")
    device = torch.device("cuda")
    _, _, _, normalization, derived = engine.prepare(config, True, "cuda")
    training, validation = engine.make_datasets(
        config, normalization, derived, True, "autoregressive"
    )
    worker_batch = next(iter(engine.make_loader(training, 4, 83, num_workers=2)))
    batch = engine.batch_to_device(worker_batch, device)

    model = build_model(config)
    provenance = initialize_from_plain_flow(model, config, root / "runs/__no_local__")
    policy = configure_trainable_policy(model, config)
    model = model.to(device)
    plain = build_model({**config, "model_kind": "super_resolution"}).to(device)
    plain.load_state_dict(
        torch.load(config["pretrained_flow_ema_path"], map_location=device, weights_only=True)
    )
    plain.eval()
    model.eval()
    flow_time = torch.full((len(batch["target"]),), 0.4, device=device)
    state = torch.randn_like(batch["target"]) * batch["mask"]
    with torch.no_grad():
        plain_velocity = plain(state, batch["condition"], batch["mask"], flow_time)
        ar_velocity = model(
            state,
            batch["condition"],
            batch["mask"],
            batch["previous"],
            flow_time,
        )
    torch.testing.assert_close(ar_velocity, plain_velocity, rtol=0.0, atol=0.0)

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    lag_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    updates = []
    model.train()
    for update in range(5):
        loss = flow_matching_loss(
            model,
            batch["target"],
            batch["condition"],
            batch["mask"],
            previous_state=batch["previous"],
            generator=torch.Generator(device=device).manual_seed(100 + update),
        )
        engine.check_finite(loss, update, "residual-memory velocity")
        loss.backward()
        norm = engine.clip_and_step(model, optimizer, update, 1.0)
        updates.append({"loss": float(loss.detach()), "gradient_norm": norm})
    changed_lag = sum(
        not torch.equal(parameter.detach(), lag_before[name])
        for name, parameter in model.named_parameters()
        if name in lag_before
    )
    if changed_lag == 0:
        raise AssertionError("no residual-memory tensor changed")
    for name, parameter in model.named_parameters():
        if name in frozen_before and not torch.equal(parameter.detach(), frozen_before[name]):
            raise AssertionError(f"frozen plain-flow tensor changed: {name}")

    model.eval()
    conditions = batch["condition"][:1, None].expand(-1, 5, -1, -1, -1).clone()
    # Exercise an evolving authoritative boundary rather than a constant input.
    conditions[:, :, 0] += torch.linspace(0, 0.04, 5, device=device)[None, :, None, None]
    with torch.no_grad():
        generated = rollout(
            model,
            batch["previous"][:1],
            conditions,
            batch["mask"][:1],
            steps=10,
            sampler="heun",
            generator=torch.Generator(device=device).manual_seed(120),
            enforce_coarse_consistency=True,
        )
    if not torch.isfinite(generated).all():
        raise FloatingPointError("residual-memory rollout is non-finite")
    maximum_coarse_error = max(
        float(
            coarse_consistency_mse(
                generated[:, lead], conditions[:, lead], batch["mask"][:1]
            )
        )
        for lead in range(5)
    )
    if maximum_coarse_error > 1e-10:
        raise AssertionError(f"coarse authority failed: {maximum_coarse_error}")

    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "initialization": provenance,
        **policy,
        "exact_plain_flow_at_initialization": True,
        "changed_lag_tensors": changed_lag,
        "frozen_backbone_bitwise_unchanged": True,
        "updates": updates,
        "finite_five_day_rollout": True,
        "maximum_coarse_consistency_mse": maximum_coarse_error,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "elapsed_seconds": time.perf_counter() - started,
        "spawned_workers": 2,
    }
    output = root / "runs/smoke/flow_ar_residual_memory/h200_report.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    training.close()
    validation.close()


if __name__ == "__main__":
    main()
