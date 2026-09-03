#!/usr/bin/env python3
"""H200 gate for the coarse-balanced legacy Flow-AR fine-tune."""

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
    config = load_config(root / "configs/flow_ar_legacy_coarse_balanced.json")
    legacy_config = load_config(root / "runs/flow_ar/config_used.json")
    device = torch.device("cuda")
    _, _, _, normalization, derived = engine.prepare(config, True, "cuda")
    training, validation = engine.make_datasets(
        config, normalization, derived, True, "autoregressive"
    )
    batch = engine.batch_to_device(
        next(iter(engine.make_loader(training, 4, 93, num_workers=2))), device
    )

    model = build_model(config)
    provenance = initialize_from_plain_flow(
        model, config, root / "runs/__coarse_balanced_initialization_gate__"
    )
    policy = configure_trainable_policy(model, config)
    model = model.to(device)
    legacy = build_model(legacy_config).to(device)
    legacy.load_state_dict(
        torch.load(config["pretrained_ar_ema_path"], map_location=device, weights_only=True)
    )
    legacy.eval()
    model.eval()

    state = torch.randn_like(batch["target"]) * batch["mask"]
    flow_time = torch.full((len(state),), 0.4, device=device)
    zero_previous = torch.zeros_like(batch["previous"])
    with torch.no_grad():
        legacy_with_lag = legacy(
            state, batch["condition"], batch["mask"], batch["previous"], flow_time
        )
        legacy_without_lag = legacy(
            state, batch["condition"], batch["mask"], zero_previous, flow_time
        )
        balanced_with_lag = model(
            state, batch["condition"], batch["mask"], batch["previous"], flow_time
        )
        balanced_without_lag = model(
            state, batch["condition"], batch["mask"], zero_previous, flow_time
        )
    legacy_lag_effect = torch.linalg.vector_norm(legacy_with_lag - legacy_without_lag)
    balanced_lag_effect = torch.linalg.vector_norm(
        balanced_with_lag - balanced_without_lag
    )
    lag_effect_ratio = float(balanced_lag_effect / legacy_lag_effect)
    if not 0.0 < lag_effect_ratio < 1.0:
        raise AssertionError(
            f"0.35 FiLM cap did not reduce legacy lag effect: {lag_effect_ratio}"
        )

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            generator=torch.Generator(device=device).manual_seed(200 + update),
        )
        engine.check_finite(loss, update, "coarse-balanced velocity")
        loss.backward()
        norm = engine.clip_and_step(model, optimizer, update, 1.0)
        updates.append({"loss": float(loss.detach()), "gradient_norm": norm})
    changed_tensors = sum(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
    )
    if changed_tensors == 0:
        raise AssertionError("no model tensor changed")

    model.eval()
    conditions = batch["condition"][:1, None].expand(-1, 5, -1, -1, -1).clone()
    conditions[:, :, 0] += torch.linspace(0, 0.04, 5, device=device)[None, :, None, None]
    with torch.no_grad():
        generated = rollout(
            model,
            batch["previous"][:1],
            conditions,
            batch["mask"][:1],
            steps=10,
            sampler="heun",
            generator=torch.Generator(device=device).manual_seed(240),
            enforce_coarse_consistency=True,
        )
    if not torch.isfinite(generated).all():
        raise FloatingPointError("coarse-balanced rollout is non-finite")
    maximum_coarse_error = max(
        float(coarse_consistency_mse(
            generated[:, lead], conditions[:, lead], batch["mask"][:1]
        ))
        for lead in range(5)
    )
    if maximum_coarse_error > 1.0e-10:
        raise AssertionError(f"coarse authority failed: {maximum_coarse_error}")

    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "initialization": provenance,
        **policy,
        "legacy_lag_effect_norm": float(legacy_lag_effect),
        "balanced_lag_effect_norm": float(balanced_lag_effect),
        "balanced_to_legacy_lag_effect_ratio": lag_effect_ratio,
        "changed_trainable_tensors": changed_tensors,
        "updates": updates,
        "finite_five_day_rollout": True,
        "maximum_coarse_consistency_mse": maximum_coarse_error,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "elapsed_seconds": time.perf_counter() - started,
        "spawned_workers": 2,
    }
    output = root / "runs/smoke/flow_ar_legacy_coarse_balanced/h200_report.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    training.close()
    validation.close()


if __name__ == "__main__":
    main()
