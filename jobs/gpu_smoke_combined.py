#!/usr/bin/env python3
"""H200 gate for the historical+RCP8.5 flow continuation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import engine
from common import atomic_json, attach_ocean_mask, load_config
from data import DerivedProduct, build_dataset
from flow import flow_matching_loss, sample
from model import build_model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/flow_sr_combined_hist_rcp85_continue_320k.json")
    normalization = attach_ocean_mask(
        engine.load_normalization(config),
        DerivedProduct(config["derived_path"]).ocean_mask,
    )
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    dataset = build_dataset(
        config, normalization, config["train_date_ranges"],
        "super_resolution", derived=derived,
    )
    validation = build_dataset(
        config, normalization, config["validation_date_ranges"],
        "super_resolution", derived=derived,
    )
    device = torch.device("cuda")

    # Exercise the actual spawned multi-source DataLoader contract.
    loader = engine.make_loader(dataset, 4, int(config["seed"]), num_workers=2)
    worker_batch = next(iter(loader))
    if not all(torch.isfinite(worker_batch[key]).all() for key in ("target", "condition", "mask")):
        raise FloatingPointError("Multi-source worker batch is not finite")

    historical_positions = np.flatnonzero(derived.source_id[dataset.indices] == 0)
    future_positions = np.flatnonzero(derived.source_id[dataset.indices] == 1)
    chosen = np.asarray([
        historical_positions[0], historical_positions[len(historical_positions) // 2],
        historical_positions[-1], future_positions[0],
        future_positions[len(future_positions) // 3],
        future_positions[2 * len(future_positions) // 3], future_positions[-2],
        future_positions[-1],
    ])
    batch = engine.batch_to_device(engine.collate_indices(dataset, chosen), device)

    state = torch.load(config["resume_from"], map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(state["module_model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["continuation_learning_rate"]),
        weight_decay=float(config["weight_decay"]), betas=(0.9, 0.99),
    )
    optimizer.load_state_dict(state["optimizer_model"])
    for group in optimizer.param_groups:
        group["lr"] = float(config["continuation_learning_rate"])
        group["initial_lr"] = float(config["continuation_learning_rate"])
    before = model.input_block.conv1.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    loss = flow_matching_loss(
        model, batch["target"], batch["condition"], batch["mask"],
        generator=torch.Generator(device=device).manual_seed(19),
    )
    engine.check_finite(loss, 220000, "combined flow loss")
    loss.backward()
    gradient_norm = engine.clip_and_step(model, optimizer, 220000, 1.0)
    parameter_delta = float((model.input_block.conv1.weight.detach() - before).norm())
    if parameter_delta <= 0:
        raise AssertionError("Combined continuation optimizer did not update the model")

    # Sample both climate periods with the completed EMA initialization.
    ema = build_model(config).to(device).eval()
    ema.load_state_dict(state["module_model_ema"])
    callback_positions = np.asarray([0, len(validation) - 1])
    callback = engine.batch_to_device(
        engine.collate_indices(validation, callback_positions), device
    )
    with torch.no_grad():
        generated = sample(
            ema, callback["condition"], callback["mask"], callback["target"].shape,
            steps=5, sampler="heun",
            generator=torch.Generator(device=device).manual_seed(23), device=device,
        )
    if not torch.isfinite(generated).all():
        raise FloatingPointError("Combined historical/future sample is not finite")
    ocean = callback["mask"].bool().expand_as(generated)
    normalized_range = [
        float(generated[ocean].min()), float(generated[ocean].max())
    ]
    report = {
        "device": torch.cuda.get_device_name(device),
        "checkpoint_step": int(state["step"]),
        "training_days": len(dataset),
        "validation_days": len(validation),
        "historical_training_days": int(len(historical_positions)),
        "future_training_days": int(len(future_positions)),
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "parameter_delta": parameter_delta,
        "sample_dates": validation.dates(callback_positions),
        "sample_normalized_range": normalized_range,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "worker_batch_finite": True,
    }
    output = root / "runs/smoke/flow_sr_combined_hist_rcp85_continue_320k/report.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    dataset.close(); validation.close()


if __name__ == "__main__":
    main()
