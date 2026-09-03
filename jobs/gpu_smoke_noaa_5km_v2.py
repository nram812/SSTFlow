#!/usr/bin/env python3
"""H200 acceptance gate for the direct frozen-trunk NOAA flow."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import engine
from common import atomic_json, load_config, load_json
from data_noaa_5km import NOAATransferDataset, NOAATransferProduct
from flow import sample
from model_noaa_5km_v2 import NOAAFrozenTrunkFlow, high_resolution_flow_losses
from train_flow_noaa_5km_v2 import configure_paths


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    config = configure_paths(
        load_config(root / "configs/flow_sr_noaa_5km_frozen_trunk_1024.json")
    )
    normalization = load_json(config["normalization_cache"])
    product = NOAATransferProduct(config["derived_path"])
    dataset = NOAATransferDataset(
        config, normalization, config["smoke_date_ranges"], product
    )
    # Verify normal production multiprocessing, not just main-process reads.
    worker_batch = next(iter(engine.make_loader(dataset, 1, 91, num_workers=2)))
    for key in ("target", "condition", "target_mask", "base_mask"):
        if not torch.isfinite(worker_batch[key]).all():
            raise FloatingPointError(f"non-finite spawned NOAA batch: {key}")

    device = torch.device("cuda")
    batch = engine.batch_to_device(worker_batch, device)
    model = NOAAFrozenTrunkFlow.from_pretrained(
        config,
        torch.from_numpy(product.base_mask.astype(np.float32)),
        device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["learning_rate"])
    frozen_before = {
        name: parameter.detach().clone() for name, parameter in model.base.named_parameters()
    }
    head_before = model.head.learned_upsample[0].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for update in range(3):
        full, coast = high_resolution_flow_losses(
            model,
            batch["target"],
            batch["condition"],
            batch["target_mask"],
            coast_radius=config["coastline_radius"],
            generator=torch.Generator(device=device).manual_seed(100 + update),
        )
        total = full + config["coastline_loss_weight"] * coast
        engine.check_finite(total, update, "direct NOAA flow")
        total.backward()
        gradient_norm = engine.clip_and_step(
            model, optimizer, update, config["gradient_clip"]
        )
        losses.append(
            {
                "full": float(full.detach()),
                "coast": float(coast.detach()),
                "total": float(total.detach()),
                "gradient_norm": gradient_norm,
            }
        )
    head_delta = float(
        (model.head.learned_upsample[0].weight.detach() - head_before).norm()
    )
    if head_delta <= 0:
        raise AssertionError("updates did not reach the learned 512-to-1024 block")
    for name, parameter in model.base.named_parameters():
        if not torch.equal(parameter.detach(), frozen_before[name]):
            raise AssertionError(f"frozen pretrained parameter changed: {name}")

    model.eval()
    with torch.no_grad():
        generated = sample(
            model,
            batch["condition"],
            batch["target_mask"],
            tuple(batch["target"].shape),
            steps=2,
            sampler="heun",
            generator=torch.Generator(device=device).manual_seed(110),
        )
    if not torch.isfinite(generated).all():
        raise FloatingPointError("direct 1024-square Heun sample is non-finite")
    if torch.count_nonzero(generated * (1 - batch["target_mask"])):
        raise AssertionError("land changed from zero during sampling")
    report = {
        "device": torch.cuda.get_device_name(device),
        "pretrained_ema": config["pretrained_ema_path"],
        "target_shape": list(product.target_shape),
        "base_shape": list(product.base_shape),
        "updates": losses,
        "learned_upsample_parameter_delta": head_delta,
        "frozen_parameters_unchanged": True,
        "pretrained_output_layers_bypassed": True,
        "finite_two_step_heun": True,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "spawned_noaa_workers": 2,
        "passed": True,
    }
    output = root / "runs/smoke/flow_sr_noaa_5km_frozen_trunk_1024/h200_report.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    dataset.close()


if __name__ == "__main__":
    main()
