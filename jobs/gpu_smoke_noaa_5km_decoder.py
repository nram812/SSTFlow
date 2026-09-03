#!/usr/bin/env python3
"""H200 gate for the step-38k NOAA decoder-and-head continuation."""

from __future__ import annotations

import argparse
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
from train_flow_noaa_5km_v2 import configure_paths, optimizer_parameter_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/flow_sr_noaa_5km_decoder_finetune_38k.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = configure_paths(load_config(config_path))
    source = Path(config["initial_checkpoint"])
    source_sha256 = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    expected_sha256 = config.get("initial_checkpoint_sha256")
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise AssertionError(
            f"source checksum is {source_sha256}, expected {expected_sha256}"
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_step = int(checkpoint["step"])
    if source_step != int(config["initial_step_required"]):
        raise AssertionError(
            f"source step is {source_step}, expected {config['initial_step_required']}"
        )

    normalization = load_json(config["normalization_cache"])
    product = NOAATransferProduct(config["derived_path"])
    dataset = NOAATransferDataset(
        config, normalization, config["smoke_date_ranges"], product
    )
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
    model.load_state_dict(checkpoint["module_model"])
    groups = optimizer_parameter_groups(model, config)
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.99),
        weight_decay=float(config["weight_decay"]),
    )

    encoder_before = {
        name: parameter.detach().clone()
        for name, parameter in model.base.named_parameters()
        if not parameter.requires_grad
    }
    decoder_probe = model.base.up[-1].conv2.weight
    decoder_before = decoder_probe.detach().clone()
    head_probe = model.head.learned_upsample[0].weight
    head_before = head_probe.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    model.train()
    for update in range(3):
        full, coast = high_resolution_flow_losses(
            model,
            batch["target"],
            batch["condition"],
            batch["target_mask"],
            coast_radius=int(config["coastline_radius"]),
            generator=torch.Generator(device=device).manual_seed(200 + update),
        )
        total = full + float(config["coastline_loss_weight"]) * coast
        engine.check_finite(total, update, "decoder NOAA flow")
        total.backward()
        gradient_norm = engine.clip_and_step(
            model, optimizer, update, float(config["gradient_clip"])
        )
        losses.append(
            {
                "full": float(full.detach()),
                "coast": float(coast.detach()),
                "total": float(total.detach()),
                "gradient_norm": gradient_norm,
            }
        )

    decoder_delta = float((decoder_probe.detach() - decoder_before).norm())
    head_delta = float((head_probe.detach() - head_before).norm())
    if decoder_delta <= 0 or head_delta <= 0:
        raise AssertionError(
            f"expected decoder/head updates, got {decoder_delta=} {head_delta=}"
        )
    for name, parameter in model.base.named_parameters():
        if name in encoder_before and not torch.equal(parameter.detach(), encoder_before[name]):
            raise AssertionError(f"frozen encoder parameter changed: {name}")

    model.eval()
    with torch.no_grad():
        generated = sample(
            model,
            batch["condition"],
            batch["target_mask"],
            tuple(batch["target"].shape),
            steps=2,
            sampler="heun",
            generator=torch.Generator(device=device).manual_seed(210),
        )
    if not torch.isfinite(generated).all():
        raise FloatingPointError("decoder continuation Heun sample is non-finite")
    if torch.count_nonzero(generated * (1 - batch["target_mask"])):
        raise AssertionError("land changed from zero during sampling")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    report = {
        "device": torch.cuda.get_device_name(device),
        "source_checkpoint": str(source),
        "source_sha256": source_sha256,
        "source_step": source_step,
        "trainable_policy": model.trainable_policy,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_fraction": trainable / (trainable + frozen),
        "decoder_parameter_delta": decoder_delta,
        "head_parameter_delta": head_delta,
        "frozen_encoder_unchanged": True,
        "optimizer_groups": [
            {"name": group["name"], "learning_rate": group["lr"]} for group in groups
        ],
        "updates": losses,
        "finite_two_step_heun": True,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "spawned_noaa_workers": 2,
        "passed": True,
    }
    output = Path(config["smoke_output_dir"]) / "h200_report.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    dataset.close()


if __name__ == "__main__":
    main()
