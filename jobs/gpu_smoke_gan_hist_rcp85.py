#!/usr/bin/env python3
"""H200 gate for combined-period GAN continuation and direct inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import engine
from common import atomic_json, attach_ocean_mask, load_config
from consistency import coarse_consistency_mse
from data import DerivedProduct, build_dataset
from infer_access_cm2 import (
    make_condition,
    validate_converted_grid,
    validate_converted_values,
)
from losses import (
    feature_matching_loss,
    hinge_discriminator_loss,
    hinge_generator_loss,
    masked_gradient_loss,
    masked_mse,
    spectral_amplitude_loss,
)
from model_gan import build_discriminator, build_generator
from train_gan import set_requires_grad
import xarray as xr


CONFIGS = (
    "gan_sr_v2_hist_rcp85_continue_220k.json",
    "gan_sr_v2b_hist_rcp85_continue_220k.json",
    "gan_sr_v3_hist_rcp85_continue_220k.json",
)


def one_variant(root: Path, config_name: str, device: torch.device) -> dict:
    config = load_config(root / "configs" / config_name)
    derived = DerivedProduct(config["derived_path"])
    normalization = attach_ocean_mask(
        engine.load_normalization(config), derived.ocean_mask
    )
    derived.verify(normalization)
    dataset = build_dataset(
        config, normalization, config["train_date_ranges"],
        "super_resolution", derived=derived,
    )
    historical = np.flatnonzero(derived.source_id[dataset.indices] == 0)
    future = np.flatnonzero(derived.source_id[dataset.indices] == 1)
    chosen = np.asarray([
        historical[0], historical[len(historical) // 2],
        future[0], future[len(future) // 2],
    ])
    batch = engine.batch_to_device(engine.collate_indices(dataset, chosen), device)

    state = torch.load(config["resume_from"], map_location=device, weights_only=False)
    if int(state["step"]) != 120000:
        raise AssertionError(f"Unexpected source checkpoint step {state['step']}")
    generator = build_generator(config).to(device)
    discriminator = build_discriminator(config).to(device)
    generator.load_state_dict(state["module_generator"])
    discriminator.load_state_dict(state["module_discriminator"])
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=float(config["continuation_learning_rate"]),
        betas=tuple(config["generator_betas"]),
        weight_decay=float(config["weight_decay"]),
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=float(config["continuation_learning_rate"]),
        betas=tuple(config["discriminator_betas"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator_optimizer.load_state_dict(state["optimizer_generator"])
    discriminator_optimizer.load_state_dict(state["optimizer_discriminator"])
    for optimizer in (generator_optimizer, discriminator_optimizer):
        for group in optimizer.param_groups:
            group["lr"] = float(config["continuation_learning_rate"])
            group["initial_lr"] = float(config["continuation_learning_rate"])

    before_discriminator = next(discriminator.parameters()).detach().clone()
    with torch.no_grad():
        fake = generator(batch["condition"], batch["mask"])
    critic_loss = hinge_discriminator_loss(
        discriminator(batch["target"], batch["condition"], batch["mask"]),
        discriminator(fake, batch["condition"], batch["mask"]),
    )
    engine.check_finite(critic_loss, 120000, "combined GAN critic loss")
    critic_loss.backward()
    critic_gradient = engine.clip_and_step(
        discriminator, discriminator_optimizer, 120000, float(config["gradient_clip"])
    )
    discriminator_delta = float(
        (next(discriminator.parameters()).detach() - before_discriminator).norm()
    )

    before_generator = generator.stem.weight.detach().clone()
    generated = generator(batch["condition"], batch["mask"])
    content = masked_mse(generated, batch["target"], batch["mask"])
    gradient = masked_gradient_loss(generated, batch["target"], batch["mask"])
    spectral = spectral_amplitude_loss(generated, batch["target"], batch["mask"])
    set_requires_grad(discriminator, False)
    fake_logits, fake_features, feature_masks = discriminator(
        generated, batch["condition"], batch["mask"], return_features=True
    )
    with torch.no_grad():
        _, real_features, _ = discriminator(
            batch["target"], batch["condition"], batch["mask"], return_features=True
        )
    adversarial = hinge_generator_loss(fake_logits)
    feature = feature_matching_loss(fake_features, real_features, feature_masks)
    generator_loss = (
        float(config["lambda_content"]) * content
        + float(config["lambda_gradient"]) * gradient
        + float(config["lambda_spectral"]) * spectral
        + float(config["lambda_feature_matching"]) * feature
        + float(config["lambda_adversarial"]) * adversarial
    )
    engine.check_finite(generator_loss, 120000, "combined GAN generator loss")
    generator_loss.backward()
    generator_gradient = engine.clip_and_step(
        generator, generator_optimizer, 120000, float(config["gradient_clip"])
    )
    generator_delta = float((generator.stem.weight.detach() - before_generator).norm())
    if generator_delta <= 0 or discriminator_delta <= 0:
        raise AssertionError("A GAN player did not update in the continuation smoke step")

    generator.eval()
    condition, mask = batch["condition"][:1], batch["mask"][:1]
    with torch.no_grad():
        first = generator(
            condition, mask,
            noise=torch.randn((1, int(config["noise_channels"]), *derived.coarse_shape), device=device),
        )
        second = generator(
            condition, mask,
            noise=torch.randn((1, int(config["noise_channels"]), *derived.coarse_shape), device=device),
        )
    noise_response = float(((first - second) * mask).square().mean().sqrt())
    if noise_response <= 1.0e-6:
        raise AssertionError("Generator does not respond to latent noise")
    consistency = float(coarse_consistency_mse(first, condition, mask))
    if config.get("enforce_coarse_consistency") and consistency > 1.0e-10:
        raise AssertionError(f"GAN-v3 hard constraint failed: {consistency}")

    # The actual spawned multi-source loader must read both files safely.
    loader = engine.make_loader(dataset, 4, int(config["seed"]), num_workers=2)
    worker_batch = next(iter(loader))
    if not all(torch.isfinite(worker_batch[key]).all() for key in ("target", "condition", "mask")):
        raise FloatingPointError("Spawned multi-source batch is not finite")

    dataset.close()
    return {
        "source_checkpoint_step": int(state["step"]),
        "training_days": int(len(historical) + len(future)),
        "historical_training_days": int(len(historical)),
        "future_training_days": int(len(future)),
        "sample_dates": dataset.dates(chosen),
        "critic_loss": float(critic_loss.detach()),
        "generator_loss": float(generator_loss.detach()),
        "critic_gradient_norm": critic_gradient,
        "generator_gradient_norm": generator_gradient,
        "critic_parameter_delta": discriminator_delta,
        "generator_parameter_delta": generator_delta,
        "noise_response_rms_normalized": noise_response,
        "coarse_consistency_mse_normalized": consistency,
    }


def access_benchmark(root: Path, device: torch.device) -> dict:
    run_dir = root / "runs/gan_sr_v2"
    config = load_config(root / "configs/gan_sr_v2.json")
    derived = DerivedProduct(config["derived_path"])
    normalization = engine.load_normalization(config)
    model = build_generator(config).to(device).eval()
    model.load_state_dict(torch.load(
        run_dir / "generator_ema.pt", map_location=device, weights_only=True
    ))
    source_path = root / "derived/sst_downscaling_access_converted.nc"
    with xr.open_dataset(source_path, engine="h5netcdf") as source:
        field = validate_converted_grid(source, derived, "sst_lr")
        coarse = validate_converted_values(field.isel(time=slice(0, 32)).values,
                                           derived.ocean_mask_lr)
    condition = torch.from_numpy(make_condition(
        coarse, derived.ocean_mask_lr,
        normalization["sst_mean"], normalization["sst_std"],
    )).to(device)
    mask = torch.from_numpy(derived.ocean_mask[None, None].astype(np.float32)).to(device)
    mask = mask.expand(len(condition), -1, -1, -1)
    noise = torch.randn((len(condition), int(config["noise_channels"]), *derived.coarse_shape),
                        device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(condition, mask, noise=noise)
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated = model(condition, mask, noise=noise)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(generated).all():
        raise FloatingPointError("Direct ACCESS GAN inference is not finite")
    return {
        "batch_size": 32,
        "elapsed_seconds": elapsed,
        "fields_per_second": 32.0 / elapsed,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda")
    variants = {name: one_variant(root, name, device) for name in CONFIGS}
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "variants": variants,
        "direct_access_benchmark": access_benchmark(root, device),
    }
    output = root / "runs/smoke/gan_hist_rcp85/report.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
