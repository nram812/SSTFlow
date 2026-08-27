#!/usr/bin/env python3
"""H200 acceptance test: production batches, sampler timing, and NetCDF I/O."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import engine
from callbacks import coarse_to_physical, save_netcdf, to_physical
from common import atomic_json, attach_ocean_mask, load_config
from consistency import coarse_consistency_mse
from data import DerivedProduct, build_dataset
from flow import flow_matching_loss, rollout, sample
from losses import hinge_discriminator_loss, hinge_generator_loss, masked_mse
from model import build_model
from model_gan import build_discriminator, build_generator


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; gpu_smoke must run on h200q")
    device = torch.device("cuda"); root = Path(__file__).resolve().parents[1]
    output = root / "runs" / "gpu_smoke"; output.mkdir(parents=True, exist_ok=True)
    report = {"device": torch.cuda.get_device_name(device), "torch": torch.__version__}
    configs = {name: load_config(root / "configs" / filename) for name, filename in
               (("flow", "flow_sr.json"), ("flow_ar", "flow_ar.json"), ("gan", "gan_sr.json"))}
    normalization = engine.load_normalization(configs["flow"])
    derived = DerivedProduct(configs["flow"]["derived_path"])
    normalization = attach_ocean_mask(normalization, derived.ocean_mask)

    for name in ("flow", "flow_ar"):
        config = configs[name]; kind = "autoregressive" if name == "flow_ar" else "super_resolution"
        dataset = build_dataset(config, normalization, config["smoke_date_ranges"], kind, derived=derived)
        batch = engine.batch_to_device(engine.collate_indices(dataset, np.arange(int(config["batch_size"]))), device)
        model = build_model(config).to(device); torch.cuda.reset_peak_memory_stats()
        loss = flow_matching_loss(model, batch["target"], batch["condition"], batch["mask"], batch.get("previous"))
        engine.check_finite(loss, 0, name); loss.backward(); torch.cuda.synchronize()
        report[name] = {"loss": float(loss), "peak_memory_mb": peak_mb(), "batch_size": int(config["batch_size"])}
        del loss, batch, model, dataset; torch.cuda.empty_cache()

    config = configs["gan"]; dataset = build_dataset(config, normalization, config["smoke_date_ranges"], "super_resolution", derived=derived)
    batch = engine.batch_to_device(engine.collate_indices(dataset, np.arange(int(config["batch_size"]))), device)
    generator = build_generator(config).to(device); discriminator = build_discriminator(config).to(device)
    torch.cuda.reset_peak_memory_stats(); fake = generator(batch["condition"], batch["mask"])
    loss = 10 * masked_mse(fake, batch["target"], batch["mask"]) + hinge_generator_loss(discriminator(fake, batch["condition"], batch["mask"]))
    loss = loss + hinge_discriminator_loss(discriminator(batch["target"], batch["condition"], batch["mask"]), discriminator(fake.detach(), batch["condition"], batch["mask"]))
    engine.check_finite(loss, 0, "gan"); loss.backward(); torch.cuda.synchronize()
    report["gan"] = {"loss": float(loss), "peak_memory_mb": peak_mb(), "batch_size": int(config["batch_size"])}
    del loss, fake, generator, discriminator, batch, dataset; torch.cuda.empty_cache()

    config = configs["flow"]; dataset = build_dataset(config, normalization, config["smoke_date_ranges"], "super_resolution", derived=derived)
    raw = engine.collate_indices(dataset, [0]); batch = engine.batch_to_device(raw, device); model = build_model(config).to(device).eval()
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    with torch.no_grad(): generated = sample(model, batch["condition"], batch["mask"], batch["target"].shape, 25, "heun", device=device)
    torch.cuda.synchronize(); report["heun_25"] = {"seconds": time.perf_counter() - started, "peak_memory_mb": peak_mb()}
    generated_np = to_physical(generated, normalization, derived.ocean_mask)[:, 0]
    target_np = to_physical(batch["target"], normalization, derived.ocean_mask)[:, 0]
    coarse_np = coarse_to_physical(batch["condition"], normalization, derived.ocean_mask_lr)
    product = save_netcdf(generated_np, target_np, coarse_np, dataset.dates([0]), derived.lat, derived.lon, derived.lat_lr, derived.lon_lr, output / "heun_25.nc", {"acceptance_test": "gpu_smoke"})
    import xarray as xr
    with xr.open_dataset(product) as check: assert np.isfinite(check.sst_generated.values[:, derived.ocean_mask]).all()

    config = configs["flow_ar"]; ar = build_dataset(config, normalization, config["smoke_date_ranges"], "autoregressive", derived=derived)
    days = min(10, len(ar)); items = [ar[index] for index in range(days)]
    conditions = torch.stack([item["condition"] for item in items])[None].to(device)
    previous = items[0]["previous"][None].to(device); mask = items[0]["mask"][None].to(device); model = build_model(config).to(device).eval()
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    generated = rollout(
        model,
        previous,
        conditions,
        mask,
        int(config["rollout_sampler_steps"]),
        config["sampler"],
        enforce_coarse_consistency=bool(
            config.get("enforce_coarse_consistency", False)
        ),
    )
    torch.cuda.synchronize()
    consistency_mse = float(coarse_consistency_mse(
        generated.flatten(0, 1),
        conditions.flatten(0, 1),
        mask.expand(days, -1, -1, -1),
    ))
    report["ar_rollout_10"] = {"seconds": time.perf_counter() - started, "peak_memory_mb": peak_mb(), "days": days, "coarse_consistency_mse_normalized": consistency_mse}
    if not torch.isfinite(generated).all(): raise FloatingPointError("non-finite AR rollout")
    if consistency_mse > 1.0e-10: raise AssertionError(f"coarse consistency MSE {consistency_mse} is too large")
    atomic_json(output / "report.json", report); print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
