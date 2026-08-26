#!/usr/bin/env python3
"""Offline test-split evaluation for flow, autoregressive-flow, and GAN runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import engine
from callbacks import coarse_to_physical, field_metrics, radial_spectrum, save_netcdf, save_rollout_netcdf, write_metrics
from common import attach_ocean_mask, load_json
from data import DerivedProduct, build_dataset
from flow import rollout, sample
from model import build_model
from model_gan import build_generator


def load_run(run_dir: Path, device: torch.device):
    config = load_json(run_dir / "config_used.json")
    normalization = load_json(run_dir / "normalization.json")
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    normalization = attach_ocean_mask(normalization, derived.ocean_mask)
    kind = config.get("model_kind", "super_resolution")
    if kind == "gan":
        model = build_generator(config)
        candidates = (run_dir / "generator_ema.pt", run_dir / "generator.pt")
    else:
        model = build_model(config)
        candidates = (run_dir / "model_ema.pt", run_dir / "model.pt")
    weights = next((path for path in candidates if path.is_file()), None)
    if weights is None:
        raise FileNotFoundError(f"No evaluation weights found in {run_dir}")
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()
    dataset_kind = "autoregressive" if kind == "autoregressive" else "super_resolution"
    dataset = build_dataset(config, normalization, config["test_date_ranges"], dataset_kind,
                            derived=derived, preload=False)
    return config, normalization, derived, dataset, model, weights


@torch.no_grad()
def predict(model, batch: dict, config: dict, device: torch.device, steps: int, seed: int):
    batch = engine.batch_to_device(batch, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    kind = config.get("model_kind")
    if kind == "gan":
        generated = model(batch["condition"], batch["mask"], generator=generator)
    else:
        generated = sample(model, batch["condition"], batch["mask"], batch["target"].shape,
                           steps=steps, sampler=config.get("sampler", "heun"),
                           previous_state=batch.get("previous"), generator=generator,
                           device=device)
    return generated, batch


def evaluate(run_dir: Path, device_name: str | None = None, samples: int = 32,
             batch_size: int = 4, sampler_steps: list[int] | None = None,
             rollout_days: int = 10) -> dict:
    device = engine.resolve_device(device_name)
    config, normalization, derived, dataset, model, weights = load_run(run_dir, device)
    indices = engine.fixed_indices(dataset, samples)
    step_values = sampler_steps or ([int(config.get("preview_sampler_steps", 25))]
                                    if config.get("model_kind") != "gan" else [1])
    results: dict[str, dict] = {}
    saved = None
    for steps in step_values:
        generated_all, target_all, coarse_all, dates = [], [], [], []
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            raw = engine.collate_indices(dataset, chosen)
            generated, batch = predict(model, raw, config, device, steps, int(config.get("seed", 42)) + start)
            generated_all.append(engine_to_physical(generated, normalization, derived.ocean_mask))
            target_all.append(engine_to_physical(batch["target"], normalization, derived.ocean_mask))
            coarse_all.append(coarse_to_physical(batch["condition"], normalization, derived.ocean_mask_lr))
            dates.extend(dataset.dates(chosen))
        generated_np = np.concatenate(generated_all)[:, 0]
        target_np = np.concatenate(target_all)[:, 0]
        coarse_np = np.concatenate(coarse_all)
        metrics = field_metrics(generated_np, target_np)
        spectra = []
        for generated_field, target_field in zip(generated_np, target_np):
            _, generated_spectrum = radial_spectrum(generated_field)
            _, target_spectrum = radial_spectrum(target_field)
            spectra.append(np.nanmean(np.abs(generated_spectrum - target_spectrum)))
        metrics["mean_spectral_amplitude_error"] = float(np.mean(spectra))
        results[str(steps)] = metrics
        saved = (generated_np, target_np, coarse_np, dates)

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_np, target_np, coarse_np, dates = saved
    save_netcdf(generated_np, target_np, coarse_np, dates, derived.lat, derived.lon,
                derived.lat_lr, derived.lon_lr, output_dir / "test_samples.nc",
                {"experiment": config["name"], "weights": weights.name})

    rollout_metrics = None
    if config.get("model_kind") == "autoregressive" and rollout_days > 0:
        length = min(rollout_days, len(dataset))
        items = [dataset[index] for index in range(length)]
        conditions = torch.stack([item["condition"] for item in items], dim=1).to(device)
        mask = items[0]["mask"][None].to(device)
        previous = items[0]["previous"][None].to(device)
        generator = torch.Generator(device=device).manual_seed(int(config.get("seed", 42)))
        generated = rollout(model, previous, conditions, mask,
                            int(config.get("rollout_sampler_steps", 25)),
                            config.get("sampler", "heun"), generator)[0]
        truth = torch.stack([item["target"] for item in items])
        generated_physical = engine_to_physical(generated, normalization, derived.ocean_mask)[:, 0]
        target_physical = engine_to_physical(truth, normalization, derived.ocean_mask)[:, 0]
        rollout_metrics = field_metrics(generated_physical, target_physical)
        save_rollout_netcdf(generated_physical, target_physical,
                            [item["date"] for item in items], derived.lat, derived.lon,
                            output_dir / "test_rollout.nc", {"experiment": config["name"]})

    summary = {"run": str(run_dir), "weights": weights.name, "samples": len(indices),
               "sampler_step_ablation": results, "rollout": rollout_metrics}
    write_metrics(output_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    dataset.close()
    return summary


def engine_to_physical(values, normalization, mask):
    from callbacks import to_physical
    return to_physical(values, normalization, mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sampler-steps", type=int, nargs="+")
    parser.add_argument("--rollout-days", type=int, default=10)
    arguments = parser.parse_args()
    evaluate(arguments.run, arguments.device, arguments.samples, arguments.batch_size,
             arguments.sampler_steps, arguments.rollout_days)


if __name__ == "__main__":
    main()
