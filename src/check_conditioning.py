#!/usr/bin/env python3
"""Ablate the coarse SST condition in a trained flow run and save evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import engine
from callbacks import coarse_to_physical, field_metrics, to_physical
from common import atomic_json, attach_ocean_mask, load_json
from data import DerivedProduct, build_dataset
from flow import sample
from losses import masked_mse
from model import build_model


def compare_predictions(conditioned, ablated, target, mask) -> dict:
    """Return normalized-space ablation metrics for a common noise draw."""
    response = masked_mse(conditioned, ablated, mask)
    conditioned_mse = masked_mse(conditioned, target, mask)
    ablated_mse = masked_mse(ablated, target, mask)
    return {
        "condition_response_rmse_normalized": float(torch.sqrt(response)),
        "conditioned_mse_normalized": float(conditioned_mse),
        "ablated_mse_normalized": float(ablated_mse),
        "conditioned_skill_vs_ablation": float(
            1.0 - conditioned_mse / ablated_mse.clamp_min(1.0e-12)
        ),
    }


@torch.no_grad()
def run(
    run_dir: Path,
    device_name: str | None = None,
    steps: int = 25,
    weights_name: str = "model_ema.pt",
) -> dict:
    device = engine.resolve_device(device_name)
    config = load_json(run_dir / "config_used.json")
    normalization = load_json(run_dir / "normalization.json")
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    normalization = attach_ocean_mask(normalization, derived.ocean_mask)
    dataset = build_dataset(
        config,
        normalization,
        config.get("validation_date_ranges", config["smoke_date_ranges"]),
        "super_resolution",
        derived=derived,
        preload=False,
    )
    indices = engine.fixed_indices(dataset, 2)
    batch = engine.batch_to_device(engine.collate_indices(dataset, indices), device)
    model = build_model(config).to(device).eval()
    weights = run_dir / weights_name
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))

    def draw(condition):
        generator = torch.Generator(device=device).manual_seed(20260827)
        return sample(
            model,
            condition,
            batch["mask"],
            batch["target"].shape,
            steps=steps,
            sampler=config.get("sampler", "heun"),
            generator=generator,
            device=device,
        )

    conditioned = draw(batch["condition"])
    ablated_condition = batch["condition"].clone()
    ablated_condition[:, 0] = 0.0  # retain mask channel; remove only coarse SST
    ablated = draw(ablated_condition)
    metrics = compare_predictions(
        conditioned, ablated, batch["target"], batch["mask"]
    )

    conditioned_physical = to_physical(
        conditioned, normalization, derived.ocean_mask
    )[:, 0]
    ablated_physical = to_physical(
        ablated, normalization, derived.ocean_mask
    )[:, 0]
    target_physical = to_physical(
        batch["target"], normalization, derived.ocean_mask
    )[:, 0]
    coarse_physical = coarse_to_physical(
        batch["condition"], normalization, derived.ocean_mask_lr
    )
    metrics["conditioned_physical"] = field_metrics(
        conditioned_physical, target_physical
    )
    metrics["ablated_physical"] = field_metrics(ablated_physical, target_physical)
    metrics.update(
        run=str(run_dir),
        weights=weights.name,
        sampler_steps=int(steps),
        dates=dataset.dates(indices),
    )

    output_dir = run_dir / "conditioning"
    output_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    truth = target_physical[index]
    finite = truth[np.isfinite(truth)]
    vmin, vmax = np.percentile(finite, [1, 99])
    difference = conditioned_physical[index] - ablated_physical[index]
    difference_scale = max(float(np.nanpercentile(np.abs(difference), 99)), 1e-6)
    from coarsen import upsample_nearest

    panels = (
        (upsample_nearest(coarse_physical[index], derived.coarsen_factor), "coarse SST", "viridis", vmin, vmax),
        (truth, "truth", "viridis", vmin, vmax),
        (conditioned_physical[index], "conditioned sample", "viridis", vmin, vmax),
        (ablated_physical[index], "coarse SST ablated", "viridis", vmin, vmax),
        (difference, "conditioned - ablated", "RdBu_r", -difference_scale, difference_scale),
    )
    figure, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    for axis, (field, title, cmap, low, high) in zip(axes, panels):
        image = axis.imshow(field, origin="lower", cmap=cmap, vmin=low, vmax=high)
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(
        f"Coarse-conditioning ablation · {metrics['dates'][0]} · {steps} sampler steps"
    )
    label = weights.stem
    figure.savefig(output_dir / f"conditioning_ablation_{label}.png", dpi=130)
    plt.close(figure)
    atomic_json(output_dir / f"metrics_{label}.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    dataset.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--weights", default="model_ema.pt")
    arguments = parser.parse_args()
    run(arguments.run, arguments.device, arguments.steps, arguments.weights)


if __name__ == "__main__":
    main()
