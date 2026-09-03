#!/usr/bin/env python3
"""Measure temporal coherence from coupled rectified-flow latent noise."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import engine
from callbacks import to_physical
from common import atomic_json, load_json
from data import AutoregressiveSuperResolutionDataset, DerivedProduct
from flow import rollout
from model import build_model
from run_flow_ar_rollout import rollout_metrics


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs/flow_ar_residual_memory"
    config = load_json(run_dir / "config_used.json")
    normalization = load_json(run_dir / "normalization.json")
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    dataset = AutoregressiveSuperResolutionDataset(
        config,
        normalization,
        config["validation_date_ranges"],
        horizon=10,
        derived=derived,
        preload=False,
    )
    try:
        batch = engine.batch_to_device(engine.collate_indices(dataset, [0]), "cuda")
        model = build_model(config).cuda()
        weights = run_dir / "model_ema.pt"
        model.load_state_dict(
            torch.load(weights, map_location="cuda", weights_only=True), strict=True
        )
        model.eval()
        target = to_physical(
            batch["targets"][0], normalization, derived.ocean_mask
        )[:, 0]
        initial = to_physical(
            batch["previous"], normalization, derived.ocean_mask
        )[0, 0]
        results = []
        for rho in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0):
            generated = rollout(
                model,
                batch["previous"],
                batch["conditions"],
                batch["mask"],
                steps=25,
                sampler="heun",
                generator=torch.Generator(device="cuda").manual_seed(22042),
                enforce_coarse_consistency=True,
                noise_correlation=rho,
            )[0]
            physical = to_physical(
                generated, normalization, derived.ocean_mask
            )[:, 0]
            metrics = rollout_metrics(physical, target, initial)
            results.append({"noise_correlation": rho, **metrics})
            print(
                f"rho={rho:.2f} RMSE={metrics['overall_rmse_c']:.4f} "
                f"evolution={metrics['evolution_ratio']:.3f}",
                flush=True,
            )
        output = run_dir / "diagnostics/noise_correlation_profile.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, {"status": "passed", "results": results})
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        rho = [entry["noise_correlation"] for entry in results]
        axes[0].plot(rho, [entry["overall_rmse_c"] for entry in results], "o-")
        axes[0].set(xlabel="latent-noise correlation", ylabel="10-day RMSE (degC)")
        axes[1].plot(rho, [entry["evolution_ratio"] for entry in results], "o-")
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].set(xlabel="latent-noise correlation", ylabel="daily evolution ratio")
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.savefig(run_dir / "predictions/noise_correlation_profile.png", dpi=180)
        plt.close(figure)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
