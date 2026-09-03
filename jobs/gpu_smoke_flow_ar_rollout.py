#!/usr/bin/env python3
"""H200 gate for exact legacy semantics and 75-step AB2-PC AR inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from callbacks import to_physical, write_metrics
from common import load_json
from data import AutoregressiveSuperResolutionDataset, DerivedProduct
from model import build_model
from run_flow_ar_rollout import generate_rollout, run


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "flow_ar"
OUTPUT = RUN / "evaluation" / "preflight_ab2pc75_5day.nc"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("H200 smoke requires CUDA")
    device = torch.device("cuda")
    started = time.monotonic()
    payload = run(
        RUN,
        "2011-01-01",
        "2011-01-06",
        OUTPUT,
        sampler="ab2_pc",
        sampler_steps=75,
        seed=2020,
        device_name="cuda",
        enforce_coarse_consistency=True,
    )

    # Independently demonstrate why the checkpoint-compatibility fix matters:
    # the same weights under the later high-pass/capped lag definition must not
    # be mistaken for the model that was actually trained.
    config = load_json(RUN / "config_used.json")
    normalization = load_json(RUN / "normalization.json")
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    dataset = AutoregressiveSuperResolutionDataset(
        config,
        normalization,
        [["2011-01-01", "2011-01-03"]],
        horizon=2,
        derived=derived,
        preload=False,
    )
    try:
        item = dataset[0]
        previous = item["previous"][None].to(device)
        conditions = item["conditions"].to(device)
        mask = item["mask"][None].to(device)
        weights = torch.load(RUN / "model_ema.pt", map_location=device, weights_only=True)

        exact = build_model(config).to(device).eval()
        exact.load_state_dict(weights, strict=True)
        changed_config = {
            **config,
            "lag_conditioning": "within_block_anomaly",
            "lag_guidance_scale": 0.25,
        }
        changed = build_model(changed_config).to(device).eval()
        changed.load_state_dict(weights, strict=True)
        exact_fields, _ = generate_rollout(
            exact,
            previous,
            conditions,
            mask,
            "ab2_pc",
            10,
            torch.Generator(device=device).manual_seed(77),
            True,
            20.0,
        )
        changed_fields, _ = generate_rollout(
            changed,
            previous,
            conditions,
            mask,
            "ab2_pc",
            10,
            torch.Generator(device=device).manual_seed(77),
            True,
            20.0,
        )
        exact_physical = to_physical(exact_fields, normalization, derived.ocean_mask)
        changed_physical = to_physical(
            changed_fields, normalization, derived.ocean_mask
        )
        semantic_rms_difference = float(
            np.sqrt(np.nanmean(np.square(exact_physical - changed_physical)))
        )
        if semantic_rms_difference <= 1.0e-4:
            raise AssertionError("Legacy/new lag semantics unexpectedly agree")
    finally:
        dataset.close()

    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.monotonic() - started,
        "production_sampler": "ab2_pc",
        "production_sampler_steps": 75,
        "preflight_days": 5,
        "preflight_overall_rmse_c": payload["metrics"]["overall_rmse_c"],
        "preflight_evolution_ratio": payload["metrics"]["evolution_ratio"],
        "preflight_max_coarse_error_normalized": max(
            payload["stability"]["coarse_max_abs_error_normalized_by_lead"]
        ),
        "legacy_vs_changed_semantics_rms_difference_c": semantic_rms_difference,
    }
    write_metrics(ROOT / "reports" / "flow_ar_rollout_preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
