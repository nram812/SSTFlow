#!/usr/bin/env python3
"""Actual-grid H200 gate for the GAN-v3 WGAN-GP three-critic fine-tune."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import netCDF4
import numpy as np
import torch

from common import atomic_json, load_config
from train_gan import train


def physical_coarse_error(path: Path, factor: int) -> float:
    with netCDF4.Dataset(path) as dataset:
        generated = np.ma.filled(dataset["sst_generated"][:], np.nan)
        target = np.ma.filled(dataset["sst_target"][:], np.nan)
        coarse = np.ma.filled(dataset["sst_coarse"][:], np.nan)
    mask = np.isfinite(target[0])
    errors = []
    for sample in range(len(generated)):
        for iy in range(coarse.shape[-2]):
            for ix in range(coarse.shape[-1]):
                block_mask = mask[
                    iy * factor : (iy + 1) * factor,
                    ix * factor : (ix + 1) * factor,
                ]
                if not np.isfinite(coarse[sample, iy, ix]) or not block_mask.any():
                    continue
                block = generated[
                    sample,
                    iy * factor : (iy + 1) * factor,
                    ix * factor : (ix + 1) * factor,
                ]
                errors.append(abs(float(np.nanmean(block[block_mask])) - float(coarse[sample, iy, ix])))
    if not errors:
        raise AssertionError("no valid blocks were checked")
    return max(errors)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/gan_sr_v3_wgan_gp_c3_hist_rcp85_finetune_270k.json"
    )
    config["smoke_output_dir"] = str(
        root / "runs/gpu_smoke/gan_sr_v3_wgan_gp_c3_hist_rcp85_finetune_270k"
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    status = train(config, smoke_steps=220003, device_name="cuda")
    elapsed = time.perf_counter() - started
    output = Path(config["smoke_output_dir"])
    history = json.loads((output / "training_history.json").read_text())["history"]
    new_records = [record for record in history if int(record["step"]) > 220000]
    if len(new_records) != 3:
        raise AssertionError(f"expected three new generator updates, got {len(new_records)}")
    required = (
        "critic", "critic_steps", "gradient_penalty", "gradient_penalty_norm",
        "wasserstein_critic_cost", "wasserstein_estimate", "adversarial_loss",
    )
    for record in new_records:
        if int(record["critic_steps"]) != 3:
            raise AssertionError("critic update ratio is not 3:1")
        for key in required:
            if key not in record or not math.isfinite(float(record[key])):
                raise FloatingPointError(f"missing/non-finite {key}: {record}")
        if float(record["gradient_penalty_norm"]) <= 0:
            raise AssertionError("gradient-penalty norm is non-positive")
    sample = output / "netcdf/sample_step_220003.nc"
    maximum_coarse_error = physical_coarse_error(sample, int(config["coarsen_factor"]))
    if maximum_coarse_error > 2.0e-5:
        raise AssertionError(f"hard coarse consistency failed: {maximum_coarse_error}")
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(),
        "source_step": 220000,
        "final_smoke_step": int(status["step"]),
        "generator_updates": 3,
        "critic_updates": 9,
        "critic_to_generator_ratio": 3,
        "adversarial_objective": status["adversarial_objective"],
        "gradient_penalty_weight": status["gradient_penalty_weight"],
        "new_records": new_records,
        "maximum_physical_coarse_error_c": maximum_coarse_error,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "elapsed_seconds": elapsed,
        "resume_provenance": status["resume_provenance"],
    }
    atomic_json(output / "h200_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
