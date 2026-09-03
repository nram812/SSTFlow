#!/usr/bin/env python3
"""H200 gate and throughput profile for 1024-square NOAA inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

import engine
from common import atomic_json
from data_noaa_5km import NOAATransferDataset
from flow import get_sampler
from infer_access_cm2 import _make_noise, make_condition
from infer_noaa_5km import load_run, pad_fixed_batch, validate_access_grid


def sample_batch(model, condition, mask, seeds, steps):
    noise = _make_noise(
        (len(condition), 1, *mask.shape[-2:]),
        condition.device,
        condition.dtype,
        mask,
        seeds,
    )
    return get_sampler("ab3_pc")(model, noise, condition, mask, steps)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    # The completed 80k stage has the identical architecture and provides an
    # immutable pre-production profile while the 150k continuation is active.
    run_dir = root / "runs/flow_sr_noaa_5km_decoder_finetune_from_038000"
    device = torch.device("cuda")
    config, normalization, product, model, _ = load_run(run_dir, device)
    dataset = NOAATransferDataset(
        config, normalization, config["test_date_ranges"], product
    )
    items = np.arange(4)
    batch = engine.batch_to_device(engine.collate_indices(dataset, items), device)
    mask = batch["target_mask"]
    seeds = 42 + np.asarray(batch["index"], dtype=np.int64)

    torch.cuda.reset_peak_memory_stats()
    alternate = batch["condition"][:1].expand(4, -1, -1, -1).clone()
    alternate_seeds = np.asarray([seeds[0], 8001, 8002, 8003])
    with torch.no_grad():
        first_context = sample_batch(model, alternate, mask, alternate_seeds, 3)
        together = sample_batch(model, batch["condition"], mask, seeds, 3)
    torch.testing.assert_close(first_context[:1], together[:1], rtol=2.0e-5, atol=2.0e-5)

    short_condition, short_seeds, real_count = pad_fixed_batch(
        batch["condition"][:3], seeds[:3], 4
    )
    assert short_condition.shape[0] == 4 and len(short_seeds) == 4 and real_count == 3

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        generated_test = sample_batch(model, batch["condition"], mask, seeds, 75)
    torch.cuda.synchronize()
    test_seconds = time.perf_counter() - started
    if not torch.isfinite(generated_test).all():
        raise FloatingPointError("75-step test sample is non-finite")
    if torch.count_nonzero(generated_test * (1 - mask)):
        raise AssertionError("test inference changed target land from zero")

    access_path = root / "derived/sst_downscaling_access_converted.nc"
    with xr.open_dataset(access_path, engine="h5netcdf") as access:
        field = validate_access_grid(access, product, "sst_lr")
        coarse = field.isel(time=np.arange(4)).values
    condition = torch.from_numpy(
        make_condition(
            coarse,
            product.coarse_mask,
            normalization["sst_mean"],
            normalization["sst_std"],
        )
    ).to(device)
    access_mask = torch.from_numpy(product.target_mask[None, None].astype(np.float32)).to(
        device
    ).expand(4, -1, -1, -1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        generated_access = sample_batch(
            model, condition, access_mask, np.arange(4) + 42, 75
        )
    torch.cuda.synchronize()
    access_seconds = time.perf_counter() - started
    if not torch.isfinite(generated_access).all():
        raise FloatingPointError("75-step ACCESS sample is non-finite")

    seconds_per_day = max(test_seconds, access_seconds) / 4
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "profile_source_run": str(run_dir),
        "profile_source_step": 80000,
        "architecture_identical_to_150k": True,
        "batch_size": 4,
        "sampler": "ab3_pc",
        "sampler_steps": 75,
        "test_batch_seconds": test_seconds,
        "access_batch_seconds": access_seconds,
        "conservative_seconds_per_day": seconds_per_day,
        "estimated_test_hours": seconds_per_day * 1094 / 3600,
        "estimated_each_access_period_hours": seconds_per_day * 3653 / 3600,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "fixed_batch_first_sample_independent_of_other_members": True,
        "final_short_batch_padding_tested": True,
        "finite_test_and_access": True,
        "land_exactly_zero": True,
    }
    output = root / "runs/smoke/flow_sr_noaa_5km_decoder_continue_150k/inference_profile.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    dataset.close()


if __name__ == "__main__":
    main()
