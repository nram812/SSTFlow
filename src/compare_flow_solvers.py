#!/usr/bin/env python3
"""Compare higher-order flow solvers on the same test days and initial noise."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import engine
from callbacks import coarse_to_physical, field_metrics, to_physical, write_metrics
from evaluate import load_run
from flow import get_sampler, masked_noise


SOLVERS = ("heun", "ab2", "ab3_pc")
SOLVER_EVALUATIONS_PER_STEP = {"heun": 2, "ab2": 1, "ab3_pc": 2}


def model_evaluations(solver: str, steps: int) -> int:
    """Return the number of velocity-network calls for one sample."""
    return SOLVER_EVALUATIONS_PER_STEP[solver] * steps


def _atomic_netcdf(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.nc")
    encoding = {
        name: {"dtype": "float32", "zlib": True, "complevel": 4}
        for name in dataset.data_vars
    }
    dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    os.replace(temporary, path)


def _figure(fields: dict[str, np.ndarray], date: str, path: Path) -> None:
    truth = fields["target"][0]
    heun, ab2, ab3 = (fields[name][0] for name in SOLVERS)
    finite = np.isfinite(truth)
    limits = np.nanpercentile(truth, [1, 99])
    differences = np.concatenate([
        (heun - truth)[finite], (ab2 - truth)[finite], (ab3 - truth)[finite],
        (ab3 - heun)[finite]
    ])
    difference_limit = float(np.nanpercentile(np.abs(differences), 99))
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    panels = (
        (truth, "OFAM target", "turbo", limits),
        (heun, "Heun (100 steps; 200 evaluations)", "turbo", limits),
        (ab2, "AB2 (100 steps; 100 evaluations)", "turbo", limits),
        (ab3, "AB3/AM3 PC (100 steps; 200 evaluations)", "turbo", limits),
        (heun - truth, "Heun - target", "RdBu_r", (-difference_limit, difference_limit)),
        (ab2 - truth, "AB2 - target", "RdBu_r", (-difference_limit, difference_limit)),
        (ab3 - truth, "AB3/AM3 PC - target", "RdBu_r", (-difference_limit, difference_limit)),
        (ab3 - heun, "AB3/AM3 PC - Heun", "RdBu_r", (-difference_limit, difference_limit)),
    )
    for axis, (values, title, cmap, scale) in zip(axes.flat, panels):
        image = axis.imshow(values, origin="lower", cmap=cmap, vmin=scale[0], vmax=scale[1])
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(f"flow_sr higher-order solver comparison · {date}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


@torch.no_grad()
def compare(
    run_dir: Path,
    days: int = 30,
    steps: int = 100,
    batch_size: int = 4,
    device_name: str | None = None,
) -> dict:
    device = engine.resolve_device(device_name)
    config, normalization, derived, dataset, model, weights = load_run(run_dir, device)
    if config.get("model_kind") != "super_resolution":
        raise ValueError("This comparison is for the non-autoregressive flow_sr model")
    count = min(int(days), len(dataset))
    indices = np.arange(count, dtype=np.int64)
    generated = {solver: [] for solver in SOLVERS}
    targets, coarse, dates = [], [], []
    elapsed = {solver: 0.0 for solver in SOLVERS}
    seed = int(config.get("seed", 42)) + 100_000

    for start in range(0, count, batch_size):
        chosen = indices[start:start + batch_size]
        batch = engine.batch_to_device(engine.collate_indices(dataset, chosen), device)
        generator = torch.Generator(device=device).manual_seed(seed + start)
        noise = masked_noise(batch["target"], batch["mask"], generator)
        for solver in SOLVERS:
            if device.type == "cuda": torch.cuda.synchronize(device)
            began = time.perf_counter()
            result = get_sampler(solver)(
                model, noise.clone(), batch["condition"], batch["mask"], steps
            )
            if device.type == "cuda": torch.cuda.synchronize(device)
            elapsed[solver] += time.perf_counter() - began
            generated[solver].append(
                to_physical(result, normalization, derived.ocean_mask)[:, 0]
            )
        targets.append(to_physical(batch["target"], normalization, derived.ocean_mask)[:, 0])
        coarse.append(coarse_to_physical(batch["condition"], normalization, derived.ocean_mask_lr))
        dates.extend(dataset.dates(chosen))
        print(f"[solver comparison] {min(start + len(chosen), count)}/{count} days", flush=True)

    fields = {name: np.concatenate(parts) for name, parts in generated.items()}
    fields["target"] = np.concatenate(targets)
    coarse_values = np.concatenate(coarse)
    metrics = {}
    for solver in SOLVERS:
        metrics[solver] = {
            **field_metrics(fields[solver], fields["target"]),
            "elapsed_seconds": elapsed[solver],
            "model_evaluations_per_sample": model_evaluations(solver, steps),
            "seconds_per_day": elapsed[solver] / count,
        }
    ocean = np.isfinite(fields["target"])
    solver_delta = fields["heun"] - fields["ab2"]
    metrics["heun_vs_ab2"] = {
        "rmse_c": float(np.sqrt(np.mean(np.square(solver_delta[ocean])))),
        "mae_c": float(np.mean(np.abs(solver_delta[ocean]))),
        "max_abs_c": float(np.max(np.abs(solver_delta[ocean]))),
        "ab2_speedup_vs_heun": elapsed["heun"] / elapsed["ab2"],
    }
    for reference in ("heun", "ab2"):
        delta = fields["ab3_pc"] - fields[reference]
        metrics[f"ab3_pc_vs_{reference}"] = {
            "rmse_c": float(np.sqrt(np.mean(np.square(delta[ocean])))),
            "mae_c": float(np.mean(np.abs(delta[ocean]))),
            "max_abs_c": float(np.max(np.abs(delta[ocean]))),
        }

    output_dir = run_dir / "evaluation" / "solver_comparison_100step_30days_ab3pc"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_out = xr.Dataset(
        data_vars={
            "sst_heun": (("time", "lat", "lon"), fields["heun"].astype(np.float32)),
            "sst_ab2": (("time", "lat", "lon"), fields["ab2"].astype(np.float32)),
            "sst_ab3_pc": (("time", "lat", "lon"), fields["ab3_pc"].astype(np.float32)),
            "sst_target": (("time", "lat", "lon"), fields["target"].astype(np.float32)),
            "heun_minus_ab2": (("time", "lat", "lon"), solver_delta.astype(np.float32)),
            "sst_coarse": (("time", "lat_lr", "lon_lr"), coarse_values.astype(np.float32)),
        },
        coords={
            "time": np.asarray(dates, dtype="datetime64[ns]"),
            "lat": derived.lat, "lon": derived.lon,
            "lat_lr": derived.lat_lr, "lon_lr": derived.lon_lr,
        },
        attrs={
            "experiment": config["name"], "weights": weights.name,
            "sampler_steps": int(steps), "initial_noise": "identical between solvers",
            "selection": "first consecutive test days", "units": "degrees C",
        },
    )
    netcdf_path = output_dir / "samples.nc"
    _atomic_netcdf(dataset_out, netcdf_path)
    figure_path = output_dir / "comparison_day_01.png"
    _figure(fields, dates[0], figure_path)
    summary = {
        "run": str(run_dir), "weights": weights.name, "days": count,
        "date_start": dates[0], "date_end": dates[-1], "sampler_steps": steps,
        "same_initial_noise": True, "metrics": metrics,
        "netcdf": str(netcdf_path), "figure": str(figure_path),
    }
    write_metrics(output_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    dataset.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/flow_sr"))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    compare(args.run, args.days, args.steps, args.batch_size, args.device)


if __name__ == "__main__":
    main()
