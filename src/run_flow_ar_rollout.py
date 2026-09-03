#!/usr/bin/env python3
"""Run a true free-running SST super-resolution forecast from one test state.

The initial high-resolution SST is observed exactly once.  Every later day is
sampled from flow noise, that day's coarse SST boundary, and the previous
*generated* high-resolution state.  The optional latent-noise correlation
couples stochastic texture between days without changing any daily marginal.
There are no truth resets.

This dedicated driver intentionally loads ``config_used.json`` from the run
directory.  In particular, the legacy ``runs/flow_ar`` checkpoint must use the
full-state, uncapped lag pathway with which it was trained; substituting the
new coarse-guided model semantics changes the learned velocity field despite
having a state dict with identical tensor shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from callbacks import coarse_to_physical, to_physical, write_metrics
from common import load_json
from consistency import masked_block_mean, project_to_coarse
from data import AutoregressiveSuperResolutionDataset, DerivedProduct
from flow import SAMPLERS, get_sampler, masked_noise
from model import build_model


def _date(value: str) -> np.datetime64:
    return np.datetime64(value, "D")


def lead_count(initial_date: str, end_date: str) -> int:
    """Number of generated days after ``initial_date`` through ``end_date``."""
    initial, end = _date(initial_date), _date(end_date)
    leads = int((end - initial) / np.timedelta64(1, "D"))
    if leads < 1:
        raise ValueError("end_date must be at least one day after initial_date")
    return leads


def require_inside_one_range(
    initial_date: str, end_date: str, ranges: list[list[str]]
) -> None:
    """Reject test rollouts that leave a configured contiguous test range."""
    initial, end = _date(initial_date), _date(end_date)
    if not any(_date(first) <= initial <= end <= _date(last) for first, last in ranges):
        raise ValueError(
            f"Rollout {initial_date}..{end_date} is not contained in one "
            f"configured test range: {ranges}"
        )


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_semantics(config: dict) -> dict:
    """Resolve explicit legacy/new lag semantics for provenance and tests."""
    return {
        "lag_conditioning": str(config.get("lag_conditioning", "full_state")),
        "lag_guidance_scale": float(config.get("lag_guidance_scale", 1.0)),
        "coarse_consistency_projection": bool(
            config.get("enforce_coarse_consistency", False)
        ),
    }


def _block_error(
    state: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
) -> tuple[float, float]:
    means, fractions = masked_block_mean(state, mask, condition.shape[-2:])
    valid = (condition[:, 1:2] > 0.5) & (fractions > 0)
    error = (means - condition[:, :1])[valid]
    if error.numel() == 0:
        raise ValueError("No valid coarse ocean cells for consistency diagnostics")
    return float(torch.sqrt(torch.mean(error.square()))), float(error.abs().max())


@torch.inference_mode()
def generate_rollout(
    model,
    previous: torch.Tensor,
    conditions: torch.Tensor,
    mask: torch.Tensor,
    sampler: str,
    sampler_steps: int,
    generator: torch.Generator,
    enforce_coarse_consistency: bool,
    normalized_abs_limit: float,
    noise_correlation: float = 0.0,
    progress_every: int = 10,
) -> tuple[torch.Tensor, dict]:
    """Generate all physical leads while chaining only generated states."""
    if conditions.ndim != 4:
        raise ValueError(
            "conditions must have shape (lead, channel, lat_lr, lon_lr), got "
            f"{tuple(conditions.shape)}"
        )
    if previous.ndim != 4 or previous.shape[0] != 1:
        raise ValueError(
            f"previous must have shape (1, channel, lat, lon), got {tuple(previous.shape)}"
        )
    if sampler_steps < 1:
        raise ValueError("sampler_steps must be positive")
    if normalized_abs_limit <= 0:
        raise ValueError("normalized_abs_limit must be positive")
    if not 0.0 <= noise_correlation <= 1.0:
        raise ValueError("noise_correlation must lie in [0, 1]")

    sampler_function = get_sampler(sampler)
    state = previous * mask
    generated = []
    rms_coarse_error = []
    max_coarse_error = []
    normalized_min = []
    normalized_max = []
    latent_noise = None
    for lead in range(len(conditions)):
        condition = conditions[lead : lead + 1]
        innovation = masked_noise(state, mask, generator)
        if latent_noise is None:
            latent_noise = innovation
        else:
            latent_noise = (
                noise_correlation * latent_noise
                + (1.0 - noise_correlation**2) ** 0.5 * innovation
            )
            latent_noise = latent_noise * mask
        # ``state`` is the prior generated physical day.  It remains fixed as
        # lag conditioning during the entire within-day flow ODE integration.
        state = sampler_function(
            model, latent_noise, condition, mask, sampler_steps, state
        )
        if enforce_coarse_consistency:
            state = project_to_coarse(state, condition, mask)
        ocean = mask.expand_as(state) > 0.5
        if not torch.isfinite(state[ocean]).all():
            raise FloatingPointError(f"Non-finite ocean state at lead {lead + 1}")
        minimum = float(state[ocean].min())
        maximum = float(state[ocean].max())
        if max(abs(minimum), abs(maximum)) > normalized_abs_limit:
            raise FloatingPointError(
                f"Normalized SST stability limit exceeded at lead {lead + 1}: "
                f"range [{minimum:.3f}, {maximum:.3f}], limit "
                f"+/-{normalized_abs_limit:.3f}"
            )
        rms, maximum_error = _block_error(state, condition, mask)
        normalized_min.append(minimum)
        normalized_max.append(maximum)
        rms_coarse_error.append(rms)
        max_coarse_error.append(maximum_error)
        generated.append(state.cpu())
        if (lead + 1) % progress_every == 0 or lead + 1 == len(conditions):
            print(
                f"[rollout] {lead + 1}/{len(conditions)} "
                f"normalized=[{minimum:.3f}, {maximum:.3f}] "
                f"coarse_max_error={maximum_error:.3e}",
                flush=True,
            )
    return torch.cat(generated, dim=0), {
        "normalized_min_by_lead": normalized_min,
        "normalized_max_by_lead": normalized_max,
        "coarse_rmse_normalized_by_lead": rms_coarse_error,
        "coarse_max_abs_error_normalized_by_lead": max_coarse_error,
    }


def rollout_metrics(
    generated: np.ndarray, target: np.ndarray, initial_state: np.ndarray
) -> dict:
    """Lead-wise skill, persistence, evolution, and finite-value diagnostics."""
    error = generated - target
    persistence = np.broadcast_to(initial_state[None], target.shape)
    persistence_error = persistence - target
    rmse = np.sqrt(np.nanmean(np.square(error), axis=(1, 2)))
    persistence_rmse = np.sqrt(
        np.nanmean(np.square(persistence_error), axis=(1, 2))
    )
    oracle_previous = np.concatenate((initial_state[None], target[:-1]), axis=0)
    oracle_daily_persistence_rmse = np.sqrt(
        np.nanmean(np.square(oracle_previous - target), axis=(1, 2))
    )
    generated_steps = np.concatenate((initial_state[None], generated), axis=0)
    target_steps = np.concatenate((initial_state[None], target), axis=0)
    generated_change = np.abs(np.diff(generated_steps, axis=0))
    target_change = np.abs(np.diff(target_steps, axis=0))
    return {
        "days": int(len(generated)),
        "overall_rmse_c": float(np.sqrt(np.nanmean(np.square(error)))),
        "overall_mae_c": float(np.nanmean(np.abs(error))),
        "overall_bias_c": float(np.nanmean(error)),
        "rmse_c_by_lead": rmse.tolist(),
        "bias_c_by_lead": np.nanmean(error, axis=(1, 2)).tolist(),
        "initial_state_persistence_rmse_c_by_lead": persistence_rmse.tolist(),
        "oracle_daily_persistence_rmse_c_by_lead": (
            oracle_daily_persistence_rmse.tolist()
        ),
        "skill_vs_initial_persistence_by_lead": (
            1.0 - np.square(rmse) / np.maximum(np.square(persistence_rmse), 1.0e-12)
        ).tolist(),
        "generated_mean_abs_daily_change_c": float(np.nanmean(generated_change)),
        "target_mean_abs_daily_change_c": float(np.nanmean(target_change)),
        "evolution_ratio": float(
            np.nanmean(generated_change) / max(np.nanmean(target_change), 1.0e-12)
        ),
        "generated_min_c": float(np.nanmin(generated)),
        "generated_max_c": float(np.nanmax(generated)),
        "target_min_c": float(np.nanmin(target)),
        "target_max_c": float(np.nanmax(target)),
        "nonfinite_target_ocean_pixels": int(
            (np.isfinite(target) & ~np.isfinite(generated)).sum()
        ),
    }


def save_diagnostics(
    generated: np.ndarray,
    target: np.ndarray,
    metrics: dict,
    dates: list[str],
    output_prefix: Path,
) -> None:
    """Save compact lead-time and representative-map diagnostics."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    leads = np.arange(1, len(generated) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(leads, metrics["rmse_c_by_lead"], label="model")
    axes[0].plot(
        leads,
        metrics["initial_state_persistence_rmse_c_by_lead"],
        label="fixed initial-state persistence",
    )
    axes[0].plot(
        leads,
        metrics["oracle_daily_persistence_rmse_c_by_lead"],
        label="one-day truth persistence",
        alpha=0.8,
    )
    axes[0].set_ylabel("RMSE (degC)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(leads, metrics["bias_c_by_lead"], label="bias")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("autoregressive lead (days)")
    axes[1].set_ylabel("bias (degC)")
    axes[1].grid(alpha=0.3)
    figure.suptitle("One-year free-running SST super-resolution diagnostics")
    figure.savefig(output_prefix.with_name(output_prefix.name + "_skill.png"), dpi=150)
    plt.close(figure)

    selected = np.unique(
        np.linspace(0, len(generated) - 1, min(4, len(generated)), dtype=int)
    )
    figure, axes = plt.subplots(len(selected), 3, figsize=(14, 4 * len(selected)),
                                constrained_layout=True, squeeze=False)
    for row, index in enumerate(selected):
        error = generated[index] - target[index]
        finite = target[index][np.isfinite(target[index])]
        vmin, vmax = np.percentile(finite, [1, 99])
        limit = max(float(np.nanpercentile(np.abs(error), 99)), 1.0e-6)
        panels = (
            (target[index], "truth", "turbo", vmin, vmax),
            (generated[index], "generated", "turbo", vmin, vmax),
            (error, "generated - truth", "RdBu_r", -limit, limit),
        )
        for column, (field, label, cmap, lower, upper) in enumerate(panels):
            image = axes[row, column].imshow(
                field, origin="lower", cmap=cmap, vmin=lower, vmax=upper
            )
            axes[row, column].set_title(
                f"lead {index + 1} · {dates[index]} · {label}"
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
    figure.savefig(
        output_prefix.with_name(output_prefix.name + "_snapshots.png"), dpi=150
    )
    plt.close(figure)


def save_product(
    output_path: Path,
    generated: np.ndarray,
    target: np.ndarray,
    coarse: np.ndarray,
    initial_state: np.ndarray,
    dates: list[str],
    derived: DerivedProduct,
    attrs: dict,
) -> None:
    # NetCDF classic-compatible attributes have no boolean dtype.  Convert
    # explicitly rather than allowing h5netcdf to reject a completed rollout.
    netcdf_attrs = {
        key: int(value) if isinstance(value, (bool, np.bool_)) else value
        for key, value in attrs.items()
    }
    output = xr.Dataset(
        data_vars={
            "sst_generated": (("time", "lat", "lon"), generated.astype(np.float32)),
            "sst_target": (("time", "lat", "lon"), target.astype(np.float32)),
            "sst_coarse": (
                ("time", "lat_lr", "lon_lr"), coarse.astype(np.float32)
            ),
            "sst_initial_state": (
                ("lat", "lon"), initial_state.astype(np.float32)
            ),
            "ocean_mask": (
                ("lat", "lon"), derived.ocean_mask.astype(np.int8)
            ),
            "coarse_ocean_mask": (
                ("lat_lr", "lon_lr"), derived.ocean_mask_lr.astype(np.int8)
            ),
        },
        coords={
            "time": np.asarray(dates, dtype="datetime64[ns]"),
            "lead": ("time", np.arange(1, len(dates) + 1, dtype=np.int32)),
            "lat": derived.lat,
            "lon": derived.lon,
            "lat_lr": derived.lat_lr,
            "lon_lr": derived.lon_lr,
        },
        attrs=netcdf_attrs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_suffix(".partial.nc")
    if temporary.exists():
        temporary.unlink()
    encoding = {
        "sst_generated": {
            "dtype": "float32", "zlib": True, "complevel": 4,
            "chunksizes": (1, len(derived.lat), len(derived.lon)),
        },
        "sst_target": {
            "dtype": "float32", "zlib": True, "complevel": 4,
            "chunksizes": (1, len(derived.lat), len(derived.lon)),
        },
        "sst_coarse": {
            "dtype": "float32", "zlib": True, "complevel": 4,
            "chunksizes": (min(32, len(dates)), len(derived.lat_lr), len(derived.lon_lr)),
        },
        "sst_initial_state": {"dtype": "float32", "zlib": True, "complevel": 4},
        "ocean_mask": {"dtype": "int8", "zlib": True, "complevel": 4},
        "coarse_ocean_mask": {"dtype": "int8", "zlib": True, "complevel": 4},
    }
    output.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    os.replace(temporary, output_path)


def run(
    run_dir: Path,
    initial_date: str,
    end_date: str,
    output_path: Path,
    sampler: str = "ab2_pc",
    sampler_steps: int = 75,
    seed: int = 2020,
    device_name: str = "cuda",
    enforce_coarse_consistency: bool = True,
    normalized_abs_limit: float = 20.0,
    noise_correlation: float = 0.0,
) -> dict:
    config = load_json(run_dir / "config_used.json")
    normalization = load_json(run_dir / "normalization.json")
    if config.get("model_kind") != "autoregressive":
        raise ValueError(f"{run_dir} is not an autoregressive run")
    require_inside_one_range(initial_date, end_date, config["test_date_ranges"])
    days = lead_count(initial_date, end_date)
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    dataset = AutoregressiveSuperResolutionDataset(
        config,
        normalization,
        [[initial_date, end_date]],
        horizon=days,
        derived=derived,
        preload=False,
    )
    try:
        if len(dataset) != 1:
            raise RuntimeError(f"Expected one exact rollout window, found {len(dataset)}")
        item = dataset[0]
        dates = dataset.date_window(0)
        if dates[0] != initial_date or dates[-1] != end_date:
            raise RuntimeError(f"Unexpected rollout window endpoints: {dates[0]}, {dates[-1]}")
        expected = np.arange(
            _date(initial_date), _date(end_date) + np.timedelta64(1, "D")
        )
        if not np.array_equal(np.asarray(dates, dtype="datetime64[D]"), expected):
            raise RuntimeError("The rollout window is not daily and gapless")

        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device(device_name)
        weights = next(
            (
                path
                for path in (run_dir / "model_ema.pt", run_dir / "model.pt")
                if path.is_file()
            ),
            None,
        )
        if weights is None:
            raise FileNotFoundError(f"No EMA/raw model weights found in {run_dir}")
        model = build_model(config).to(device)
        model.load_state_dict(
            torch.load(weights, map_location=device, weights_only=True), strict=True
        )
        model.eval()

        previous = item["previous"][None].to(device)
        conditions = item["conditions"].to(device)
        mask = item["mask"][None].to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        generated_tensor, stability = generate_rollout(
            model,
            previous,
            conditions,
            mask,
            sampler,
            sampler_steps,
            generator,
            enforce_coarse_consistency,
            normalized_abs_limit,
            noise_correlation,
        )
        generated = to_physical(
            generated_tensor, normalization, derived.ocean_mask
        )[:, 0]
        target = to_physical(
            item["targets"], normalization, derived.ocean_mask
        )[:, 0]
        initial_state = to_physical(
            item["previous"][None], normalization, derived.ocean_mask
        )[0, 0]
        coarse = coarse_to_physical(
            item["conditions"], normalization, derived.ocean_mask_lr
        )
        metrics = rollout_metrics(generated, target, initial_state)
        semantics = checkpoint_semantics(config)
        payload = {
            "status": "passed",
            "run": str(run_dir.resolve()),
            "weights": weights.name,
            "weights_sha256": file_sha256(weights),
            "initial_state_date": initial_date,
            "first_generated_date": dates[1],
            "last_generated_date": dates[-1],
            "truth_resets": 0,
            "sampler": sampler,
            "sampler_description": (
                "Adams-Bashforth 2 predictor / trapezoidal "
                "Adams-Moulton 2 corrector"
            ),
            "sampler_steps": int(sampler_steps),
            "seed": int(seed),
            "noise_correlation": float(noise_correlation),
            "enforce_coarse_consistency": bool(enforce_coarse_consistency),
            "checkpoint_semantics": semantics,
            "metrics": metrics,
            "stability": stability,
        }
        attrs = {
            "experiment": config["name"],
            "weights": weights.name,
            "weights_sha256": payload["weights_sha256"],
            "mode": "free_running_autoregressive",
            "initial_state_date": initial_date,
            "first_generated_date": dates[1],
            "end_date": end_date,
            "truth_resets": 0,
            "sampler": sampler,
            "sampler_steps": int(sampler_steps),
            "seed": int(seed),
            "lag_conditioning": semantics["lag_conditioning"],
            "lag_guidance_scale": semantics["lag_guidance_scale"],
            "coarse_consistency_projection": int(enforce_coarse_consistency),
            "noise_correlation": float(noise_correlation),
            "units": "degrees C",
        }
        save_product(
            output_path,
            generated,
            target,
            coarse,
            initial_state,
            dates[1:],
            derived,
            attrs,
        )
        metrics_path = output_path.with_suffix(".metrics.json")
        write_metrics(metrics_path, payload)
        save_diagnostics(
            generated,
            target,
            metrics,
            dates[1:],
            output_path.with_suffix(""),
        )
        print(json.dumps(payload, indent=2), flush=True)
        return payload
    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--initial-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampler", choices=SAMPLERS, default="ab2_pc")
    parser.add_argument("--sampler-steps", type=int, default=75)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--normalized-abs-limit", type=float, default=20.0)
    parser.add_argument(
        "--noise-correlation",
        type=float,
        default=0.0,
        help="AR(1) correlation of consecutive daily flow-noise fields",
    )
    parser.add_argument(
        "--no-coarse-consistency",
        action="store_true",
        help="Disable the daily exact coarse-block projection (diagnostics only)",
    )
    arguments = parser.parse_args()
    run(
        arguments.run,
        arguments.initial_date,
        arguments.end_date,
        arguments.output,
        arguments.sampler,
        arguments.sampler_steps,
        arguments.seed,
        arguments.device,
        not arguments.no_coarse_consistency,
        arguments.normalized_abs_limit,
        arguments.noise_correlation,
    )


if __name__ == "__main__":
    main()
