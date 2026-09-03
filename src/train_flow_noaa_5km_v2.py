#!/usr/bin/env python3
"""Train a direct NOAA 1024-square flow using pretrained OFAM features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import engine
from common import atomic_json, load_config, load_json
from data_noaa_5km import NOAATransferDataset, NOAATransferProduct
from flow import sample
from model_noaa_5km_v2 import (
    NOAAFrozenTrunkFlow,
    coastline_ocean_mask,
    high_resolution_flow_losses,
    ocean_block_mean,
)


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def configure_paths(config: dict) -> dict:
    root = Path(__file__).resolve().parents[1]
    result = dict(config)
    for key in (
        "source_path",
        "ofam_derived_path",
        "pretrained_ema_path",
        "initial_checkpoint",
    ):
        if key not in result:
            continue
        result[key] = str(_resolve(result[key], root))
    return result


def optimizer_parameter_groups(model: NOAAFrozenTrunkFlow, config: dict) -> list[dict]:
    """Build disjoint head/decoder groups for stable transfer learning."""
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    groups = [{"params": head, "lr": float(config["learning_rate"]), "name": "head"}]
    if model.trainable_policy == "decoder_and_head":
        decoder = [
            parameter for parameter in model.decoder_parameters() if parameter.requires_grad
        ]
        groups.append(
            {
                "params": decoder,
                "lr": float(config["decoder_learning_rate"]),
                "name": "decoder",
            }
        )
    flattened = [parameter for group in groups for parameter in group["params"]]
    if not flattened or len({id(parameter) for parameter in flattened}) != len(flattened):
        raise ValueError("optimizer parameter groups are empty or overlap")
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if {id(parameter) for parameter in flattened} != expected:
        raise ValueError("optimizer groups do not exactly cover trainable parameters")
    return groups


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_stage(
    output_dir: Path,
    modules: dict,
    optimizers: dict,
    schedulers: dict,
    device: torch.device,
    config: dict,
) -> tuple[int, list[dict], dict, dict]:
    """Resume locally or initialize a new stage from a weights-only fork."""
    if (output_dir / "checkpoint.pt").is_file():
        step, history, validation = engine.restore_training_state(
            output_dir, modules, optimizers, schedulers, device
        )
        return step, history, validation, {"mode": "local_resume", "step": step}
    source = config.get("initial_checkpoint")
    if source is None:
        return 0, [], {}, {"mode": "pretrained_initialization", "step": 0}
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"initial checkpoint is missing: {source_path}")
    source_sha256 = _file_sha256(source_path)
    expected_sha256 = config.get("initial_checkpoint_sha256")
    if expected_sha256 is not None and source_sha256 != str(expected_sha256):
        raise ValueError(
            "initial checkpoint checksum differs from the configured source: "
            f"{source_sha256} != {expected_sha256}"
        )
    state = torch.load(source_path, map_location=device, weights_only=False)
    for name, module in modules.items():
        module.load_state_dict(state[f"module_{name}"])
    step = int(state["step"])
    expected = config.get("initial_step_required")
    if expected is not None and step != int(expected):
        raise ValueError(
            f"initial checkpoint step {step} does not equal required step {expected}"
        )
    # A fresh optimiser is intentional: newly unfrozen parameters have no old
    # Adam moments, and importing the head-only optimiser would be ambiguous.
    return step, [], {}, {
        "mode": "weights_only_stage_fork",
        "step": step,
        "source": str(source_path),
        "source_sha256": source_sha256,
        "optimizer_state_restored": False,
    }


def _physical(values: torch.Tensor, mean: float, std: float, mask: np.ndarray):
    fields = values.detach().cpu().numpy()[:, 0] * std + mean
    return np.where(mask[None], fields, np.nan)


def _coast_metrics(error: torch.Tensor, mask: torch.Tensor) -> dict:
    metrics = {}
    for radius in (1, 2, 4, 8):
        coast = coastline_ocean_mask(mask, radius).bool().expand_as(error)
        selected = error[coast]
        metrics[f"coast_{radius}px_bias_c"] = float(selected.mean())
        metrics[f"coast_{radius}px_rmse_c"] = float(selected.square().mean().sqrt())
    interior = (mask - coastline_ocean_mask(mask, 8)).bool().expand_as(error)
    selected = error[interior]
    metrics["interior_gt8px_bias_c"] = float(selected.mean())
    metrics["interior_gt8px_rmse_c"] = float(selected.square().mean().sqrt())
    return metrics


def _save_preview(
    path: Path,
    condition: np.ndarray,
    target: np.ndarray,
    generated: np.ndarray,
    target_mask: np.ndarray,
    base_mask: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    error = generated - target
    coast = coastline_ocean_mask(
        torch.from_numpy(target_mask.astype(np.float32))[None, None], 4
    )[0, 0].numpy().astype(bool)
    coast_error = np.where(coast, error, np.nan)
    repeated_base = np.repeat(np.repeat(base_mask, 2, axis=0), 2, axis=1)
    agreement = np.zeros(target_mask.shape, dtype=np.float32)
    agreement[target_mask & repeated_base] = 1
    agreement[target_mask & ~repeated_base] = 2
    agreement[~target_mask & repeated_base] = -1

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    fields = (condition, target, generated)
    names = ("32x32 NOAA condition", "NOAA truth 0.05°", "direct generated 0.05°")
    for axis, field, name in zip(axes[0], fields, names):
        image = axis.imshow(field, origin="lower", cmap="turbo", vmin=0, vmax=35)
        axis.set_title(name)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    limit = max(float(np.nanpercentile(np.abs(error), 99)), 0.25)
    image = axes[1, 0].imshow(error, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1, 0].set_title("generated − NOAA")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.03)
    coast_limit = max(float(np.nanpercentile(np.abs(coast_error), 99)), 0.25)
    image = axes[1, 1].imshow(
        coast_error, origin="lower", cmap="RdBu_r", vmin=-coast_limit, vmax=coast_limit
    )
    axes[1, 1].set_title("error within 4 NOAA pixels of land")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.03)
    image = axes[1, 2].imshow(agreement, origin="lower", cmap="coolwarm", vmin=-2, vmax=2)
    axes[1, 2].set_title("mask contract: +2 NOAA-only, −1 OFAM-only")
    fig.colorbar(image, ax=axes[1, 2], fraction=0.046, pad=0.03)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_netcdf(
    path: Path,
    product: NOAATransferProduct,
    date: str,
    target: np.ndarray,
    generated: np.ndarray,
    condition: np.ndarray,
    coarsened: np.ndarray,
    attrs: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        data_vars={
            "generated_sst": (("time", "lat_target", "lon_target"), generated[None].astype(np.float32)),
            "target_sst": (("time", "lat_target", "lon_target"), target[None].astype(np.float32)),
            "generated_sst_coarsened_2x": (("time", "lat", "lon"), coarsened[None].astype(np.float32)),
            "condition_sst": (("time", "lat_lr", "lon_lr"), condition[None].astype(np.float32)),
        },
        coords={
            "time": np.asarray([date], dtype="datetime64[ns]"),
            "lat_target": product.target_lat,
            "lon_target": product.target_lon,
            "lat": product.base_lat,
            "lon": product.base_lon,
            "lat_lr": product.coarse_lat,
            "lon_lr": product.coarse_lon,
        },
        attrs=attrs,
    )
    for name in dataset.data_vars:
        dataset[name].attrs["units"] = "degrees_Celsius"
    temporary = path.with_suffix(".partial.nc")
    encoding = {name: {"zlib": True, "complevel": 4} for name in dataset.data_vars}
    dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    os.replace(temporary, path)


@torch.no_grad()
def callbacks(
    model: NOAAFrozenTrunkFlow,
    dataset: NOAATransferDataset,
    device: torch.device,
    normalization: dict,
    config: dict,
    output_dir: Path,
    step: int,
    write_netcdf: bool,
) -> dict:
    model.eval()
    chosen = engine.fixed_indices(dataset, 1)
    batch = engine.batch_to_device(engine.collate_indices(dataset, chosen), device)
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + step)
    generated = sample(
        model,
        batch["condition"],
        batch["target_mask"],
        tuple(batch["target"].shape),
        steps=int(config.get("preview_sampler_steps", 25)),
        sampler=str(config.get("sampler", "heun")),
        generator=generator,
        device=device,
    )
    mean, std = float(normalization["sst_mean"]), float(normalization["sst_std"])
    target = _physical(batch["target"], mean, std, dataset.product.target_mask)[0]
    generated_p = _physical(generated, mean, std, dataset.product.target_mask)[0]
    condition = batch["condition"][0, 0].cpu().numpy() * std + mean
    condition = np.where(dataset.product.coarse_mask, condition, np.nan)
    error_normalized = (generated - batch["target"]) * std
    ocean_error = error_normalized[batch["target_mask"].bool().expand_as(error_normalized)]
    metrics = {
        "rmse_c": float(ocean_error.square().mean().sqrt()),
        "bias_c": float(ocean_error.mean()),
        **_coast_metrics(error_normalized, batch["target_mask"]),
    }
    date = dataset.dates(chosen)[0]
    _save_preview(
        output_dir / "predictions" / f"preview_step_{step:06d}.png",
        condition,
        target,
        generated_p,
        dataset.product.target_mask,
        dataset.product.base_mask,
        f"{config['name']} · step {step} · {date}",
    )
    if write_netcdf:
        coarsened, valid = ocean_block_mean(generated, batch["target_mask"], 2)
        coarsened_p = coarsened[0, 0].cpu().numpy() * std + mean
        coarsened_p = np.where(valid[0, 0].cpu().numpy().astype(bool), coarsened_p, np.nan)
        _write_netcdf(
            output_dir / "netcdf" / f"sample_step_{step:06d}.nc",
            dataset.product,
            date,
            target,
            generated_p,
            condition,
            coarsened_p,
            {
                "experiment": config["name"],
                "step": step,
                "architecture": (
                    "trainable pretrained bottleneck/decoder; learned "
                    "PixelShuffle 512-to-1024 head"
                    if model.trainable_policy == "decoder_and_head"
                    else "frozen pretrained trunk; learned PixelShuffle "
                    "512-to-1024 head"
                ),
                "hard_block_constraint": "none",
                "sampler": config.get("sampler", "heun"),
                **metrics,
            },
        )
    model.train()
    return metrics


def train(config: dict, smoke_steps: int | None = None, device_name: str | None = None) -> dict:
    config = configure_paths(config)
    is_smoke = smoke_steps is not None
    if is_smoke:
        config = dict(config)
        config["preview_sampler_steps"] = min(2, int(config.get("preview_sampler_steps", 25)))
    device = engine.resolve_device(device_name)
    seed = int(config.get("seed", 42))
    from common import seed_everything

    seed_everything(seed)
    output_dir = Path(config["smoke_output_dir"] if is_smoke else config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization = load_json(config["normalization_cache"])
    product = NOAATransferProduct(config["derived_path"])
    product.verify(normalization)
    train_ranges = config["smoke_date_ranges"] if is_smoke else config["train_date_ranges"]
    validation_ranges = config["smoke_date_ranges"] if is_smoke else config["validation_date_ranges"]
    training = NOAATransferDataset(config, normalization, train_ranges, product)
    validation = NOAATransferDataset(config, normalization, validation_ranges, product)

    model = NOAAFrozenTrunkFlow.from_pretrained(
        config, torch.from_numpy(product.base_mask.astype(np.float32)), device=device
    ).to(device)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, config),
        betas=(0.9, 0.99),
        weight_decay=float(config.get("weight_decay", 1.0e-5)),
    )
    scheduler = engine.build_scheduler(optimizer, config)
    ema = engine.ExponentialMovingAverage(model, float(config.get("ema_decay", 0.999)))
    ema.module.to(device)
    modules = {"model": model, "model_ema": ema}
    step, history, validation_history, initialization = initialize_stage(
        output_dir,
        modules,
        {"model": optimizer},
        {"model": scheduler},
        device,
        config,
    )
    atomic_json(output_dir / "config_used.json", config)
    atomic_json(output_dir / "normalization.json", normalization)
    atomic_json(
        output_dir / "transfer_contract.json",
        {
            "pretrained_ema_path": config["pretrained_ema_path"],
            "initialization": initialization,
            "trainable_policy": model.trainable_policy,
            "encoder_parameters_frozen": True,
            "pretrained_decoder_trainable": model.trainable_policy == "decoder_and_head",
            "bypassed_pretrained_layers": ["output_norm", "output"],
            "trainable_path": (
                "pretrained bottleneck and full decoder plus learned PixelShuffle "
                "512-to-1024 head with direct 1024 flow-state pathway"
                if model.trainable_policy == "decoder_and_head"
                else "learned PixelShuffle 512-to-1024 head with direct 1024 flow-state pathway"
            ),
            "target_mask": "NOAA static 1024x1024 ocean mask",
            "hard_block_constraint": False,
            "coastline_loss_radius_pixels": int(config.get("coastline_radius", 4)),
            "coastline_loss_weight": float(config.get("coastline_loss_weight", 0.5)),
        },
    )

    batch_size = 1 if is_smoke else int(config["batch_size"])
    workers = 0 if is_smoke else int(config.get("num_workers", 0))
    batches = engine.infinite_batches(
        engine.make_loader(training, batch_size, seed, num_workers=workers)
    )
    max_steps = int(
        smoke_steps
        or config.get("stop_after_step", config["max_steps"])
    )
    started = time.monotonic()
    deadline = engine.deadline_from(config, is_smoke, started)
    coast_weight = float(config.get("coastline_loss_weight", 0.5))
    model.train()
    print(
        f"[noaa-v2] device={device} target={product.target_shape} base={product.base_shape} "
        f"train_days={len(training)} step={step}->{max_steps} trainable="
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )
    while step < max_steps and not engine.stop_requested():
        batch = engine.batch_to_device(next(batches), device)
        full_loss, coast_loss = high_resolution_flow_losses(
            model,
            batch["target"],
            batch["condition"],
            batch["target_mask"],
            coast_radius=int(config.get("coastline_radius", 4)),
        )
        total = full_loss + coast_weight * coast_loss
        engine.check_finite(total, step, "direct NOAA flow loss")
        total.backward()
        gradient_norm = engine.clip_and_step(
            model, optimizer, step, float(config.get("gradient_clip", 1.0))
        )
        scheduler.step()
        ema.update(model)
        step += 1
        record = {
            "step": step,
            "total": float(total.detach()),
            "full_ocean_flow": float(full_loss.detach()),
            "coastal_flow": float(coast_loss.detach()),
            "gradient_norm": gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if len(optimizer.param_groups) > 1:
            record["decoder_learning_rate"] = optimizer.param_groups[1]["lr"]
        history.append(record)
        if is_smoke or engine.should_run(step, int(config.get("log_every", 50))):
            engine.log(record)
        preview_every = int(config.get("preview_every", 2000))
        netcdf_every = int(config.get("netcdf_every", 10000))
        if engine.should_run(step, preview_every) or engine.should_run(step, netcdf_every):
            metrics = callbacks(
                ema.module,
                validation,
                device,
                normalization,
                config,
                output_dir,
                step,
                engine.should_run(step, netcdf_every),
            )
            engine.record_validation(validation_history, "validation", metrics, step)
        if engine.should_run(step, int(config.get("checkpoint_every", 5000))):
            engine.save_training_state(
                output_dir,
                step,
                modules,
                {"model": optimizer},
                {"model": scheduler},
                history,
                validation_history,
                config,
                normalization,
            )
        if time.monotonic() >= deadline:
            break
    engine.save_training_state(
        output_dir,
        step,
        modules,
        {"model": optimizer},
        {"model": scheduler},
        history,
        validation_history,
        config,
        normalization,
    )
    metrics = (
        callbacks(ema.module, validation, device, normalization, config, output_dir, step, True)
        if is_smoke
        else {}
    )
    status = engine.finish(
        output_dir,
        config,
        step,
        max_steps,
        history,
        {
            "smoke_test": is_smoke,
            "training_days": len(training),
            "target_shape": list(product.target_shape),
            "frozen_parameters": sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            ),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "loss_keys": ("total", "full_ocean_flow", "coastal_flow"),
            **metrics,
        },
        started,
    )
    training.close()
    validation.close()
    print(json.dumps(status, indent=2), flush=True)
    return status


def main() -> None:
    engine.install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    train(load_config(args.config), args.smoke_steps, args.device)


if __name__ == "__main__":
    main()
