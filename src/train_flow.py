#!/usr/bin/env python3
"""Plain low-to-high resolution flow matching for sea-surface temperature.

One optimiser step draws a flow time, interpolates between masked noise and the
high-resolution truth, and regresses the velocity - all restricted to ocean
pixels.  Periodic callbacks write NetCDF products, preview figures, and loss
curves, and the run checkpoints itself so a wall-clock limited PBS job can be
resubmitted without losing progress.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import engine
from callbacks import (
    coarse_to_physical,
    field_metrics,
    save_loss_curve,
    save_netcdf,
    save_preview,
    to_physical,
)
from common import load_config
from flow import flow_matching_loss, sample
from losses import masked_mse
from model import build_model, parameter_count

DATASET_KIND = "super_resolution"


@torch.no_grad()
def validation_metrics(
    model, dataset, device, normalization, derived, config, seed: int
) -> dict:
    """Velocity loss plus sampled-field skill on a fixed validation subset."""
    model.eval()
    indices = engine.fixed_indices(dataset, int(config.get("validation_samples", 8)))
    batch = engine.batch_to_device(engine.collate_indices(dataset, indices), device)
    generator = torch.Generator(device=device).manual_seed(seed)

    velocity_loss = float(
        flow_matching_loss(
            model,
            batch["target"],
            batch["condition"],
            batch["mask"],
            generator=generator,
        )
    )
    generated = sample(
        model,
        batch["condition"],
        batch["mask"],
        batch["target"].shape,
        steps=int(config.get("validation_sampler_steps", 10)),
        sampler=str(config.get("sampler", "heun")),
        generator=generator,
        device=device,
    )
    sampled_mse = float(masked_mse(generated, batch["target"], batch["mask"]))
    metrics = field_metrics(
        to_physical(generated, normalization, derived.ocean_mask)[:, 0],
        to_physical(batch["target"], normalization, derived.ocean_mask)[:, 0],
    )
    model.train()
    return {
        "velocity_mse": velocity_loss,
        "sampled_mse_normalized": sampled_mse,
        "samples": int(len(indices)),
        **metrics,
    }


@torch.no_grad()
def run_callbacks(
    model,
    dataset,
    device,
    normalization,
    derived,
    config,
    output_dir: Path,
    step: int,
    seed: int,
    write_netcdf: bool,
) -> dict:
    """Preview figure and, optionally, a NetCDF product for fixed days."""
    model.eval()
    indices = engine.fixed_indices(dataset, int(config.get("preview_samples", 4)))
    batch = engine.batch_to_device(engine.collate_indices(dataset, indices), device)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = sample(
        model,
        batch["condition"],
        batch["mask"],
        batch["target"].shape,
        steps=int(config.get("preview_sampler_steps", 25)),
        sampler=str(config.get("sampler", "heun")),
        generator=generator,
        device=device,
    )
    generated_physical = to_physical(generated, normalization, derived.ocean_mask)[:, 0]
    target_physical = to_physical(batch["target"], normalization, derived.ocean_mask)[
        :, 0
    ]
    coarse_physical = coarse_to_physical(
        batch["condition"], normalization, derived.ocean_mask_lr
    )
    dates = dataset.dates(indices)

    save_preview(
        coarse_physical[0],
        target_physical[0],
        generated_physical[0],
        output_dir / "predictions" / f"preview_step_{step:06d}.png",
        f"{config['name']} · step {step} · {dates[0]}",
        derived.coarsen_factor,
    )
    metrics = field_metrics(generated_physical, target_physical)
    if write_netcdf:
        path = save_netcdf(
            generated_physical,
            target_physical,
            coarse_physical,
            dates,
            derived.lat,
            derived.lon,
            derived.lat_lr,
            derived.lon_lr,
            output_dir / "netcdf" / f"sample_step_{step:06d}.nc",
            {
                "step": int(step),
                "experiment": config["name"],
                "sampler": str(config.get("sampler", "heun")),
                "sampler_steps": int(config.get("preview_sampler_steps", 25)),
                "coarsen_factor": int(derived.coarsen_factor),
            },
        )
        print(f"[callback] wrote {path}", flush=True)
    model.train()
    return metrics


def train(
    config: dict,
    smoke_steps: int | None = None,
    device_name: str | None = None,
) -> dict:
    is_smoke = smoke_steps is not None
    if is_smoke:
        config = engine.smoke_config(config)
    seed, device, output_dir, normalization, derived = engine.prepare(
        config, is_smoke, device_name
    )
    train_dataset, validation_dataset = engine.make_datasets(
        config, normalization, derived, is_smoke, DATASET_KIND
    )

    model = build_model(config).to(device)
    ema = engine.ExponentialMovingAverage(model, float(config.get("ema_decay", 0.999)))
    ema.module.to(device)
    active_learning_rate = float(
        config.get("continuation_learning_rate", config["learning_rate"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=active_learning_rate,
        weight_decay=float(config.get("weight_decay", 1.0e-5)),
        betas=(0.9, 0.99),
    )
    scheduler = engine.build_scheduler(optimizer, config)

    modules = {"model": model, "model_ema": ema}
    step, history, validation = engine.restore_training_state(
        output_dir,
        modules,
        {"model": optimizer},
        {"model": scheduler},
        device,
        resume_from=config.get("resume_from"),
        continuation_learning_rate=active_learning_rate,
    )

    max_steps = int(smoke_steps or config["max_steps"])
    batch_size = int(config["batch_size"]) if not is_smoke else min(
        2, int(config["batch_size"])
    )
    loader = engine.make_loader(
        train_dataset,
        batch_size,
        seed,
        num_workers=0 if is_smoke else int(config.get("num_workers", 0)),
    )
    batches = engine.infinite_batches(loader)

    started = time.monotonic()
    deadline = engine.deadline_from(config, is_smoke, started)
    model.train()
    print(
        f"[train] {config['name']} device={device} "
        f"parameters={parameter_count(model):,} days={len(train_dataset)} "
        f"grid={derived.shape} coarse={derived.coarse_shape} "
        f"steps={step}->{max_steps}",
        flush=True,
    )

    while step < max_steps and not engine.stop_requested():
        batch = engine.batch_to_device(next(batches), device)
        loss = flow_matching_loss(
            model, batch["target"], batch["condition"], batch["mask"]
        )
        engine.check_finite(loss, step, "flow loss")
        loss.backward()
        gradient_norm = engine.clip_and_step(
            model, optimizer, step, float(config.get("gradient_clip", 1.0))
        )
        scheduler.step()
        ema.update(model)
        step += 1

        record = {
            "step": step,
            "total": float(loss.detach()),
            "gradient_norm": gradient_norm,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        if engine.should_run(step, int(config.get("log_every", 50))) or is_smoke:
            engine.log(record)

        if engine.should_run(step, int(config.get("validation_every", 2000))):
            metrics = validation_metrics(
                ema.module,
                validation_dataset,
                device,
                normalization,
                derived,
                config,
                seed + 100 + step,
            )
            engine.record_validation(validation, "validation", metrics, step)
            engine.dump_metrics(output_dir, step, {"validation": metrics})

        preview_every = int(config.get("preview_every", 1000))
        netcdf_every = int(config.get("netcdf_every", 5000))
        if engine.should_run(step, preview_every) or engine.should_run(
            step, netcdf_every
        ):
            run_callbacks(
                ema.module,
                validation_dataset,
                device,
                normalization,
                derived,
                config,
                output_dir,
                step,
                seed + 10_000 + step,
                write_netcdf=engine.should_run(step, netcdf_every),
            )
            save_loss_curve(
                history, output_dir / "predictions" / f"loss_curve_step_{step:06d}.png"
            )

        if engine.should_run(step, int(config.get("checkpoint_every", 2000))):
            engine.save_training_state(
                output_dir,
                step,
                modules,
                {"model": optimizer},
                {"model": scheduler},
                history,
                validation,
                config,
                normalization,
            )
        if time.monotonic() >= deadline:
            print("[runtime] wall-clock guard reached", flush=True)
            break

    engine.save_training_state(
        output_dir,
        step,
        modules,
        {"model": optimizer},
        {"model": scheduler},
        history,
        validation,
        config,
        normalization,
    )
    if is_smoke:
        run_callbacks(
            ema.module,
            validation_dataset,
            device,
            normalization,
            derived,
            config,
            output_dir,
            step,
            seed,
            write_netcdf=True,
        )
    status = engine.finish(
        output_dir,
        config,
        step,
        max_steps,
        history,
        {
            "smoke_test": is_smoke,
            "parameters": parameter_count(model),
            "training_days": len(train_dataset),
            "final_loss": float(np.mean([r["total"] for r in history[-50:]]))
            if history
            else None,
            "loss_keys": ("total",),
        },
        started,
    )
    train_dataset.close()
    validation_dataset.close()
    return status


def main() -> None:
    engine.install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    arguments = parser.parse_args()
    train(
        load_config(arguments.config),
        smoke_steps=arguments.smoke_steps,
        device_name=arguments.device,
    )


if __name__ == "__main__":
    main()
