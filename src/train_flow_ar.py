#!/usr/bin/env python3
"""Autoregressive flow-matching super-resolution with single-step rollout.

Each sample is a consecutive day pair.  The model predicts the high-resolution
field ``y(t+1)`` from the coarse predictor ``x(t+1)`` **and** the previous
high-resolution state ``y(t)``.

Two objectives are combined:

``velocity``  the ordinary masked flow-matching loss with teacher forcing;
``rollout``   a differentiable *single-step* rollout - the sampler is unrolled
              for a few steps, the generated day is compared with the truth in
              state space, and the gradient flows through the whole ODE solve.

The rollout term is switched on after ``rollout_start_step`` and weighted by
``rollout_weight`` so early training is cheap and stable.
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
    save_rollout_netcdf,
    save_rollout_skill_plot,
    to_physical,
)
from common import load_config
from flow import (
    flow_matching_loss,
    rollout,
    sample,
    single_step_rollout_loss,
)
from losses import masked_mse
from model import build_model, parameter_count

DATASET_KIND = "autoregressive"


def rollout_active(config: dict, step: int, is_smoke: bool) -> bool:
    if float(config.get("rollout_weight", 0.0)) <= 0.0:
        return False
    if is_smoke:
        return True
    start = int(config.get("rollout_start_step", 20000))
    every = max(int(config.get("rollout_every", 4)), 1)
    return step >= start and step % every == 0


@torch.no_grad()
def validation_metrics(
    model, dataset, device, normalization, derived, config, seed: int
) -> dict:
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
            previous_state=batch["previous"],
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
        previous_state=batch["previous"],
        generator=generator,
        device=device,
    )
    sampled_mse = float(masked_mse(generated, batch["target"], batch["mask"]))
    persistence_mse = float(
        masked_mse(batch["previous"], batch["target"], batch["mask"])
    )
    metrics = field_metrics(
        to_physical(generated, normalization, derived.ocean_mask)[:, 0],
        to_physical(batch["target"], normalization, derived.ocean_mask)[:, 0],
    )
    model.train()
    return {
        "velocity_mse": velocity_loss,
        "sampled_mse_normalized": sampled_mse,
        "persistence_mse_normalized": persistence_mse,
        "skill_vs_persistence": float(
            1.0 - sampled_mse / max(persistence_mse, 1.0e-12)
        ),
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
        previous_state=batch["previous"],
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
    dates = [window[-1] for window in (dataset.date_window(int(i)) for i in indices)]

    save_preview(
        coarse_physical[0],
        target_physical[0],
        generated_physical[0],
        output_dir / "predictions" / f"preview_step_{step:06d}.png",
        f"{config['name']} · step {step} · {dates[0]} (single step)",
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
                "mode": "single_step_teacher_forced",
            },
        )
        print(f"[callback] wrote {path}", flush=True)
    model.train()
    return metrics


@torch.no_grad()
def free_running_rollout(
    model,
    config: dict,
    normalization: dict,
    derived,
    device,
    output_dir: Path,
    step: int,
    seed: int,
) -> dict:
    """Generate a multi-day free-running forecast from a single truth state."""
    from data import AutoregressiveSuperResolutionDataset

    days = int(config.get("rollout_days", 10))
    dataset = AutoregressiveSuperResolutionDataset(
        config,
        normalization,
        config["validation_date_ranges"],
        horizon=days,
        derived=derived,
        preload=False,
    )
    try:
        batch = engine.batch_to_device(engine.collate_indices(dataset, [0]), device)
        generator = torch.Generator(device=device).manual_seed(seed)
        model.eval()
        predictions = rollout(
            model,
            batch["previous"],
            batch["conditions"],
            batch["mask"],
            steps=int(config.get("rollout_sampler_steps", 25)),
            sampler=str(config.get("sampler", "heun")),
            generator=generator,
        )[0]
        targets = batch["targets"][0]
        generated_physical = to_physical(
            predictions, normalization, derived.ocean_mask
        )[:, 0]
        target_physical = to_physical(targets, normalization, derived.ocean_mask)[:, 0]
        dates = dataset.date_window(0)[1:]
        path = save_rollout_netcdf(
            generated_physical,
            target_physical,
            dates,
            derived.lat,
            derived.lon,
            output_dir / "netcdf" / f"rollout_step_{step:06d}.nc",
            {
                "step": int(step),
                "experiment": config["name"],
                "days": int(days),
                "initial_state_date": dataset.date_window(0)[0],
                "sampler": str(config.get("sampler", "heun")),
                "sampler_steps": int(config.get("rollout_sampler_steps", 25)),
                "mode": "free_running",
            },
        )
        save_rollout_skill_plot(
            generated_physical,
            target_physical,
            output_dir / "predictions" / f"rollout_skill_step_{step:06d}.png",
            f"{config['name']} · free-running rollout · step {step}",
        )
        print(f"[callback] wrote {path}", flush=True)
        error = generated_physical - target_physical
        metrics = {
            "days": int(days),
            "rmse_by_lead": [
                float(value)
                for value in np.sqrt(np.nanmean(np.square(error), axis=(1, 2)))
            ],
            "bias_by_lead": [
                float(value) for value in np.nanmean(error, axis=(1, 2))
            ],
        }
        model.train()
        return metrics
    finally:
        dataset.close()


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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1.0e-5)),
        betas=(0.9, 0.99),
    )
    scheduler = engine.build_scheduler(optimizer, config)
    modules = {"model": model, "model_ema": ema}
    step, history, validation = engine.restore_training_state(
        output_dir, modules, {"model": optimizer}, {"model": scheduler}, device
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
    rollout_loader = engine.make_loader(
        train_dataset,
        int(config.get("rollout_batch_size", 2)),
        seed + 1,
        num_workers=0,
    )
    rollout_batches = engine.infinite_batches(rollout_loader)

    started = time.monotonic()
    deadline = engine.deadline_from(config, is_smoke, started)
    model.train()
    print(
        f"[train] {config['name']} device={device} "
        f"parameters={parameter_count(model):,} pairs={len(train_dataset)} "
        f"grid={derived.shape} coarse={derived.coarse_shape} "
        f"steps={step}->{max_steps}",
        flush=True,
    )

    while step < max_steps and not engine.stop_requested():
        batch = engine.batch_to_device(next(batches), device)
        velocity_loss = flow_matching_loss(
            model,
            batch["target"],
            batch["condition"],
            batch["mask"],
            previous_state=batch["previous"],
        )
        engine.check_finite(velocity_loss, step, "velocity loss")
        loss = velocity_loss
        record = {"step": step + 1, "velocity": float(velocity_loss.detach())}

        if rollout_active(config, step, is_smoke):
            rollout_batch = engine.batch_to_device(next(rollout_batches), device)
            rollout_mse, _ = single_step_rollout_loss(
                model,
                rollout_batch["previous"],
                rollout_batch["condition"],
                rollout_batch["target"],
                rollout_batch["mask"],
                steps=int(config.get("rollout_train_steps", 4)),
            )
            engine.check_finite(rollout_mse, step, "rollout loss")
            weight = float(config.get("rollout_weight", 0.0))
            loss = loss + weight * rollout_mse
            record["rollout"] = float(rollout_mse.detach())
            record["rollout_weight"] = weight

        engine.check_finite(loss, step, "total loss")
        loss.backward()
        gradient_norm = engine.clip_and_step(
            model, optimizer, step, float(config.get("gradient_clip", 1.0))
        )
        scheduler.step()
        ema.update(model)
        step += 1

        record.update(
            step=step,
            total=float(loss.detach()),
            gradient_norm=gradient_norm,
            learning_rate=float(scheduler.get_last_lr()[0]),
        )
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
                history,
                output_dir / "predictions" / f"loss_curve_step_{step:06d}.png",
                keys=("total", "velocity", "rollout"),
            )

        if engine.should_run(step, int(config.get("rollout_netcdf_every", 10000))):
            metrics = free_running_rollout(
                ema.module,
                config,
                normalization,
                derived,
                device,
                output_dir,
                step,
                seed + 20_000 + step,
            )
            engine.record_validation(validation, "free_running_rollout", metrics, step)

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
        free_running_rollout(
            ema.module,
            {**config, "rollout_days": 2, "rollout_sampler_steps": 2},
            normalization,
            derived,
            device,
            output_dir,
            step,
            seed,
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
            "training_pairs": len(train_dataset),
            "rollout_exercised": any("rollout" in record for record in history),
            "final_loss": float(np.mean([r["total"] for r in history[-50:]]))
            if history
            else None,
            "loss_keys": ("total", "velocity", "rollout"),
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
