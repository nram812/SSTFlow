#!/usr/bin/env python3
"""Compact multi-scale GAN for fine-detail SST super-resolution (PyTorch).

Training alternates a masked hinge critic step with a generator step whose
objective is

``loss_G = pixel MSE + gradient + spectrum + feature matching + adversarial``

The content term is a **single-sample** masked MSE, deliberately replacing the
ensemble-mean MSE that is common in GAN downscaling papers.  Every reduction is
restricted to ocean pixels, and the critic's logit map is masked as well, so
land can influence neither player.
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
from losses import (
    feature_matching_loss,
    hinge_discriminator_loss,
    hinge_generator_loss,
    masked_gradient_loss,
    masked_mse,
    spectral_amplitude_loss,
)
from model import parameter_count
from model_gan import build_discriminator, build_generator

DATASET_KIND = "super_resolution"


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    """Freeze/unfreeze a player without blocking gradients to its inputs."""
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def validation_metrics(
    generator_module, dataset, device, normalization, derived, config, seed: int
) -> dict:
    generator_module.eval()
    indices = engine.fixed_indices(dataset, int(config.get("validation_samples", 8)))
    batch = engine.batch_to_device(engine.collate_indices(dataset, indices), device)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = generator_module(batch["condition"], batch["mask"], generator=generator)
    content = float(masked_mse(generated, batch["target"], batch["mask"]))
    metrics = field_metrics(
        to_physical(generated, normalization, derived.ocean_mask)[:, 0],
        to_physical(batch["target"], normalization, derived.ocean_mask)[:, 0],
    )
    generator_module.train()
    return {
        "content_mse_normalized": content,
        "samples": int(len(indices)),
        **metrics,
    }


@torch.no_grad()
def run_callbacks(
    generator_module,
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
    generator_module.eval()
    indices = engine.fixed_indices(dataset, int(config.get("preview_samples", 4)))
    batch = engine.batch_to_device(engine.collate_indices(dataset, indices), device)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = generator_module(batch["condition"], batch["mask"], generator=generator)
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
                "model": "conditional GAN",
                "coarsen_factor": int(derived.coarsen_factor),
            },
        )
        print(f"[callback] wrote {path}", flush=True)
    generator_module.train()
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

    generator_module = build_generator(config).to(device)
    discriminator = build_discriminator(config).to(device)
    ema = engine.ExponentialMovingAverage(
        generator_module, float(config.get("ema_decay", 0.999))
    )
    ema.module.to(device)

    generator_optimizer = torch.optim.AdamW(
        generator_module.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1.0e-5)),
        betas=tuple(config.get("generator_betas", (0.0, 0.99))),
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(config.get("discriminator_learning_rate", config["learning_rate"])),
        weight_decay=float(config.get("weight_decay", 1.0e-5)),
        betas=tuple(config.get("discriminator_betas", (0.0, 0.99))),
    )
    generator_scheduler = engine.build_scheduler(generator_optimizer, config)

    modules = {
        "generator": generator_module,
        "discriminator": discriminator,
        "generator_ema": ema,
    }
    optimizers = {
        "generator": generator_optimizer,
        "discriminator": discriminator_optimizer,
    }
    step, history, validation = engine.restore_training_state(
        output_dir, modules, optimizers, {"generator": generator_scheduler}, device
    )

    max_steps = int(smoke_steps or config["max_steps"])
    batch_size = (
        int(config["batch_size"]) if not is_smoke else min(2, int(config["batch_size"]))
    )
    loader = engine.make_loader(
        train_dataset,
        batch_size,
        seed,
        num_workers=0 if is_smoke else int(config.get("num_workers", 0)),
    )
    batches = engine.infinite_batches(loader)

    lambda_content = float(config.get("lambda_content", 10.0))
    lambda_adversarial = float(config.get("lambda_adversarial", 1.0))
    lambda_gradient = float(config.get("lambda_gradient", 0.0))
    lambda_spectral = float(config.get("lambda_spectral", 0.0))
    lambda_feature = float(config.get("lambda_feature_matching", 0.0))
    critic_steps = max(int(config.get("critic_steps", 1)), 1)
    adversarial_start = (
        0 if is_smoke else int(config.get("adversarial_start_step", 2000))
    )

    started = time.monotonic()
    deadline = engine.deadline_from(config, is_smoke, started)
    generator_module.train()
    discriminator.train()
    print(
        f"[train] {config['name']} device={device} "
        f"generator={parameter_count(generator_module):,} "
        f"discriminator={parameter_count(discriminator):,} "
        f"days={len(train_dataset)} grid={derived.shape} steps={step}->{max_steps}",
        flush=True,
    )

    while step < max_steps and not engine.stop_requested():
        adversarial = step >= adversarial_start
        record = {"step": step + 1, "adversarial": bool(adversarial)}

        # ---- critic ----------------------------------------------------
        if adversarial:
            for _ in range(critic_steps):
                batch = engine.batch_to_device(next(batches), device)
                with torch.no_grad():
                    fake = generator_module(batch["condition"], batch["mask"])
                real_logits = discriminator(
                    batch["target"], batch["condition"], batch["mask"]
                )
                fake_logits = discriminator(fake, batch["condition"], batch["mask"])
                critic_loss = hinge_discriminator_loss(real_logits, fake_logits)
                engine.check_finite(critic_loss, step, "critic loss")
                critic_loss.backward()
                engine.clip_and_step(
                    discriminator,
                    discriminator_optimizer,
                    step,
                    float(config.get("gradient_clip", 1.0)),
                )
            record["critic"] = float(critic_loss.detach())
            record["real_logit"] = float(real_logits.detach().mean())
            record["fake_logit"] = float(fake_logits.detach().mean())

        # ---- generator -------------------------------------------------
        batch = engine.batch_to_device(next(batches), device)
        generated = generator_module(batch["condition"], batch["mask"])
        content_loss = masked_mse(generated, batch["target"], batch["mask"])
        engine.check_finite(content_loss, step, "content loss")
        loss = lambda_content * content_loss
        record["content"] = float(content_loss.detach())
        if lambda_gradient:
            gradient_loss = masked_gradient_loss(generated, batch["target"], batch["mask"])
            loss = loss + lambda_gradient * gradient_loss
            record["gradient"] = float(gradient_loss.detach())
        if lambda_spectral:
            spectral_loss = spectral_amplitude_loss(generated, batch["target"], batch["mask"])
            loss = loss + lambda_spectral * spectral_loss
            record["spectral"] = float(spectral_loss.detach())
        if adversarial:
            # The critic supplies d(score)/d(generated), but its own parameters
            # must not receive generator-loss gradients. Leaving those grads
            # in place contaminates the next critic update with the opposite
            # objective and drives both real/fake logits upward.
            set_requires_grad(discriminator, False)
            fake_logits, fake_features, feature_masks = discriminator(
                generated, batch["condition"], batch["mask"], return_features=True
            )
            with torch.no_grad():
                _, real_features, _ = discriminator(
                    batch["target"], batch["condition"], batch["mask"],
                    return_features=True,
                )
            adversarial_loss = hinge_generator_loss(fake_logits)
            engine.check_finite(adversarial_loss, step, "adversarial loss")
            loss = loss + lambda_adversarial * adversarial_loss
            record["adversarial_loss"] = float(adversarial_loss.detach())
            if lambda_feature:
                matching_loss = feature_matching_loss(
                    fake_features, real_features, feature_masks
                )
                loss = loss + lambda_feature * matching_loss
                record["feature_matching"] = float(matching_loss.detach())
        engine.check_finite(loss, step, "generator loss")
        loss.backward()
        gradient_norm = engine.clip_and_step(
            generator_module,
            generator_optimizer,
            step,
            float(config.get("gradient_clip", 1.0)),
        )
        if adversarial:
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
        generator_scheduler.step()
        ema.update(generator_module)
        step += 1

        record.update(
            step=step,
            total=float(loss.detach()),
            gradient_norm=gradient_norm,
            learning_rate=float(generator_scheduler.get_last_lr()[0]),
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
                keys=("total", "content", "gradient", "spectral", "critic"),
            )

        if engine.should_run(step, int(config.get("checkpoint_every", 2000))):
            engine.save_training_state(
                output_dir,
                step,
                modules,
                optimizers,
                {"generator": generator_scheduler},
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
        optimizers,
        {"generator": generator_scheduler},
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
            "generator_parameters": parameter_count(generator_module),
            "discriminator_parameters": parameter_count(discriminator),
            "training_days": len(train_dataset),
            "adversarial_exercised": any(
                record.get("adversarial") for record in history
            ),
            "final_loss": float(np.mean([r["total"] for r in history[-50:]]))
            if history
            else None,
            "loss_keys": ("total", "content", "critic"),
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
