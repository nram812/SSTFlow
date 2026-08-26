"""Shared training engine: EMA, checkpoints, wall-clock guards, and callbacks.

The three entrypoints (``train_flow``, ``train_flow_ar``, ``train_gan``) differ
only in how one optimiser step is computed.  Everything around that - dataset
construction, resumption, periodic NetCDF/figures, non-finite guards, status
files - lives here so the experiments stay comparable and only need testing
once.
"""

from __future__ import annotations

import copy
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from callbacks import save_loss_curve, write_metrics
from common import (
    atomic_json,
    atomic_torch_save,
    attach_ocean_mask,
    json_safe,
    load_json,
    restore_rng_state,
    rng_state_payload,
    seed_everything,
)
from data import DerivedProduct, build_dataset

STOP_REQUESTED = False


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[signal] received {signum}; stopping after the current step", flush=True)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def stop_requested() -> bool:
    return STOP_REQUESTED


class ExponentialMovingAverage:
    """Parameter EMA kept on the training device; buffers are copied verbatim."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for source, averaged in zip(model.parameters(), self.module.parameters()):
            averaged.mul_(self.decay).add_(source.detach(), alpha=1.0 - self.decay)
        for source, averaged in zip(model.buffers(), self.module.buffers()):
            averaged.copy_(source)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state) -> None:
        self.module.load_state_dict(state)


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smoke_config(config: dict) -> dict:
    """Bound expensive sampling callbacks while preserving the full grid/model."""
    config = dict(config)
    config["preview_samples"] = min(int(config.get("preview_samples", 4)), 2)
    config["validation_samples"] = min(int(config.get("validation_samples", 8)), 2)
    config["preview_sampler_steps"] = min(int(config.get("preview_sampler_steps", 25)), 2)
    config["validation_sampler_steps"] = min(int(config.get("validation_sampler_steps", 10)), 2)
    config["rollout_sampler_steps"] = min(int(config.get("rollout_sampler_steps", 25)), 2)
    config["rollout_train_steps"] = min(int(config.get("rollout_train_steps", 4)), 2)
    config["rollout_days"] = min(int(config.get("rollout_days", 10)), 2)
    return config


def load_normalization(config: dict) -> dict:
    path = Path(config["normalization_cache"])
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run `pixi run preprocess` before training"
        )
    return load_json(path)


def prepare(config: dict, is_smoke: bool, device_name: str | None):
    """Build the device, normalisation, derived product, and output directory."""
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(device_name)
    output_dir = Path(
        config["smoke_output_dir"] if is_smoke else config["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    normalization = load_normalization(config)
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    normalization = attach_ocean_mask(normalization, derived.ocean_mask)

    atomic_json(output_dir / "config_used.json", config)
    atomic_json(output_dir / "normalization.json", json_safe(normalization))
    return seed, device, output_dir, normalization, derived


def training_ranges(config: dict, is_smoke: bool) -> list[list[str]]:
    return (
        config["smoke_date_ranges"] if is_smoke else config["train_date_ranges"]
    )


def make_loader(
    dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    # netCDF4/HDF5 library state is not safe to inherit after a POSIX fork.
    # Spawn gives every worker a clean interpreter and lets _SourceReader open
    # its own handle, which is slower to start but reliable on long PBS jobs.
    worker_options = ({"multiprocessing_context": "spawn"} if num_workers else {})
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=drop_last and len(dataset) > batch_size,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
        generator=torch.Generator().manual_seed(seed),
        **worker_options,
    )


def infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def check_finite(loss: torch.Tensor, step: int, name: str = "loss") -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite {name} at step {step}: {float(loss)}. "
            "Inspect the normalisation statistics and the land mask."
        )


def clip_and_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    max_norm: float = 1.0,
) -> float:
    """Clip gradients, refuse to apply a non-finite update, and step."""
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(
            f"Non-finite gradient norm at step {step}: {float(gradient_norm)}"
        )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(gradient_norm)


def build_scheduler(optimizer, config: dict):
    """Linear warmup followed by cosine decay to ``min_learning_rate_factor``."""
    warmup = int(config.get("warmup_steps", 500))
    total = int(config.get("max_steps", 100000))
    floor = float(config.get("min_learning_rate_factor", 0.05))

    def schedule(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(path: Path, payload: dict) -> None:
    atomic_torch_save(payload, path)


def save_training_state(
    output_dir: Path,
    step: int,
    modules: dict,
    optimizers: dict,
    schedulers: dict,
    history: list[dict],
    validation: dict,
    config: dict,
    normalization: dict,
) -> None:
    payload = {
        "step": int(step),
        "history": history,
        "validation": validation,
        "config": config,
        "normalization": json_safe(normalization),
        **{f"module_{name}": module.state_dict() for name, module in modules.items()},
        **{
            f"optimizer_{name}": optimizer.state_dict()
            for name, optimizer in optimizers.items()
        },
        **{
            f"scheduler_{name}": scheduler.state_dict()
            for name, scheduler in schedulers.items()
            if scheduler is not None
        },
        **rng_state_payload(),
    }
    save_checkpoint(output_dir / "checkpoint.pt", payload)
    for name, module in modules.items():
        atomic_torch_save(module.state_dict(), output_dir / f"{name}.pt")
    atomic_json(
        output_dir / "training_history.json",
        {"step": int(step), "history": history, "validation": validation},
    )
    print(f"[checkpoint] saved step {step}", flush=True)


def restore_training_state(
    output_dir: Path,
    modules: dict,
    optimizers: dict,
    schedulers: dict,
    device: torch.device,
):
    """Resume from ``checkpoint.pt`` when one exists; returns the new state."""
    path = output_dir / "checkpoint.pt"
    if not path.is_file():
        return 0, [], {}
    state = torch.load(path, map_location=device, weights_only=False)
    for name, module in modules.items():
        module.load_state_dict(state[f"module_{name}"])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state[f"optimizer_{name}"])
    for name, scheduler in schedulers.items():
        key = f"scheduler_{name}"
        if scheduler is not None and key in state:
            scheduler.load_state_dict(state[key])
    restore_rng_state(state)
    step = int(state["step"])
    print(f"[resume] step {step} from {path}", flush=True)
    return step, list(state.get("history", [])), dict(state.get("validation", {}))


def should_run(step: int, every: int) -> bool:
    return every > 0 and step % every == 0


def finish(
    output_dir: Path,
    config: dict,
    step: int,
    max_steps: int,
    history: list[dict],
    extra: dict,
    started: float,
) -> dict:
    save_loss_curve(
        history,
        output_dir / "predictions" / "loss_curve_final.png",
        keys=tuple(extra.get("loss_keys", ("total",))),
    )
    status = {
        "status": "passed" if step >= max_steps else "checkpointed",
        "experiment": config["name"],
        "step": int(step),
        "max_steps": int(max_steps),
        "elapsed_seconds": time.monotonic() - started,
        **{key: value for key, value in extra.items() if key != "loss_keys"},
    }
    atomic_json(output_dir / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return status


def log(record: dict) -> None:
    print(f"[train] {json.dumps(record, sort_keys=True)}", flush=True)


def deadline_from(config: dict, is_smoke: bool, started: float) -> float:
    hours = 0.25 if is_smoke else float(config.get("max_runtime_hours", 23.0))
    return started + hours * 3600.0


def make_datasets(config: dict, normalization: dict, derived, is_smoke: bool, kind: str):
    """Training, validation, and callback datasets for one experiment."""
    ranges = training_ranges(config, is_smoke)
    train = build_dataset(
        config,
        normalization,
        ranges,
        kind,
        derived=derived,
        preload=bool(config.get("preload", False)) and not is_smoke,
    )
    validation_ranges = (
        config["smoke_date_ranges"] if is_smoke else config["validation_date_ranges"]
    )
    validation = build_dataset(
        config, normalization, validation_ranges, kind, derived=derived, preload=False
    )
    return train, validation


def fixed_indices(dataset, count: int) -> np.ndarray:
    """Evenly spaced, deterministic sample indices for previews and metrics."""
    count = max(1, min(int(count), len(dataset)))
    return np.linspace(0, len(dataset) - 1, count).astype(int)


def collate_indices(dataset, indices) -> dict:
    """Stack individual dataset items into a batch without a DataLoader."""
    items = [dataset[int(index)] for index in indices]
    batch = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch


def record_validation(
    validation: dict, name: str, metrics: dict, step: int
) -> None:
    metrics = {**metrics, "step": int(step)}
    validation.setdefault(name, []).append(metrics)
    print(f"[validation] {name}: {json.dumps(metrics, sort_keys=True)}", flush=True)


def dump_metrics(output_dir: Path, step: int, payload: dict) -> None:
    write_metrics(output_dir / "metrics" / f"metrics_step_{step:06d}.json", payload)
