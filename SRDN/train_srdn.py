"""Train a mask-aware deterministic SRDCNN or ResAFNO model."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_srdn_advanced import (
    CoarseConsistencyProjection,
    SRDCNN_SST_v3,
    SRDN_ResAFNO_v4,
)
from srdn_data import DerivedProduct, SRDNData
from srdn_metrics import denormalize, masked_field_metrics, write_json
from srdn_previews import save_prediction_preview


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    root = path.parent.parent
    for key in ("source_path", "derived_path", "normalization_path", "output_dir", "smoke_output_dir"):
        if key in config:
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else root / value)
    config["config_path"] = str(path)
    return config


def configure_device(device_name: str | None):
    device_name = device_name or "cpu"
    if device_name == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        return "/CPU:0"
    if device_name != "cuda":
        raise ValueError("device must be cpu or cuda")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("CUDA requested but TensorFlow sees no GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return "/GPU:0"


def build_model(config: dict):
    kwargs = {
        "numHiddenUnits": int(config.get("hidden_units", 128)),
        "numLats": int(config.get("fine_height", 512)),
        "numLongs": int(config.get("fine_width", 512)),
        "shrink": int(config.get("coarsen_factor", 16)),
        "enforce_coarse_consistency": bool(
            config.get("enforce_coarse_consistency", True)
        ),
    }
    if config["model_variant"] == "srdcnn":
        return SRDCNN_SST_v3(**kwargs)
    if config["model_variant"] == "resafno":
        kwargs.update(
            {
                "trunk_blocks": int(config.get("trunk_blocks", 6)),
                "num_freq_blocks": int(config.get("num_freq_blocks", 8)),
                "afno_sparsity_threshold": float(
                    config.get("afno_sparsity_threshold", 0.01)
                ),
            }
        )
        return SRDN_ResAFNO_v4(**kwargs)
    raise ValueError(f"unknown model_variant: {config['model_variant']}")


def masked_mse(prediction, target, mask):
    mask = tf.cast(mask, prediction.dtype)
    difference = tf.square(prediction - target) * mask
    return tf.reduce_sum(difference) / tf.maximum(tf.reduce_sum(mask), 1.0e-8)


@tf.function
def train_step(model, optimizer, inputs, target):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = masked_mse(prediction, target, inputs["fine_mask"])
        regularization = tf.add_n(model.losses) if model.losses else 0.0
        total = loss + regularization
    gradients = tape.gradient(total, model.trainable_variables)
    gradients, gradient_norm = tf.clip_by_global_norm(gradients, 1.0)
    tf.debugging.check_numerics(total, "non-finite SRDN loss")
    tf.debugging.check_numerics(gradient_norm, "non-finite SRDN gradient norm")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total, gradient_norm


def learning_rate(config: dict, step: int) -> float:
    warmup = int(config.get("warmup_steps", 500))
    total = max(int(config.get("max_steps", 10000)), warmup + 1)
    floor = float(config.get("min_learning_rate_factor", 0.05))
    if step < warmup:
        return float(config["learning_rate"]) * (step + 1) / max(warmup, 1)
    progress = min(max((step - warmup) / float(total - warmup), 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(config["learning_rate"]) * (floor + (1.0 - floor) * cosine)


def validation_metrics(model, dataset: SRDNData, positions: np.ndarray):
    inputs, target = dataset.batch(positions)
    prediction = model(inputs, training=False).numpy()
    mask = inputs["fine_mask"][..., 0].astype(bool)
    mean, std = dataset.mean, dataset.std
    prediction = denormalize(prediction[..., 0], mean, std)
    target = denormalize(target[..., 0], mean, std)
    return masked_field_metrics(prediction, target, mask)


def save_validation_preview(
    model,
    model_label: str,
    dataset: SRDNData,
    derived: DerivedProduct,
    positions: np.ndarray,
    run_dir: Path,
    step: int,
    device: str,
    enforce_coarse_consistency: bool,
) -> None:
    """Write a fixed validation example so checkpoint images are comparable."""
    position = int(positions[0])
    inputs, target_normalized = dataset.batch([position])
    tensor_inputs = {
        key: tf.convert_to_tensor(value) for key, value in inputs.items()
    }
    with tf.device(device):
        prediction_normalized = model(tensor_inputs, training=False).numpy()
        bilinear_normalized = tf.image.resize(
            tensor_inputs["coarse_sst"], derived.fine_shape, method="bilinear"
        ) * tensor_inputs["fine_mask"]
        if enforce_coarse_consistency:
            bilinear_normalized = CoarseConsistencyProjection(derived.coarsen_factor)(
                [
                    bilinear_normalized,
                    tensor_inputs["coarse_sst"],
                    tensor_inputs["coarse_mask"],
                    tensor_inputs["fine_mask"],
                ]
            )
    mask = inputs["fine_mask"][0, ..., 0].astype(bool)
    target = denormalize(target_normalized[0, ..., 0], dataset.mean, dataset.std)
    bilinear = denormalize(bilinear_normalized.numpy()[0, ..., 0], dataset.mean, dataset.std)
    prediction = denormalize(prediction_normalized[0, ..., 0], dataset.mean, dataset.std)
    save_prediction_preview(
        run_dir / "predictions" / f"prediction_step_{step:06d}.png",
        target,
        bilinear,
        prediction,
        mask,
        derived.lat,
        derived.lon,
        f"{model_label} validation example — step {step:,} — "
        f"{dataset.dates[position]}",
    )


def save_checkpoint(manager, run_dir: Path, model, optimizer, step):
    checkpoint_path = manager.save(checkpoint_number=int(step))
    model_path = run_dir / "model.weights.h5"
    temporary = run_dir / "model.weights.partial.h5"
    model.save_weights(temporary)
    os.replace(temporary, model_path)
    return checkpoint_path


def train(
    config: dict,
    smoke_steps: int | None = None,
    device_name: str | None = None,
    max_steps_override: int | None = None,
    output_dir_override: str | Path | None = None,
    resume_from_override: str | Path | None = None,
):
    device = configure_device(device_name or config.get("device", "cpu"))
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if output_dir_override is not None:
        run_dir = Path(output_dir_override)
        if not run_dir.is_absolute():
            run_dir = Path(config["config_path"]).parent.parent / run_dir
    else:
        run_dir = Path(
            config["smoke_output_dir"] if smoke_steps is not None else config["output_dir"]
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["effective_output_dir"] = str(run_dir.resolve())
    if resume_from_override is not None:
        config["resume_from"] = str(Path(resume_from_override).resolve())
    write_json(run_dir / "config_used.json", config)
    write_json(run_dir / "status.json", {"status": "running", "step": 0})

    derived = DerivedProduct(config["derived_path"])
    train_data = SRDNData(
        config["source_path"], derived, config["normalization_path"],
        config["train_date_ranges"],
    )
    validation_data = SRDNData(
        config["source_path"], derived, config["normalization_path"],
        config["validation_date_ranges"],
    )
    model = build_model(config)
    first_inputs, first_target = train_data.batch([0])
    with tf.device(device):
        # Build all variables before restoring optimizer/model state.
        model(first_inputs, training=False)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(config["learning_rate"]),
            beta_1=0.9,
            beta_2=0.999,
        )
        optimizer.build(model.trainable_variables) if hasattr(optimizer, "build") else None
        step_variable = tf.Variable(0, dtype=tf.int64, trainable=False, name="step")
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=optimizer, step=step_variable
        )
        manager = tf.train.CheckpointManager(
            checkpoint, str(run_dir / "checkpoints"), max_to_keep=3
        )
        # A continuation can write to a new run directory while restoring from
        # an earlier run.  Once that continuation has checkpointed, prefer its
        # own latest checkpoint so PBS resubmissions do not restart from the
        # original source checkpoint.
        restore_path = manager.latest_checkpoint
        if restore_path is None and resume_from_override is not None:
            resume_path = Path(resume_from_override)
            if resume_path.is_dir():
                restore_path = tf.train.latest_checkpoint(str(resume_path))
            else:
                restore_path = str(resume_path)
                if restore_path.endswith(".index"):
                    restore_path = restore_path[:-len(".index")]
            if not restore_path or not Path(restore_path + ".index").exists():
                raise FileNotFoundError(
                    f"no resume checkpoint found at {resume_path}"
                )
        if restore_path:
            checkpoint.restore(restore_path).expect_partial()
            print(f"[resume] {restore_path}", flush=True)

    step = int(step_variable.numpy())
    start_step = step
    max_steps = int(
        smoke_steps
        if smoke_steps is not None
        else max_steps_override
        if max_steps_override is not None
        else config["max_steps"]
    )
    history = []
    validation_positions = validation_data.random_positions(
        int(config.get("validation_samples", 8)), seed=seed + 100
    )
    started = time.monotonic()
    deadline = started + 3600.0 * float(config.get("max_runtime_hours", 23.0))
    batch_size = int(config.get("batch_size", 2))
    batches_per_epoch = max(1, (len(train_data) + batch_size - 1) // batch_size)
    epoch, batch_offset = divmod(step, batches_per_epoch)
    preview_every = int(config.get("preview_every", 0))

    try:
        while step < max_steps:
            made_batch = False
            for batch_index, (inputs, target) in enumerate(
                train_data.iter_epoch(batch_size, seed, epoch)
            ):
                if batch_index < batch_offset:
                    continue
                if step >= max_steps:
                    break
                made_batch = True
                with tf.device(device):
                    optimizer.learning_rate.assign(learning_rate(config, step))
                    total, gradient_norm = train_step(
                        model,
                        optimizer,
                        {key: tf.convert_to_tensor(value) for key, value in inputs.items()},
                        tf.convert_to_tensor(target),
                    )
                    step += 1
                    step_variable.assign(step)
                record = {
                    "step": step,
                    "loss": float(total.numpy()),
                    "gradient_norm": float(gradient_norm.numpy()),
                    "learning_rate": float(optimizer.learning_rate.numpy()),
                }
                history.append(record)
                if step % int(config.get("log_every", 50)) == 0 or smoke_steps:
                    print(
                        f"[train] {config['model_variant']} step={step} "
                        f"loss={record['loss']:.6f} grad={record['gradient_norm']:.4f}",
                        flush=True,
                    )
                if step % int(config.get("validation_every", 1000)) == 0:
                    metrics = validation_metrics(model, validation_data, validation_positions)
                    print(f"[validation] {json.dumps(metrics, sort_keys=True)}", flush=True)
                    write_json(run_dir / f"validation_step_{step:06d}.json", metrics)
                if preview_every > 0 and step % preview_every == 0:
                    save_validation_preview(
                        model,
                        config["model_variant"].upper(),
                        validation_data,
                        derived,
                        validation_positions,
                        run_dir,
                        step,
                        device,
                        bool(config.get("enforce_coarse_consistency", True)),
                    )
                    print(
                        f"[preview] wrote {run_dir / 'predictions' / f'prediction_step_{step:06d}.png'}",
                        flush=True,
                    )
                if step % int(config.get("checkpoint_every", 1000)) == 0:
                    save_checkpoint(manager, run_dir, model, optimizer, step)
                    write_json(run_dir / "training_history.json", {"history": history})
                if time.monotonic() >= deadline:
                    break
            if not made_batch:
                raise RuntimeError("training dataset yielded no batches")
            epoch += 1
            batch_offset = 0
            if time.monotonic() >= deadline:
                break
    except Exception as error:
        write_json(run_dir / "status.json", {
            "status": "failed", "step": int(step), "error": repr(error)
        })
        train_data.close(); validation_data.close()
        raise

    save_checkpoint(manager, run_dir, model, optimizer, step)
    write_json(run_dir / "training_history.json", {"history": history})
    status = "passed" if step >= max_steps else "checkpointed"
    write_json(run_dir / "status.json", {
        "status": status,
        "start_step": int(start_step),
        "step": int(step),
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "batches_per_epoch": int(batches_per_epoch),
        "effective_epochs": float(step / batches_per_epoch),
        "model_variant": config["model_variant"],
        "parameters": int(model.count_params()),
        "elapsed_seconds": time.monotonic() - started,
    })
    train_data.close(); validation_data.close()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config batch_size; useful for matched continuation runs.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint prefix or checkpoint directory to restore when the output directory is new.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        config["batch_size"] = args.batch_size
    train(
        config,
        args.smoke_steps,
        args.device,
        args.max_steps,
        args.output_dir,
        args.resume_from,
    )


if __name__ == "__main__":
    main()
