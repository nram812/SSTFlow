"""H200 gate for both mask-aware SRDN model variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SRDN"))

from model_srdn_advanced import SRDCNN_SST_v3, SRDN_ResAFNO_v4  # noqa: E402
from srdn_data import DerivedProduct, SRDNData  # noqa: E402
from train_srdn import build_model, load_config, masked_mse  # noqa: E402


def _projection_error(output, inputs, derived):
    values = output.numpy()[0, ..., 0]
    fine_mask = inputs["fine_mask"][0, ..., 0].astype(bool)
    coarse_mask = inputs["coarse_mask"][0, ..., 0].astype(bool)
    coarse = inputs["coarse_sst"][0, ..., 0]
    block = values.reshape(
        derived.coarse_shape[0], derived.coarsen_factor,
        derived.coarse_shape[1], derived.coarsen_factor,
    )
    block_mask = fine_mask.reshape(
        derived.coarse_shape[0], derived.coarsen_factor,
        derived.coarse_shape[1], derived.coarsen_factor,
    )
    means = (block * block_mask).sum(axis=(1, 3)) / np.maximum(
        block_mask.sum(axis=(1, 3)), 1
    )
    valid = coarse_mask & (block_mask.sum(axis=(1, 3)) > 0)
    return float(np.max(np.abs(means[valid] - coarse[valid])))


def run_variant(config_path: Path, data, derived, device):
    config = load_config(config_path)
    inputs, target = data.batch([0])
    inputs = {key: tf.convert_to_tensor(value) for key, value in inputs.items()}
    target = tf.convert_to_tensor(target)
    with tf.device(device):
        model = build_model(config)
        output = model(inputs, training=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-4)
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            loss = masked_mse(prediction, target, inputs["fine_mask"])
        gradients = tape.gradient(loss, model.trainable_variables)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError(f"{config['model_variant']} has an unconnected gradient")
        grad_norm = tf.linalg.global_norm(gradients)
        tf.debugging.check_numerics(output, "non-finite model output")
        tf.debugging.check_numerics(loss, "non-finite model loss")
        tf.debugging.check_numerics(grad_norm, "non-finite model gradient")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        saved_output = model(inputs, training=False)

        checkpoint_path = ROOT / "runs" / "smoke" / "srdn_gpu" / config["model_variant"]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = tf.train.Checkpoint(model=model)
        prefix = checkpoint.write(str(checkpoint_path))

        reloaded = build_model(config)
        reloaded(inputs, training=False)
        tf.train.Checkpoint(model=reloaded).restore(prefix).expect_partial()
        reloaded_output = reloaded(inputs, training=False)

    mask = inputs["fine_mask"].numpy() == 0.0
    return {
        "model_variant": config["model_variant"],
        "parameters": int(model.count_params()),
        "output_shape": list(output.shape),
        "loss": float(loss.numpy()),
        "gradient_norm": float(grad_norm.numpy()),
        "land_leakage_max": float(np.max(np.abs(output.numpy()[mask]))),
        "coarse_consistency_max": _projection_error(saved_output, {k: v.numpy() for k, v in inputs.items()}, derived),
        "reload_max_abs_difference": float(
            np.max(np.abs(saved_output.numpy() - reloaded_output.numpy()))
        ),
        "checkpoint_prefix": prefix,
    }


def main():
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("H200 gate failed: TensorFlow sees no GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("H200 gate failed: TensorFlow exposed no logical GPU")
    device = logical[0].name
    derived = DerivedProduct(ROOT / "derived" / "sst_downscaling_f16.nc")
    data = SRDNData(
        ROOT / "sst_10km_OFAM_historical_Australia.nc",
        derived,
        ROOT / "reports" / "normalization_f16.json",
        [["2011-01-01", "2011-01-01"]],
    )
    results = {
        "tensorflow_version": tf.__version__,
        "build_info": tf.sysconfig.get_build_info(),
        "physical_gpus": [device.name for device in physical],
        "logical_gpus": [device.name for device in logical],
        "variants": [],
    }
    for name in ("srdn_srdcnn.json", "srdn_resafno.json"):
        results["variants"].append(
            run_variant(ROOT / "configs" / name, data, derived, device)
        )
    data.close()
    output = ROOT / "runs" / "smoke" / "srdn_gpu" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
