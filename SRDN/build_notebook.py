"""Build the runnable SRDN notebook in the style of the standard notebook."""

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("Jupyter_SRDCNN_ResAFNO.20260903.ipynb")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    code(
        """# Environment checks, following the standard SRDCNN notebook entry cell.
from pathlib import Path
import shutil
import subprocess
import sys

print("Python:", sys.executable)
print("Python version:", sys.version.split()[0])
print("nvidia-smi:", shutil.which("nvidia-smi"))
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "-L"], check=False)
"""
    ),
    code(
        """# Imports and the user-editable project configuration.
import json
import sys
import numpy as np
import tensorflow as tf

HERE = Path.cwd()
if (HERE / "configs").exists() and (HERE / "SRDN").exists():
    PROJECT = HERE
elif HERE.name == "SRDN" and (HERE.parent / "configs").exists():
    PROJECT = HERE.parent
else:
    raise FileNotFoundError("Run this notebook from the SSTDownscaling project or its SRDN directory")
sys.path.insert(0, str(PROJECT / "SRDN"))

from model_srdn_advanced import AFNO2D, CoarseConsistencyProjection
from srdn_data import DerivedProduct, SRDNData
from srdn_metrics import denormalize
from train_srdn import build_model, load_config

CONFIG_PATH = PROJECT / "configs" / "srdn_resafno.json"
BASELINE_CONFIG_PATH = PROJECT / "configs" / "srdn_srdcnn.json"
CONFIG = load_config(CONFIG_PATH)
BASELINE_CONFIG = load_config(BASELINE_CONFIG_PATH)
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("ResAFNO configuration:", json.dumps(CONFIG, indent=2))
"""
    ),
    code(
        """# Standard-style model/data function: configure, build, read, and validate.
def dummy(sample_start="2011-01-01", sample_stop="2011-01-02"):
    derived = DerivedProduct(CONFIG["derived_path"])
    data = SRDNData(
        CONFIG["source_path"],
        derived,
        CONFIG["normalization_path"],
        [[sample_start, sample_stop]],
    )
    positions = np.arange(min(2, len(data)), dtype=np.int64)
    inputs, target = data.batch(positions)
    models = {
        "SRDCNN": build_model(BASELINE_CONFIG),
        "ResAFNO": build_model(CONFIG),
    }
    outputs = {}
    for name, model in models.items():
        output = model(inputs, training=False)
        outputs[name] = output.numpy()
        print(name, "parameters:", model.count_params())
        print(name, "input shapes:", [tuple(value.shape) for value in model.inputs])
        print(name, "output shape:", output.shape)
        assert output.shape == (len(positions), 512, 512, 1)
        assert np.isfinite(output.numpy()).all()
        assert np.all(output.numpy()[inputs["fine_mask"] == 0] == 0)
    assert np.isfinite(target).all()
    assert np.all(target[inputs["fine_mask"] == 0] == 0)

    # AFNO checks use controlled properties rather than an invalid claim that
    # a single impulse must activate every output pixel.
    tf.random.set_seed(42)
    afno = AFNO2D(embed_dim=32, num_blocks=4, sparsity_threshold=0.01)
    base = tf.random.normal([1, 64, 64, 32])
    changed = base.numpy().copy()
    changed[0, 4, 7, :] += 1.0
    distant_response = float(tf.abs(afno(changed) - afno(base))[0, 32, 32, 0])
    shift = [7, 13]
    equivariance_error = float(tf.reduce_max(tf.abs(
        afno(tf.roll(base, shift=shift, axis=[1, 2]))
        - tf.roll(afno(base), shift=shift, axis=[1, 2])
    )))
    print("AFNO distant response:", distant_response)
    print("AFNO translation-equivariance error:", equivariance_error)
    assert distant_response > 1e-8
    assert equivariance_error < 1e-4

    mean, std = data.mean, data.std
    dates = list(data.dates[: len(positions)])
    data.close()
    return {
        "derived": derived,
        "inputs": inputs,
        "target": target,
        "outputs": outputs,
        "models": models,
        "mean": mean,
        "std": std,
        "dates": dates,
    }
"""
    ),
    code(
        """# Run the same kind of explicit execution cell as the standard notebook.
results = dummy()
print("Real-data SRDN smoke test passed")
"""
    ),
    code(
        """# Example predictions: target, bilinear reference, both learned models, and errors.
import matplotlib.pyplot as plt

derived = results["derived"]
inputs = results["inputs"]
mask = inputs["fine_mask"][0, ..., 0].astype(bool)
target = denormalize(results["target"][0, ..., 0], results["mean"], results["std"])
target = np.where(mask, target, np.nan)
coarse = tf.image.resize(
    tf.convert_to_tensor(inputs["coarse_sst"][:1]), derived.fine_shape, method="bilinear"
)
coarse = CoarseConsistencyProjection(derived.coarsen_factor)([
    coarse * inputs["fine_mask"][:1],
    inputs["coarse_sst"][:1],
    inputs["coarse_mask"][:1],
    inputs["fine_mask"][:1],
]).numpy()[0, ..., 0]
bilinear = np.where(mask, denormalize(coarse, results["mean"], results["std"]), np.nan)
fields = [target, bilinear]
names = ["Target", "Bilinear"]
for name in ("SRDCNN", "ResAFNO"):
    prediction = denormalize(results["outputs"][name][0, ..., 0], results["mean"], results["std"])
    fields.append(np.where(mask, prediction, np.nan))
    names.append(name)
values = np.concatenate([field[np.isfinite(field)] for field in fields])
vmin, vmax = np.quantile(values, [0.01, 0.99])
errors = [field - target for field in fields[1:]]
error_limit = max(float(np.quantile(np.abs(np.concatenate([e[np.isfinite(e)] for e in errors])), 0.99)), 0.05)
extent = [derived.lon.min(), derived.lon.max(), derived.lat.min(), derived.lat.max()]
figure, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
for axis, field, name in zip(axes[0], fields, names):
    image = axis.imshow(np.ma.masked_invalid(field), origin="lower", extent=extent, cmap="turbo", vmin=vmin, vmax=vmax)
    axis.set_title(name)
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
figure.colorbar(image, ax=axes[0].tolist(), label="SST (°C)", shrink=0.8)
error_names = ["Bilinear − target", "SRDCNN − target", "ResAFNO − target", "ResAFNO − SRDCNN"]
error_fields = errors + [fields[3] - fields[2]]
for axis, field, name in zip(axes[1], error_fields, error_names):
    image = axis.imshow(np.ma.masked_invalid(field), origin="lower", extent=extent, cmap="RdBu_r", vmin=-error_limit, vmax=error_limit)
    axis.set_title(name)
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
figure.colorbar(image, ax=axes[1].tolist(), label="difference (°C)", shrink=0.8)
figure.suptitle("SRDN real-data example — " + str(results["dates"][0]))
plt.show()
"""
    ),
    code(
        """# Inspect the source NetCDF in the same final data-inspection style.
import netCDF4

with netCDF4.Dataset(CONFIG["source_path"], "r") as dataset:
    print(dataset)
    print("\\nVariables:")
    print(list(dataset.variables))
    print("temp shape:", dataset.variables["temp"].shape)
    print("time range:", dataset.variables["Time"][0], dataset.variables["Time"][-1])
print("derived fine mask:", derived.ocean_mask.shape, "ocean fraction:", derived.ocean_mask.mean())
print("derived coarse predictor:", derived.sst_lr.shape)
"""
    ),
    markdown(
        """## Training and evaluation entry points

The notebook is a runnable validation front end; long jobs use the same
configuration files and canonical Python modules:

```bash
SRDN/venv_srdn_gpu/bin/python SRDN/train_srdn.py --config configs/srdn_resafno.json --device cuda
SRDN/venv_srdn_gpu/bin/python SRDN/evaluate_srdn.py --run runs/srdn_resafno_mask_aware_f16 --config configs/srdn_resafno.json
```

The current configuration is 32×32 → 512×512 (16×), mask-aware, and writes
prediction PNGs every 15,000 training steps under `runs/.../predictions/`.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "SRDN TensorFlow",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {NOTEBOOK_PATH}")
