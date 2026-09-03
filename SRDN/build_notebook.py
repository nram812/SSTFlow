"""Build the validation notebook from the canonical SRDN Python modules."""

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
    markdown(
        """# SRDN ResAFNO: mask-aware 16x validation

This notebook is a thin executable front end to the canonical implementation in
`model_srdn_advanced.py`.  The OFAM experiment is **32x32 -> 512x512 (16x)**.
Both models receive named `coarse_sst`, `coarse_mask`, and `fine_mask` inputs,
use the training-only normalization, return `(B, 512, 512, 1)`, emit exact
zero on land, and enforce valid 16x16 coarse consistency.

The notebook intentionally does not duplicate model definitions.  That avoids
the old notebook/script divergence in which the notebook retained an 8x input,
three upsampling stages, no mask inputs, and an invalid 100%-impulse AFNO test.
"""
    ),
    code(
        """from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

HERE = Path.cwd()
if HERE.name != "SRDN":
    HERE = HERE / "SRDN" if (HERE / "SRDN").exists() else HERE
sys.path.insert(0, str(HERE))

from model_srdn_advanced import (
    AFNO2D,
    CoarseConsistencyProjection,
    SRDCNN_SST_v3,
    SRDN_ResAFNO_v4,
)

print("TensorFlow:", tf.__version__)
print("Python:", sys.executable)
print("GPUs:", tf.config.list_physical_devices("GPU"))
"""
    ),
    markdown(
        """## Model contract and parameter counts

The conventional baseline keeps the transpose-convolution decoder; it has four
2x blocks because this real dataset is 16x.  ResAFNO has four progressive 2x
blocks after its 32x32 spectral trunk.  Parameter-count differences are
reported explicitly and are not evidence that AFNO itself is responsible for
any skill difference.
"""
    ),
    code(
        """baseline = SRDCNN_SST_v3(numHiddenUnits=64, shrink=16)
resafno = SRDN_ResAFNO_v4(numHiddenUnits=128, shrink=16)
print("baseline params:", baseline.count_params())
print("ResAFNO params:", resafno.count_params())
print("baseline input shapes:", [tuple(value.shape) for value in baseline.inputs])
print("ResAFNO input shapes:", [tuple(value.shape) for value in resafno.inputs])
print("outputs:", baseline.output_shape, resafno.output_shape)
"""
    ),
    markdown(
        """## AFNO property checks

AFNO is a global Fourier operation, but a single impulse need not activate
every output pixel.  The meaningful checks are a distant perturbation,
periodic translation equivariance, and finite gradients.
"""
    ),
    code(
        """tf.random.set_seed(42)
afno = AFNO2D(embed_dim=32, num_blocks=4, sparsity_threshold=0.01)
base = tf.random.normal([1, 64, 64, 32])
changed = base.numpy().copy()
changed[0, 4, 7, :] += 1.0
changed = tf.constant(changed)
distant_response = float(tf.abs(afno(changed) - afno(base))[0, 32, 32, 0])
shift = [7, 13]
equivariance_error = float(tf.reduce_max(tf.abs(
    afno(tf.roll(base, shift=shift, axis=[1, 2])) -
    tf.roll(afno(base), shift=shift, axis=[1, 2])
)))
print("distant response:", distant_response)
print("translation-equivariance max error:", equivariance_error)
assert distant_response > 1e-8
assert equivariance_error < 1e-4
"""
    ),
    markdown(
        """## Real OFAM mask/data smoke test

This reads two dates from the immutable source file, validates the derived
mask fingerprint and date axis, checks packed-value decoding, then runs both
models on the actual named-input batch.
"""
    ),
    code(
        """from srdn_data import DerivedProduct, SRDNData

PROJECT = HERE.parent if HERE.name == "SRDN" else HERE
source = PROJECT / "sst_10km_OFAM_historical_Australia.nc"
derived_path = PROJECT / "derived" / "sst_downscaling_f16.nc"
normalization = PROJECT / "reports" / "normalization_f16.json"
derived = DerivedProduct(derived_path)
data = SRDNData(source, derived, normalization, [["2011-01-01", "2011-01-02"]])
inputs, target = data.batch([0, 1])
print({key: value.shape for key, value in inputs.items()}, target.shape)
assert np.isfinite(target).all()
assert np.all(target[inputs["fine_mask"] == 0] == 0)
for model in (baseline, resafno):
    output = model(inputs, training=False)
    assert output.shape == (2, 512, 512, 1)
    assert np.isfinite(output.numpy()).all()
    assert np.all(output.numpy()[inputs["fine_mask"] == 0] == 0)
    print(model.name, "forward passed")
data.close()
"""
    ),
    markdown(
        """## Reproducible training/evaluation entry points

For a short CPU run, use `train_srdn.py` with one of the JSON configurations;
the evaluator streams the full 2011--2014 test period and applies the paired
per-day bootstrap decision rule.  The H200 workflow is in
`jobs/srdn_gpu_smoke.pbs`, `jobs/srdn_train_pilot.pbs`, and
`jobs/srdn_evaluate.pbs`.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {NOTEBOOK_PATH}")
