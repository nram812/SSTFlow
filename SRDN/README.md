# SRDN ResAFNO validation

This directory contains the mask-aware TensorFlow SRDN experiment for the
OFAM Australia product.  The real problem is **32x32 -> 512x512 (16x)**, not
the 8x default in the original notebook.

## Model contract

Both `SRDCNN_SST_v3` and `SRDN_ResAFNO_v4` accept the same named inputs:

```text
coarse_sst:  (B, 32, 32, 1), normalized SST with invalid cells zero-filled
coarse_mask: (B, 32, 32, 1), valid coarse-cell indicator
fine_mask:   (B, 512, 512, 1), valid high-resolution ocean indicator
```

Both return exactly `(B, 512, 512, 1)`.  Outputs are exactly zero on land and,
when enabled, the valid fine-grid mean in each valid 16x16 block is projected
to the corresponding normalized coarse SST.  Training uses masked MSE and the
normalization in `../reports/normalization_f16.json`, fitted only on 1979-2008.

The conventional baseline preserves the original three 7x7 stride-2
transpose convolutions.  ResAFNO adds the canonical shared channel-block AFNO
trunk, FiLM conditioning, and progressive bilinear/refinement upsampling.  The
AFNO layer is not claimed to make every pixel nonzero for an impulse; its
diagnostics test controlled nonlocal response, translation equivariance,
finite gradients, and spectral behavior instead.

## Validation commands

From the `2026/SSTDownscaling` root:

```bash
SRDN/venv_srdn_cpu/bin/python -m pytest SRDN/test_diagnostics.py SRDN/test_real_data.py
SRDN/venv_srdn_cpu/bin/python SRDN/test_cpu_dummy.py
```

The real-data loader streams target days from
`sst_10km_OFAM_historical_Australia.nc` while loading only the small derived
coarse predictor and static masks.  It never rewrites the raw NetCDF file.

## Training and evaluation

The reproducible configurations are:

```text
configs/srdn_srdcnn.json
configs/srdn_resafno.json
```

For a local short run:

```bash
PYTHONPATH=SRDN SRDN/venv_srdn_cpu/bin/python SRDN/train_srdn.py \
  --config configs/srdn_resafno.json --device cpu --max-steps 3 \
  --output-dir runs/smoke/local_resafno
```

After a checkpoint exists:

```bash
PYTHONPATH=SRDN SRDN/venv_srdn_cpu/bin/python SRDN/evaluate_srdn.py \
  --run runs/smoke/local_resafno --config configs/srdn_resafno.json
```

Evaluation reports ocean-only RMSE/MAE/bias, daily paired MSE, skill against
consistent bilinear interpolation, spatial correlation, coastal/interior
errors, spectral log-power error, and nonfinite/land leakage checks.  Use
`compare_srdn.py` after both learned models are evaluated; ResAFNO is labelled
better only when the paired per-day 95% bootstrap upper confidence limit for
`ResAFNO - SRDCNN` MSE is below zero.

## PBS workflow

1. Install `venv_srdn_gpu` as described in `environment.md`.
2. From the project root submit `qsub jobs/srdn_gpu_smoke.pbs`.
3. If the gate passes, submit one pilot per model, for example:

```bash
qsub -v MODEL=srdcnn jobs/srdn_train_pilot.pbs
qsub -v MODEL=resafno jobs/srdn_train_pilot.pbs
```

4. Only if both 10,000-step pilots are stable should the 120,000-step jobs and
`jobs/srdn_evaluate.pbs` be submitted.

The GPU smoke job hard-fails if TensorFlow exposes no GPU and checks forward,
backward, exact masking, coarse consistency, and checkpoint reload for both
variants.  The full trainer writes `status.json` and self-resubmits after a
walltime checkpoint in the same pattern as the existing project jobs.
