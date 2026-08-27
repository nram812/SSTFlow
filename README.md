# SSTFlow

This repository trains three masked 16× sea-surface-temperature super-resolution models on the OFAM Australian-domain archive: rectified flow matching, coarse-authoritative autoregressive rectified flow matching with bounded fine-scale lag guidance, and a conditional PyTorch GAN. Land is represented by a static mask, filled with zero only after normalization, and excluded from every loss.

The complete design rationale, data forensics, task ledger, test matrix, and operational assumptions are in [plan.md](plan.md).

## Reproduce the environment

```bash
pixi install
pixi run test
```

The default environment is CPU-only. GPU commands use `pixi run -e gpu ...` and are intended for an H200 compute node.

## Prepare and validate data

The source file is expected at the absolute `source_path` recorded in each config.

```bash
pixi run preprocess
pixi run validate-data
```

Preprocessing creates `derived/sst_downscaling_f16.nc` and `reports/normalization_f16.json`. It never copies the 512×512 target archive; it materializes only the 32×32 mask-aware coarse predictor and streaming training statistics.

## Smoke tests and production

```bash
pixi run smoke-flow
pixi run smoke-flow-ar
pixi run smoke-gan

# Longer full-grid flow check plus an explicit coarse-conditioning ablation:
qsub jobs/flow_1000_smoke.pbs

qsub jobs/gpu_smoke.pbs
qsub jobs/train_flow.pbs
qsub jobs/train_flow_ar.pbs
qsub jobs/train_gan.pbs
```

Training resumes from `runs/<experiment>/checkpoint.pt`. The corrected AR experiment writes to `runs/flow_ar_coarse_guided` and intentionally does not resume the retired `runs/flow_ar` weights. PBS training scripts resubmit themselves when `status.json` says `checkpointed` and stop when it says `passed`.

## Evaluate a trained run

```bash
pixi run evaluate-flow -- --samples 32 --sampler-steps 5 10 25
pixi run evaluate-flow-ar
pixi run evaluate-gan

# Entire 2011-01-01 through 2014-12-31 flow_sr test set, 50-step Heun, on H200:
qsub jobs/infer_flow_full_test.pbs

# Converted ACCESS-CM2, two exact ten-year windows (1980-1989 and 2080-2089):
qsub jobs/infer_access_cm2_periods.pbs
```

Evaluation writes test metrics and NetCDF products under `runs/<experiment>/evaluation/`. Autoregressive evaluation additionally writes a free-running diagnostic sequence. Each valid generated ocean block is projected onto the current day's coarse mean; the lag field contributes only its within-block anomaly.

## Outputs

- `checkpoint.pt` and named weight files: resumable and inference checkpoints
- `predictions/*.png`: field comparisons, spectra, loss curves, rollout skill
- `netcdf/*.nc`: periodic physical-unit predictions with NaN restored over land
- `metrics/*.json`: validation metrics
- `status.json`: scheduler handshake (`passed` or `checkpointed`)

Do not launch production runs until `pixi run validate-data`, the CPU smoke tests, and `jobs/gpu_smoke.pbs` pass.

## ACCESS-CM2 operational workflow

ACCESS conversion and model inference are separate. The tracked standalone
converter is `derived/convert_access_to_training_grid.py`; its NetCDF output is
not versioned. The inference program performs no remapping and refuses any
input whose coordinates or static mask differ from the training predictor.

All operational choices are recorded in `configs/access_cm2_inference.json`.
The current production configuration uses the completed `runs/flow_sr` EMA
weights, the AB3/AM3 predictor-corrector with 75 steps, seed 42, and two
3,653-day windows: 1980-01-01–1989-12-31 and
2080-01-01–2089-12-31. See
`docs/access_cm2_operational_inference.md` for the conversion method, grid
contract, restart behavior, and output schema.

```bash
# Recreate the converted predictor when needed (protects an existing output):
pixi run convert-access

# Submit both configured periods on H200:
qsub jobs/infer_access_cm2_periods.pbs
```

## Repository boundary

Git contains the Python source, tests, JSON configurations, PBS launchers,
documentation, `pixi.toml`, and the locked dependency graph. Raw/derived
NetCDF, weights, run products, figures, notebook checkpoints, and scheduler
logs are intentionally excluded. Consequently, a clone reproduces the code
and environment; the documented source datasets and trained weights must be
available at their configured paths to reproduce numerical products.
