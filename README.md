# SST Downscaling

This repository trains three masked 16× sea-surface-temperature super-resolution models on the OFAM Australian-domain archive: rectified flow matching, autoregressive rectified flow matching with a differentiable one-day rollout, and a conditional PyTorch GAN. Land is represented by a static mask, filled with zero only after normalization, and excluded from every loss.

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

qsub jobs/gpu_smoke.pbs
qsub jobs/train_flow.pbs
qsub jobs/train_flow_ar.pbs
qsub jobs/train_gan.pbs
```

Training resumes from `runs/<experiment>/checkpoint.pt`. PBS training scripts resubmit themselves when `status.json` says `checkpointed` and stop when it says `passed`.

## Evaluate a trained run

```bash
pixi run evaluate-flow -- --samples 32 --sampler-steps 5 10 25
pixi run evaluate-flow-ar
pixi run evaluate-gan
```

Evaluation writes test metrics and NetCDF products under `runs/<experiment>/evaluation/`. Autoregressive evaluation additionally writes a free-running rollout.

## Outputs

- `checkpoint.pt` and named weight files: resumable and inference checkpoints
- `predictions/*.png`: field comparisons, spectra, loss curves, rollout skill
- `netcdf/*.nc`: periodic physical-unit predictions with NaN restored over land
- `metrics/*.json`: validation metrics
- `status.json`: scheduler handshake (`passed` or `checkpointed`)

Do not launch production runs until `pixi run validate-data`, the CPU smoke tests, and `jobs/gpu_smoke.pbs` pass.
