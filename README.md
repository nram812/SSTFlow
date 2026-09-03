# SSTFlow

This repository trains three masked 16× sea-surface-temperature super-resolution models on the OFAM Australian-domain archive: rectified flow matching, coarse-authoritative autoregressive rectified flow matching with bounded fine-scale lag guidance, and a conditional PyTorch GAN. Land is represented by a static mask, filled with zero only after normalization, and excluded from every loss.

The complete design rationale, data forensics, task ledger, test matrix, and operational assumptions are in [plan.md](plan.md).

## Where to start

| If you want to… | Start here |
|---|---|
| Reproduce the environment and basic workflow | This README |
| Understand a model or deployment method | [Documentation map](docs/README.md) |
| Choose the correct PBS launcher | [Job catalogue](jobs/README.md) |
| Change or ablate GAN losses safely | [GAN experiment guide](docs/gan_experiment_guide.md) |
| Continue the editable paper draft | [Skeleton manuscript source](paper/SST_Downscaling_Skeleton_Paper.md) or `SST_Downscaling_Skeleton_Paper.docx` |
| Audit every engineering decision and test | [Living project plan](plan.md) |
| Rebuild climate-change figures and tables | `analysis/generate_climate_change_evaluation.py` |

The active reference models are `flow_sr`, `flow_ar_residual_memory`,
`gan_sr_v2`, `gan_sr_v2b_image_only_critic`, and
`gan_sr_v3_hard_consistency`, plus their explicitly named continuations. Avoid
unnumbered legacy GAN/AR runs unless reproducing a documented failure.
Scientifically rejected experiments are quarantined under
[`unsuccessful_experiments/`](unsuccessful_experiments/README.md) and are not
part of the supported workflow or publication analysis.

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
pixi run smoke-gan-v2
pixi run smoke-gan-v2b
pixi run smoke-gan-v3

# Longer full-grid flow check plus an explicit coarse-conditioning ablation:
qsub jobs/flow_1000_smoke.pbs

qsub jobs/gpu_smoke.pbs
qsub jobs/gpu_smoke_gan_v2b.pbs
qsub jobs/train_flow.pbs
qsub jobs/train_flow_ar.pbs
qsub jobs/train_gan_v2.pbs
qsub jobs/train_gan_v2b.pbs
qsub jobs/train_gan_v3.pbs
```

Training resumes from `runs/<experiment>/checkpoint.pt`. The corrected AR experiment writes to `runs/flow_ar_coarse_guided` and intentionally does not resume the retired `runs/flow_ar` weights. PBS training scripts resubmit themselves when `status.json` says `checkpointed` and stop when it says `passed`.

The legacy `runs/flow_ar` checkpoint can still be reproduced for diagnosis, but
must be loaded with its saved full-state, uncapped lag pathway. A dedicated
one-year free-running driver, AB2 predictor/Adams-Moulton corrector, H200 gate,
and independent product validator are documented in
`docs/flow_ar_rollout_audit.md`:

```bash
qsub jobs/gpu_smoke_flow_ar_rollout.pbs
# Submit only after the smoke report passes:
qsub jobs/infer_flow_ar_test_year_ab2pc75.pbs
```

The legacy `runs/gan_sr` weights must not be resumed: two consecutive
zero-initialized projections trapped that generator at bilinear interpolation
plus a spatially constant bias. The corrected RRDB experiment is
`configs/gan_sr_v2.json`, writes only to `runs/gan_sr_v2`, and is launched with
`jobs/train_gan_v2.pbs`.

GAN-v2b is a controlled discriminator-input ablation of GAN-v2. Its two
PatchGAN critics receive only the masked high-resolution SST image: neither the
coarse SST/mask condition nor the high-resolution mask is concatenated as an
input channel. The high-resolution mask remains outside the critic to set land
to zero and remove invalid patches from adversarial reductions. All other
generator, loss, optimizer, schedule, and callback settings are identical to
GAN-v2. Its config, job, and independent output directory are
`configs/gan_sr_v2b_image_only_critic.json`, `jobs/train_gan_v2b.pbs`, and
`runs/gan_sr_v2b_image_only_critic`.

GAN-v3 keeps the corrected GAN-v2 objective and architecture, then applies an
exact differentiable projection to every generated field. On each valid 16x16
coarse-ocean block, the projection adds the blockwise difference between the
input coarse SST and the generated ocean mean. Thus re-coarsening the generated
field reproduces the conditioning SST to floating-point precision; fine land is
still zero and invalid coarse blocks are left unchanged. Its independent config,
job, and outputs are `configs/gan_sr_v3_hard_consistency.json`,
`jobs/train_gan_v3.pbs`, and `runs/gan_sr_v3_hard_consistency`.

## Combined historical and RCP8.5 continuation

The combined flow experiment uses each date exactly once: historical OFAM for
1979-2014 and RCP8.5 for 2015-2101. It deliberately drops the overlapping
2006-2014 RCP8.5 copies rather than weighting that climate interval twice. The
derived product stores only the 32x32 predictor plus source-file/source-index
mapping; 512x512 targets continue to be read from the immutable source files.
It retains the historical training normalization required by the pretrained
220,000-step model.

```bash
pixi run preprocess-combined
qsub jobs/gpu_smoke_combined.pbs
qsub jobs/train_flow_combined_hist_rcp85.pbs
```

Training resumes from `runs/flow_sr_continue_220k/checkpoint.pt`, performs
100,000 additional updates, and writes only to
`runs/flow_sr_combined_hist_rcp85_continue_320k`. The train ranges are
1979-2008 and 2015-2095; validation is 2009-2010 and 2096-2097; testing is
2011-2014 and 2098-2101. None of these sets overlap.

The three corrected GANs use the same combined predictor/target mapping and
split contract. Each forks its completed historical step-120,000 checkpoint,
continues for 100,000 updates at a constant `5e-6` learning rate, and writes to
a distinct `*_hist_rcp85_continue_220k` directory:

```bash
qsub jobs/gpu_smoke_gan_hist_rcp85.pbs
qsub jobs/train_gan_v2_hist_rcp85.pbs
qsub jobs/train_gan_v2b_hist_rcp85.pbs
qsub jobs/train_gan_v3_hist_rcp85.pbs
```

The fork restores generator, generator EMA, discriminator, both AdamW states,
training history, and RNG state. It never writes into the historical run.

## Evaluate a trained run

```bash
pixi run evaluate-flow -- --samples 32 --sampler-steps 5 10 25
pixi run evaluate-flow-ar
pixi run evaluate-gan-v2
pixi run evaluate-gan-v2b
pixi run evaluate-gan-v3

# Entire 2011-01-01 through 2014-12-31 flow_sr test set, 50-step Heun, on H200:
qsub jobs/infer_flow_full_test.pbs

# Converted ACCESS-CM2, two exact ten-year windows (1980-1989 and 2080-2089):
qsub jobs/infer_access_cm2_periods.pbs

# Direct GAN inference for all completed historical models:
qsub jobs/infer_gan_historical_models.pbs

# Submit after all combined GAN continuations have passed:
qsub jobs/infer_gan_hist_rcp85_models.pbs
```

Evaluation writes test metrics and NetCDF products under `runs/<experiment>/evaluation/`. Autoregressive evaluation additionally writes a free-running diagnostic sequence. Each valid generated ocean block is projected onto the current day's coarse mean; the lag field contributes only its within-block anomaly.

GAN inference is a single generator forward pass (`sampler=direct`, one step),
using the EMA generator and one latent field seeded from each absolute input-time
index. The ACCESS products retain the same converted 32x32 predictor, exact
1980-1989 and 2080-2089 windows, resumable atomic writer, static masks, and
provenance as the flow products. `src/validate_gan_inference.py` checks dates,
finiteness, physical ranges, departure from bilinear interpolation, and—for
GAN-v3—blockwise agreement with the coarse input.

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

## NOAA 0.05° transfer

The active satellite-transfer experiment uses the completed combined-climate
OFAM flow only as a frozen 512×512 feature trunk. Its old output layers are
bypassed and a learned PixelShuffle head predicts the complete 1024×1024 NOAA
flow velocity directly. It does not resize a generated 0.1° SST field or impose
a 2×2 block-mean constraint. See
`docs/noaa_5km_frozen_trunk_1024_methods.md` for the coastal-failure audit and
full method.

```bash
qsub -q h200q jobs/gpu_smoke_noaa_5km_v2.pbs
qsub -q h200q jobs/train_flow_noaa_5km_v2.pbs
```

Outputs are isolated under `runs/flow_sr_noaa_5km_frozen_trunk_1024`.

The active broader-decoder continuation is checksum-forked at step 80,000 and
trains to step 150,000:

```bash
qsub -q h200q jobs/gpu_smoke_noaa_5km_decoder_150k.pbs
qsub -q h200q jobs/train_flow_noaa_5km_decoder_150k.pbs
# After status.json reports passed at exactly step 150000:
qsub -q h200q jobs/infer_noaa_5km_test_150k.pbs
qsub -q h200q jobs/infer_noaa_5km_access_150k.pbs
```

The test product uses the NOAA 2021-2023 holdout. ACCESS products use the same
converted 32x32 1980-1989 and 2080-2089 conditions as the 0.1-degree models but
generate on the fixed NOAA 1024x1024 target grid. See
`docs/noaa_5km_frozen_trunk_1024_methods.md`.

## Residual-memory autoregressive flow

The replacement for the memory-dominated legacy AR model starts exactly from
the completed `flow_sr` EMA, freezes that current-SST backbone, and trains only
a bounded fine-scale lag correction:

```bash
qsub -q h200q jobs/gpu_smoke_flow_ar_residual_memory.pbs
qsub -q h200q jobs/train_flow_ar_residual_memory.pbs
```

Its run is isolated at `runs/flow_ar_residual_memory`; details and the legacy
failure analysis are in `docs/flow_ar_residual_memory.md`.

## Repository boundary

Git contains the Python source, tests, JSON configurations, PBS launchers,
documentation, `pixi.toml`, and the locked dependency graph. Raw/derived
NetCDF, weights, run products, figures, notebook checkpoints, and scheduler
logs are intentionally excluded. Consequently, a clone reproduces the code
and environment; the documented source datasets and trained weights must be
available at their configured paths to reproduce numerical products.
