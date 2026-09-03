# PBS job catalogue

The job directory preserves production provenance, so it contains more files
than a new user normally needs.  The tables below identify the supported entry
points.  Run CPU tests and the matching H200 smoke job before any production
submission.  All commands assume the repository root as the working directory.

## Core 0.1° models

| Purpose | Smoke / gate | Production or inference |
|---|---|---|
| Historical Flow-SR | `gpu_smoke.pbs` | `train_flow.pbs`, `infer_flow_full_test_ab3pc75.pbs`, `infer_access_cm2_periods.pbs` |
| Historical autoregressive flow | `gpu_smoke_flow_ar_rollout.pbs` | `train_flow_ar.pbs`, `infer_flow_ar_test_year_ab2pc75.pbs` |
| Residual-memory Flow-AR | `gpu_smoke_flow_ar_residual_memory.pbs` | `train_flow_ar_residual_memory.pbs`, `infer_flow_ar_residual_memory_year_ab2pc75.pbs` |
| Coarse-balanced legacy AR ablation | `gpu_smoke_flow_ar_legacy_coarse_balanced.pbs` | `train_flow_ar_legacy_coarse_balanced.pbs`, `infer_flow_ar_legacy_coarse_balanced_year_ab2pc75.pbs` |
| Historical GAN-v2 | `gpu_smoke.pbs` | `train_gan_v2.pbs`, `infer_gan_historical_models.pbs` |
| Image-only-critic GAN-v2b | `gpu_smoke_gan_v2b.pbs` | `train_gan_v2b.pbs`, `infer_gan_historical_models.pbs` |
| Hard-consistent GAN-v3 | `gpu_smoke.pbs` | `train_gan_v3.pbs`, `infer_gan_historical_models.pbs` |

## Historical + future continuations

| Purpose | Smoke / gate | Production or inference |
|---|---|---|
| Prepare combined OFAM mapping | — | `preprocess_combined_hist_rcp85.pbs` |
| Combined Flow-SR | `gpu_smoke_combined.pbs` | `train_flow_combined_hist_rcp85.pbs`, `infer_combined_flow_test_access.pbs` |
| Combined GAN-v2/v2b/v3 | `gpu_smoke_gan_hist_rcp85.pbs` | `train_gan_v2_hist_rcp85.pbs`, `train_gan_v2b_hist_rcp85.pbs`, `train_gan_v3_hist_rcp85.pbs`, then `infer_gan_hist_rcp85_models.pbs` |

## NOAA 0.05° transfer

| Stage | Smoke / gate | Production or inference |
|---|---|---|
| Prepare NOAA predictor/target mapping | — | `preprocess_noaa_5km.pbs` |
| Frozen-trunk head stage | `gpu_smoke_noaa_5km_v2.pbs` | `train_flow_noaa_5km_v2.pbs` |
| Decoder-unfrozen stage | `gpu_smoke_noaa_5km_decoder.pbs` | `train_flow_noaa_5km_decoder_38k.pbs` |
| Low-rate continuation to 150k | `gpu_smoke_noaa_5km_decoder_150k.pbs` | `train_flow_noaa_5km_decoder_150k.pbs` |
| Final inference | `profile_noaa_5km_inference.pbs` | `infer_noaa_5km_test_150k.pbs`, `infer_noaa_5km_access_150k.pbs` |

## Evaluation and rendering

- `generate_all_figures_and_animations.pbs`: full figure/animation pipeline.
- `render_all_deliverables.pbs`: rebuild static deliverables.
- `render_all_animations_shortq.pbs`: render animations on the short queue.
- `render_flow_ar_legacy_coarse_balanced_animation.pbs`: one specific AR animation.
- `postprocess_coarse_balanced_ar_comparison.pbs`: refresh AR comparison metrics.

## SRDN 16x ResAFNO validation

| Stage | Launcher | Output / gate |
|---|---|---|
| CPU diagnostics | `SRDN/test_diagnostics.py`, `SRDN/test_real_data.py` | mask, AFNO, projection, and real-data contract |
| H200 gate | `srdn_gpu_smoke.pbs` | `runs/smoke/srdn_gpu/report.json` |
| 10k matched pilots | `srdn_train_pilot.pbs` with `MODEL=srdcnn` and `MODEL=resafno` | `runs/srdn_*_pilot_10k` |
| Full training | `srdn_train_full.pbs` with `MODEL=srdcnn` and `MODEL=resafno` | `runs/srdn_*_mask_aware_f16` |
| Held-out evaluation | `srdn_evaluate.pbs` | `runs/srdn_comparison.json` |

These jobs use `SRDN/environment.md` and deliberately require the separate
`venv_srdn_gpu`; the old notebook `venv_srdn` is CPU-only and incomplete.

## Diagnostic or superseded launchers

Files named `check_*`, `profile_*`, `preflight_*`, `bridge_*`, or
`smoke_then_*` are reproducibility and debugging tools rather than the normal
starting point. `train_gan.pbs` targets the retired first GAN and should not be
used for new science. Prefer GAN-v2 or v3 and read
[the GAN experiment guide](../docs/gan_experiment_guide.md).

Check job state with `qstat`; inspect `logs/<job-name>.o<job-id>`; and confirm
`runs/<experiment>/status.json` plus the independent validation JSON before
using a product in analysis.
