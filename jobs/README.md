# PBS Job Catalogue & Training Guide

This directory contains PBS job submission scripts for the SST downscaling models (16x super-resolution from 32x32 to 512x512 over the Australia OFAM domain).

All commands assume the repository root (`/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling`) as the working directory.

---

## 1. Master Training Job Catalogue

| Model Family | Variant | Smoke Job | Production Training Job | Evaluation Job | Environment | Output Directory |
|---|---|---|---|---|---|---|
| **SRDN** | **SRDCNN** (Baseline, 0.61M) | `srdn_gpu_smoke.pbs` | `srdn_train_full.pbs` (`MODEL=srdcnn`) | `srdn_evaluate.pbs` | `SRDN/venv_srdn_gpu` | `runs/srdn_srdcnn_mask_aware_f16` |
| **SRDN** | **ResAFNO** (Canonical, ~5M) | `srdn_gpu_smoke.pbs` | `srdn_train_full.pbs` (`MODEL=resafno`) | `srdn_evaluate.pbs` | `SRDN/venv_srdn_gpu` | `runs/srdn_resafno_mask_aware_f16` |
| **SRDN** | **Batch-8 Continuation** | — | `srdn_train_batch8.pbs` | `srdn_evaluate.pbs` | `SRDN/venv_srdn_gpu` | `runs/srdn_*_batch8_continue` |
| **GAN** | **GAN-v3** (Hard Consistency) | `gpu_smoke.pbs` | `train_gan_v3.pbs` | `infer_gan_historical_models.pbs` | `.pixi/envs/gpu` | `runs/gan_sr_v3_hard_consistency` |
| **GAN** | **GAN-v2b** (Image-Only Critic) | `gpu_smoke_gan_v2b.pbs` | `train_gan_v2b.pbs` | `infer_gan_historical_models.pbs` | `.pixi/envs/gpu` | `runs/gan_sr_v2b_image_only_critic` |
| **Flow** | **Flow-SR** (Continuous-Time OT-CFM) | `gpu_smoke.pbs` | `train_flow.pbs` / `train_flow_continue.pbs` | `infer_flow_full_test_ab3pc75.pbs` | `.pixi/envs/gpu` | `runs/flow_sr_continue_220k` |
| **Flow** | **Flow-AR** (Autoregressive) | `gpu_smoke_flow_ar_rollout.pbs` | `train_flow_ar.pbs` | `infer_flow_ar_test_year_ab2pc75.pbs` | `.pixi/envs/gpu` | `runs/flow_ar` |
| **Flow** | **Residual-Memory Flow-AR** | `gpu_smoke_flow_ar_residual_memory.pbs` | `train_flow_ar_residual_memory.pbs` | `infer_flow_ar_residual_memory_year_ab2pc75.pbs` | `.pixi/envs/gpu` | `runs/flow_ar_residual_memory` |

---

## 2. Environment Specifications

> [!IMPORTANT]
> The repository uses two distinct Python environments depending on the framework:
> - **TensorFlow SRDN Suite**: Uses `SRDN/venv_srdn_gpu` (Python 3.9 + TensorFlow 2.15.1 + CUDA 12).
> - **PyTorch Production Suite**: Uses Pixi (`.pixi/envs/gpu` or `.pixi/envs/default`, Python 3.12 + PyTorch 2.5).
> 
> Never attempt to run the SRDN jobs with Pixi or PyTorch jobs with `venv_srdn_gpu`.

### A. TensorFlow Environment (`SRDN/venv_srdn_gpu`)
- **Python**: `/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling/SRDN/venv_srdn_gpu/bin/python`
- **Packages**: TensorFlow 2.15.1, CUDA 12 support, xarray, netCDF4, matplotlib
- **Used by**: `srdn_gpu_smoke.pbs`, `srdn_train_pilot.pbs`, `srdn_train_full.pbs`, `srdn_train_batch8.pbs`, `srdn_evaluate.pbs`, and all notebooks in `SRDN/notebooks/`.

### B. PyTorch Environment (Pixi)
- **Python**: Managed via Pixi (`pixi run -e gpu ...` or `.pixi/envs/gpu/bin/python`)
- **Packages**: PyTorch 2.5+, CUDA 12, xarray, netCDF4, dask, matplotlib
- **Used by**: `train_flow*.pbs`, `train_gan*.pbs`, `gpu_smoke*.pbs`, `infer_*.pbs`.

---

## 3. How to Submit Correct Training Jobs

### A. SRDN Models (SRDCNN and ResAFNO)

1. **Pre-flight Gate (Smoke Test)**:
   ```bash
   qsub jobs/srdn_gpu_smoke.pbs
   ```
   Checks forward pass, backward pass, land masking, and checkpoint loading. Output: `runs/smoke/srdn_gpu/report.json`.

2. **10,000-Step Matched Pilot Training**:
   ```bash
   qsub -v MODEL=srdcnn jobs/srdn_train_pilot.pbs
   qsub -v MODEL=resafno jobs/srdn_train_pilot.pbs
   ```

3. **Full Production Training (150,000 steps with auto-resubmit)**:
   ```bash
   qsub -v MODEL=srdcnn jobs/srdn_train_full.pbs
   qsub -v MODEL=resafno jobs/srdn_train_full.pbs
   ```
   The job automatically checkpoints and resubmits itself if the 24-hour walltime expires before reaching `MAX_STEPS`.

4. **Production Batch-8 Continuation (300,000 steps)**:
   ```bash
   qsub -v MODEL=srdcnn,OUTPUT_DIR=runs/srdn_srdcnn_mask_aware_f16_batch8_continue,RESUME_FROM=runs/srdn_srdcnn_mask_aware_f16/model.weights.h5 jobs/srdn_train_batch8.pbs
   qsub -v MODEL=resafno,OUTPUT_DIR=runs/srdn_resafno_mask_aware_f16_batch8_continue,RESUME_FROM=runs/srdn_resafno_mask_aware_f16/model.weights.h5 jobs/srdn_train_batch8.pbs
   ```

5. **Independent Held-Out Evaluation**:
   ```bash
   qsub jobs/srdn_evaluate.pbs
   ```
   Evaluates both models on test set, computing paired MSE bootstrap confidence intervals in `runs/srdn_comparison.json`.

---

### B. GAN Models

1. **Smoke Test**:
   ```bash
   qsub jobs/gpu_smoke.pbs
   ```

2. **Production Training**:
   ```bash
   # Primary hard-consistent GAN-v3:
   qsub jobs/train_gan_v3.pbs
   
   # Image-only critic ablation GAN-v2b:
   qsub jobs/train_gan_v2b.pbs
   ```

3. **Inference & Metrics**:
   ```bash
   qsub jobs/infer_gan_historical_models.pbs
   ```

---

### C. Flow Matching Models

1. **Smoke Test**:
   ```bash
   qsub jobs/gpu_smoke.pbs
   ```

2. **Production Training**:
   ```bash
   # Flow-SR (from scratch):
   qsub jobs/train_flow.pbs
   
   # Flow-SR continuation (step 120k -> 220k):
   qsub jobs/train_flow_continue.pbs
   
   # Autoregressive Flow:
   qsub jobs/train_flow_ar.pbs
   ```

3. **Inference & Rollout**:
   ```bash
   qsub jobs/infer_flow_full_test_ab3pc75.pbs
   qsub jobs/infer_flow_ar_test_year_ab2pc75.pbs
   ```

---

## 4. Monitoring & Diagnostics

- Check running jobs: `qstat -u $USER`
- Inspect live job log: `tail -f logs/<job-name>.o<job-id>`
- Check checkpoint status: `cat runs/<run_name>/status.json`
- Loss curves and preview maps are automatically written to `runs/<run_name>/` during training.
