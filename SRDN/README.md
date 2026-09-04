# SRDN & Downscaling Notebooks Catalogue

This directory contains the TensorFlow downscaling models, canonical ResAFNO and SRDCNN architectures, interactive standard-style Jupyter notebooks, and PBS training workflows for the OFAM Australia sea-surface temperature (SST) downscaling project (16x super-resolution: 32x32 -> 512x512).

---

## 1. Directory Structure

```text
SRDN/
├── notebooks/
│   ├── training/                      # Full interactive standard training notebooks (model.fit style)
│   │   ├── Jupyter_SRDCNN_stand.20260901.ipynb       # Canonical Transpose-Conv baseline (0.61M params)
│   │   ├── Jupyter_SRDCNN_ResAFNO_stand.20260904.ipynb # Canonical ResAFNO model (~5M params: 4,831,569)
│   │   ├── Jupyter_SST_GAN_stand.20260904.ipynb     # RRDB Progressive GAN (1.83M G / 1.39M D)
│   │   └── Jupyter_SST_Flow_stand.20260904.ipynb    # Continuous-Time OT-CFM Flow Matching (5.68M params)
│   └── smoke/                         # Quick smoke-test and evaluation notebooks
│       └── Jupyter_SRDCNN_ResAFNO.20260903.ipynb    # 2-sample validation smoke test
├── evaluation/                        # Test split evaluation and model comparison
│   ├── evaluate_srdn.py               # Comprehensive evaluation script (RMSE, MAE, correlation, PSD)
│   └── compare_srdn.py                # Paired bootstrap comparison between SRDCNN and ResAFNO
├── plotting/                          # Visualization and prediction preview generators
│   ├── plot_srdn_examples.py          # Visual audit of matched predictions
│   └── srdn_previews.py               # Checkpoint preview rendering utility
├── tests/                             # Unit tests, diagnostics, and multi-epoch convergence checks
│   ├── test_cpu_dummy.py              # Synthetic CPU dummy data test
│   ├── test_diagnostics.py            # Unit and property diagnostics (Parseval, equivariance, FiLM, softshrink)
│   ├── test_real_data.py              # Contract and streaming data loader validation
│   └── verify_all_training_convergence.py # Multi-epoch weight-delta & loss-decrease verification
├── model_srdn_advanced.py             # Mask-aware TensorFlow architectures (SRDCNN_SST_v3, SRDN_ResAFNO_v4)
├── train_srdn.py                      # Multi-GPU / CLI production trainer with checkpoints and status.json
├── srdn_data.py                       # High-performance streaming data loaders
├── srdn_metrics.py                    # Metric utilities (RMSE, PSD, bootstrap deltas)
├── srdn_previews.py                   # Compatibility bridge importing plotting/srdn_previews
├── venv_srdn_gpu/                     # GPU virtual environment (Python 3.9 + TensorFlow 2.15.1)
├── venv_srdn/                         # CPU fallback virtual environment
└── README.md                          # This documentation file

*(Note: Notebook generation and executor scripts are centralized in `../tools/notebook_builders/`)*
```

---

## 2. Notebook Models Overview

All standard training notebooks share an interchangeable `dummy()` interface with the exact same data-loading contract, xarray streaming, Horovod/single-GPU execution, and standard Keras `model.fit()` callbacks:

| Notebook | Model Architecture | Parameters | Key Config / Loss Arguments |
|---|---|---|---|
| [`Jupyter_SRDCNN_stand.20260901.ipynb`](notebooks/training/Jupyter_SRDCNN_stand.20260901.ipynb) | 4-level Conv2DTranspose baseline | **608,705 (~0.61M)** | `numHiddenUnits: 64`, `shrink: 16`, L2 reg |
| [`Jupyter_SRDCNN_ResAFNO_stand.20260904.ipynb`](notebooks/training/Jupyter_SRDCNN_ResAFNO_stand.20260904.ipynb) | Spectral AFNO Trunk (6 blocks, 8 modes) + Progressive Upsampling + Coarse Projection | **4,831,569 (~5M)** | `numHiddenUnits: 128`, `trunk_blocks: 6`, `num_freq_blocks: 8`, `sparsity_threshold: 0.01` |
| [`Jupyter_SST_GAN_stand.20260904.ipynb`](notebooks/training/Jupyter_SST_GAN_stand.20260904.ipynb) | RRDB Generator (4 blocks, 24 growth) + 2-Scale PatchCritic | **1,831,249 (G)**<br>**1,385,922 (D)** | `lambda_content: 5.0`, `lambda_adv: 0.05`, `lambda_gradient: 1.0`, `lambda_spectral: 0.2`, `lambda_feature_matching: 1.0` |
| [`Jupyter_SST_Flow_stand.20260904.ipynb`](notebooks/training/Jupyter_SST_Flow_stand.20260904.ipynb) | Continuous-Time OT-CFM Velocity UNet + Sinusoidal Time Emb + GroupNorm + Attention | **5,684,409 (~5.68M)** | `sigma_min: 1e-4`, `lambda_velocity: 1.0`, `sampler: "heun"`, `solver_steps: 10` |

---

## 3. ResAFNO Early-Epoch Behavior & Zero-Prediction Diagnosis

If predictions in the initial 1–2 epochs appear flat or near zero, this is due to the interaction of three specific architectural and data-normalization design principles:

1. **Zero-Initialized Residual Head**:
   `head_conv_detail` is initialized with `kernel_initializer="zeros", bias_initializer="zeros"`. At epoch 0, the high-frequency detail output is mathematically zero, so the model initially predicts only the bilinearly upsampled coarse field (`coarse_skip`). This is intentional to guarantee numerical stability and prevent gradient explosion at step 0.
2. **AFNO Softshrink Sparsity**:
   `AFNO2D` applies complex softshrink:
   $$\text{scale} = \frac{\max(\text{magnitude} - \lambda, 0)}{\max(\text{magnitude}, 10^{-8})}$$
   With initial random weights ($\text{stddev} = 0.02$) and `sparsity_threshold = 0.01`, high-frequency modes with magnitude $< 0.01$ are zeroed out initially. As the optimizer updates the complex Fourier weights, modes pass the threshold and high-frequency details emerge.
3. **Data Normalization & Masking**:
   If target data is standardized using $(y - \mu) / \sigma$ with $\mu \approx 20.65^\circ\text{C}$, ocean pixels near $20.65^\circ\text{C}$ evaluate to $y \approx 0.0$.
   > [!IMPORTANT]
   > Never compute the ocean mask via thresholding normalized values (e.g. `|y| > 1e-7`), because valid ocean pixels near $20.65^\circ\text{C}$ will be treated as land zeros! Always use the authoritative binary land-sea mask `fine_mask`.

---

## 4. Environment Specifications

| Environment | Path / Tool | Packages | Used By |
|---|---|---|---|
| **GPU TensorFlow** | `SRDN/venv_srdn_gpu/` | Python 3.9, TensorFlow 2.15.1, CUDA 12, cuDNN, xarray, netCDF4, matplotlib | All SRDN models (`train_srdn.py`, `evaluate_srdn.py`), all notebooks in `SRDN/notebooks/`, `srdn_*.pbs` jobs |
| **PyTorch Production** | `.pixi/envs/default` (CPU)<br>`.pixi/envs/gpu` (GPU) | Python 3.12, PyTorch 2.5+, xarray, netCDF4, h5netcdf, dask, matplotlib | PyTorch models in `src/` (`train_flow.py`, `train_gan.py`, `evaluate.py`), `train_flow*.pbs`, `train_gan*.pbs` |

---

## 5. PBS Training Job Guide

To run on the NCI Gadi / cluster H200 GPU queue:

### A. Smoke Test (Runs forward, backward, masking, checkpointing in ~2 mins)
```bash
qsub jobs/srdn_gpu_smoke.pbs
```
Inspect: `runs/smoke/srdn_gpu/report.json`.

### B. 10,000-Step Matched Pilots
```bash
qsub -v MODEL=srdcnn jobs/srdn_train_pilot.pbs
qsub -v MODEL=resafno jobs/srdn_train_pilot.pbs
```
Outputs: `runs/srdn_srdcnn_pilot_10k` and `runs/srdn_resafno_pilot_10k`.

### C. Full Production Training (150,000 steps with auto-resubmit)
```bash
qsub -v MODEL=srdcnn jobs/srdn_train_full.pbs
qsub -v MODEL=resafno jobs/srdn_train_full.pbs
```
Outputs: `runs/srdn_srdcnn_mask_aware_f16` and `runs/srdn_resafno_mask_aware_f16`.

### D. Production Batch-8 Continuation (300,000 steps)
```bash
qsub -v MODEL=srdcnn,OUTPUT_DIR=runs/srdn_srdcnn_mask_aware_f16_batch8_continue,RESUME_FROM=runs/srdn_srdcnn_mask_aware_f16/model.weights.h5 jobs/srdn_train_batch8.pbs
qsub -v MODEL=resafno,OUTPUT_DIR=runs/srdn_resafno_mask_aware_f16_batch8_continue,RESUME_FROM=runs/srdn_resafno_mask_aware_f16/model.weights.h5 jobs/srdn_train_batch8.pbs
```

### E. Held-Out Evaluation
```bash
qsub jobs/srdn_evaluate.pbs
```
Generates comprehensive bootstrap comparison in `runs/srdn_comparison.json`.
