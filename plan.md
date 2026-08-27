# SST Super-Resolution: Flow Matching, Autoregressive Flow Matching, and GAN

**Project root:** `/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling`
**Status of this document:** living plan — Part A records what is already built, Part B records what remains, Part C is the exhaustive test matrix.

---

## 0. Executive summary

We build three models that map a **coarse (low-resolution) sea-surface temperature field** to the **native 0.1° (~10 km) OFAM field** over the Australian domain:

| # | Experiment | Entry point | Config | Description |
|---|---|---|---|---|
| 1 | **Flow matching super-resolution** | `src/train_flow.py` | `configs/flow_sr.json` | Rectified flow, coarse → fine, no temporal memory. Analogous to the NZ CPM flow model but low→high resolution. |
| 2 | **Autoregressive flow matching** | `src/train_flow_ar.py` | `configs/flow_ar.json` | Current-day coarse SST is authoritative; the previous high-resolution state contributes only bounded within-block fine-scale guidance. |
| 3 | **Conditional GAN** | `src/train_gan.py` | `configs/gan_sr.json` | PyTorch conditional GAN with a masked hinge PatchGAN critic and a **single-sample masked MSE** content loss (explicitly *not* an ensemble-mean MSE). |

Everything is masked: **the loss is computed over ocean pixels only.** Land is NaN in the source data, is replaced by zeros after normalisation, and is excluded from every reduction.

### The five requirements and how they are met

| Requirement (from the brief) | Where it is implemented |
|---|---|
| "normal flow matching low to high resolution like the CPM flow matching" | `src/train_flow.py`, `src/flow.py`, `src/model.py::SuperResolutionFlowUNet` |
| "autoregressive flow matching" | `src/train_flow_ar.py`, `src/model.py::AutoregressiveSuperResolutionFlowUNet`, `src/consistency.py` |
| "one GAN implementation … adapt code to pytorch … single mse loss as opposed to an ensemble MSE loss" | `src/model_gan.py`, `src/train_gan.py` (content term is `masked_mse(one_sample, truth)`) |
| "netcdf outputs every so often and do illustrations after a certain number of epochs and save weights in a similar manner" | `src/callbacks.py`, `src/engine.py`; controlled by `netcdf_every`, `preview_every`, `checkpoint_every` |
| "standard mean variance normalization (mean based on entire grid in the first pass), set all the nans to zeros … same nan mask each time step … loss only over [ocean] pixels" | `src/preprocess.py::training_statistics`, `src/data.py::_normalize_target`, `src/losses.py` |
| "coarsen … block averaging, where over 50% of the domain is nan" | `src/coarsen.py::coarse_ocean_mask` (`min_valid_fraction = 0.5`) |
| "create a new environment that is easily reproducible with pixi" | `pixi.toml` + `pixi.lock`, two environments (`default` = CPU, `gpu` = CUDA 12) |
| "average of high resolution should be low resolution" | `consistency.project_to_coarse` exactly restores valid ocean-block means after AR sampling without altering the velocity loss |

> **Terminology note.** The brief says "only compute the loss in flow matching over land pixels". Land in this dataset is *permanently missing* (NaN), so the only interpretable reading — and the one consistent with the earlier sentence "have a loss function that masks out the land values" — is **compute the loss over ocean pixels and mask out land**. That is what is implemented.

---

## 1. The data

### 1.1 Source file

`/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling/sst_10km_OFAM_historical_Australia.nc`

Verified properties (measured, not assumed):

| Property | Value |
|---|---|
| Format | `NETCDF3_CLASSIC`, **uncompressed**, no chunking |
| Size on disk | 6.9 GB |
| Variable | `temp(Time, st_ocean, yt_ocean, xt_ocean)`, `int16` packed |
| Packing | `scale_factor = 0.001678518`, `add_offset = 45.0`, `_FillValue = -32768` |
| Units | degrees C (`sea_water_potential_temperature`) |
| Depth levels | 1 (`st_ocean = 2.5` m) — squeezed away |
| Grid | 512 × 512, uniform 0.1° |
| Latitude | −52.95 … −1.85 |
| Longitude | 107.35 … 158.45 |
| Time | 13 149 daily steps, 1979-01-01 → 2014-12-31, **no gaps** (all Δt = 24 h) |
| Land (NaN) fraction | **0.3140**, *identical in all 40 probed time steps* |
| Ocean statistics | min 0.99, max 36.73, mean 19.85, std 8.69 °C |
| Random read speed | ≈ 9 ms per day (524 288 B per slice, no decompression) |

**Consequences of these measurements**

* The land mask is genuinely static → compute once, reuse everywhere. `preprocess.py` re-verifies this on 24 evenly spaced days and raises if a single cell differs.
* Random access is cheap → the high-resolution target is **never copied to a derived file**. Training reads the 6.9 GB source directly. Only the tiny coarse predictor (54 MB) is materialised.
* SST is extremely persistent: day-to-day difference std is **0.092 °C** against a field std of **8.71 °C**; lag-1 spatial correlation is **0.99993**. This dominates the autoregressive design (see §4.2) — a naive AR model will learn "copy yesterday" and ignore the predictor.

### 1.2 Coarsening decision

512 is not divisible by 20, so an exact 2.0° / 20× coarsening requires either cropping to 480 (losing a 3.2° strip) or ragged blocks. **Decision taken: keep all 512 × 512 and coarsen by a factor of 16 → 32 × 32 at 1.6°.** This keeps every pixel, gives power-of-two grids that a 4-level U-Net divides cleanly, and is a 16× linear (256× areal) super-resolution problem.

`coarsen_factor` is a config field; setting it to 8 (0.8°) or 32 (3.2°) requires no code change. The alternatives that were measured:

| Option | Coarse grid | Coarse res. | Fully-land coarse cells | Cells failing the 50 % rule |
|---|---|---|---|---|
| crop 480, factor 20 | 24 × 24 | 2.0° | 145 / 576 | 195 / 576 |
| **keep 512, factor 16** | **32 × 32** | **1.6°** | **230 / 1024** | **308 / 1024** |

### 1.3 Splits

Chronological, no shuffling across the boundary:

| Split | Range | Days |
|---|---|---|
| train | 1979-01-01 → 2008-12-31 | 10 958 |
| validation | 2009-01-01 → 2010-12-31 | 730 |
| test | 2011-01-01 → 2014-12-31 | 1 461 |
| smoke | 1979-01-01 → 1979-01-31 | 31 |

The autoregressive dataset additionally refuses to build a pair that straddles a range boundary (`common.consecutive_pair_starts`), so no validation day can leak into training through the lag channel.

---

## 2. NaN strategy — the single most important design decision

Every NaN is eliminated *before* a tensor reaches the network, and every reduction is masked. Five layers of defence:

1. **Static mask, verified.** `preprocess.build_masks` reads 24 evenly spaced days and raises `ValueError` if the finite-value pattern changes at all.
2. **Mask-aware coarsening.** `coarsen.coarsen` never lets NaN enter arithmetic: land is zeroed, the block sum is divided by the *ocean count*, and a coarse cell whose ocean fraction is below `min_valid_fraction = 0.5` is declared invalid. This is exactly the "block averaging where over 50 % of the domain is NaN" rule.
3. **Fill after normalisation.** `data._normalize_target` / `_normalize_condition` standardise, then write `0.0` (= the mean of the normalised ocean distribution) into land and invalid coarse cells, then run a belt-and-braces `np.nan_to_num(..., posinf=0, neginf=0)`.
4. **Mask channels.** The network is *told* where the missing data is: the condition tensor carries the coarse validity mask as channel 1, and the high-resolution ocean mask is an explicit input channel (and an input to the lag encoder and the critic).
5. **Masked losses + guards.** `losses.masked_mean` divides by `mask.sum().clamp_min(1e-8)`. `flow.masked_noise` zeroes the noise over land so the ODE trajectory keeps land at exactly zero. `engine.check_finite` raises `FloatingPointError` on any non-finite loss; `engine.clip_and_step` refuses to apply an update when the gradient norm is not finite.

Additional stability measures: the last convolution of every residual block, the attention output projection, the FiLM projections, and the model output head are **zero-initialised** (identity at step 0); gradient clipping at 1.0; linear warmup then cosine decay; AdamW `betas = (0.9, 0.99)`; EMA of the weights used for all evaluation.

---

## 3. Normalisation

A **single global scalar mean and standard deviation** over every ocean cell of every training day — "mean based on entire grid in the first pass" — computed by a streaming pass in `preprocess.training_statistics`:

```
mean  = Σ x / N              N = ocean_cells × training_days ≈ 3.85e9
std   = sqrt(max(Σ x² / N − mean², 1e-8))
```

Expected values from the probe: mean ≈ 19.85 °C, std ≈ 8.69 °C.

Both the coarse predictor and the fine target use the **same** scalars, so the network sees a consistent scale and the identity mapping "upsample the predictor" is a sensible initialisation for the GAN residual generator.

`normalization.json` also records `sst_lr_mean`/`sst_lr_std` (diagnostic), the ocean-mask SHA-256, grid shapes, and the exact date ranges used. `data.DerivedProduct.verify` refuses to train if the fingerprint disagrees with the derived file.

---

## 4. Model designs

### 4.1 Shared backbone — `src/model.py::SuperResolutionFlowUNet`

* 4-level U-Net; channels `base × [1, 2, 4, 8]` with `base_channels = 32`.
* `ResidualBlock`: GroupNorm → 3×3 conv → FiLM from the time embedding → GroupNorm → 3×3 conv → channel attention → residual. Second conv zero-initialised.
* Bottleneck at 32 × 32 (512 / 16) with multi-head `SelfAttention` via `F.scaled_dot_product_attention` (1024 tokens — affordable).
* **Coarse conditioning is re-injected at every level.** The 32 × 32 condition is bilinearly resized to each stage's grid and concatenated. This is the mechanism that performs the 16× super-resolution; there is no transposed convolution or pixel-shuffle stack to tune.
* Inputs: flow state (1) + condition (2) + high-resolution ocean mask (1).

Adapted from: `NZ_domain_CPM/ml_downscaling/src/models.py` (residual block, time embedding, FiLM context) and `Autoregressive_Model/src/model.py` (channel attention, self attention, group count helper).

### 4.2 Autoregressive variant — coarse-authoritative revision

The first production attempt was stopped at about 86,000 steps because its free-running sequence was too persistent and validation remained worse than copying yesterday. The replacement preserves the `LagEncoder` + `GatedFiLM` structure but makes the hierarchy explicit:

* The current coarse SST and mask still enter every U-Net resolution.
* Before lag encoding, the ocean-only mean is removed independently from every coarse block of the previous field. The lag path therefore carries fronts and texture, not yesterday's block-scale temperature.
* `Dropout2d(lag_dropout = 0.10)` remains and whole-path dropout is raised to `0.50`.
* Gated FiLM is hard-capped by `lag_guidance_scale = 0.25`; learned gates cannot make the lag path authoritative.
* At the end of every one-day sample, `project_to_coarse` adds a constant within each valid ocean block so its mean exactly equals the current coarse input. This leaves every within-block anomaly unchanged and is outside the rectified-flow velocity loss.

Given the measured lag-1 correlation of 0.99993, **the headline diagnostic for this experiment is `skill_vs_persistence`** (logged every validation): `1 − MSE(model) / MSE(persistence)`. A value ≤ 0 means the model is not beating "yesterday's field" and the lag pathway needs further throttling.

### 4.3 Flow matching — `src/flow.py`

Standard rectified flow, adapted from `Autoregressive_Model/src/flow.py` and `NZ_domain_CPM/ml_downscaling/src/flow.py`:

```
t ~ U(0,1)
z(t) = (1−t)·noise + t·target        (noise is masked → land stays 0)
u    = target − noise
loss = masked_mse(v_θ(z, cond, mask, t[, y_prev]), u)
```

Samplers: `euler`, `heun` (2 evaluations/step, default), `ab2` (2nd order, 1 evaluation/step). All re-apply the mask after every update so land can never drift. `heun_sample` is deliberately **not** wrapped in `no_grad` so it doubles as the differentiable rollout solver.

The differentiable single-step rollout penalty has been retired. AR training now uses the same masked velocity objective as ordinary flow matching. `flow.rollout` remains only as a multi-day diagnostic and projects every generated day onto that day's coarse input before chaining it as lag guidance.

### 4.4 GAN — `src/model_gan.py`, `src/train_gan.py`

* **Generator** reuses `SuperResolutionFlowUNet` with the flow time pinned to 1, takes latent noise as its "state" input, and predicts a **residual around the bilinearly upsampled coarse field** (`generator_residual = true`). The output head is zero-initialised, so the generator *starts as exact bilinear interpolation* — a strong, NaN-free initialisation.
* **Critic**: conditional PatchGAN, 4 strided spectral-norm convolutions, sees `(field, upsampled condition, ocean mask)`. Its logit map is multiplied by a pooled ocean mask so land patches score nothing.
* **Losses**: hinge critic loss; generator loss `= λ_content · masked_mse(one sample, truth) + λ_adv · (−mean logit)` with `λ_content = 10`, `λ_adv = 1`. **Single-sample MSE, not ensemble-mean MSE**, as requested. Adversarial term switches on at `adversarial_start_step = 2000` so the generator first learns the deterministic mapping.

No GAN existed in any of the three reference repositories (checked with a repo-wide search for `discriminator`/`adversarial`/`wgan`) — this component is new, written directly in PyTorch.

---

## 5. Repository layout and provenance

```
SSTDownscaling/
├── plan.md                  ← this file
├── README.md                ← quick start
├── AGENTS.md                ← maintenance conventions
├── pixi.toml / pixi.lock    ← reproducible environments (default = CPU, gpu = CUDA 12)
├── sst_10km_OFAM_historical_Australia.nc   ← raw input (untouched)
├── derived/                 ← preprocessing output (gitignored)
│   └── sst_downscaling_f16.nc
├── reports/                 ← normalization_*.json (gitignored)
├── runs/                    ← checkpoints, netcdf/, predictions/, metrics/ (gitignored)
├── logs/                    ← PBS logs (gitignored)
├── configs/
│   ├── flow_sr.json         ├── flow_ar.json        └── gan_sr.json
├── src/
│   ├── common.py            ├── coarsen.py          ├── preprocess.py
│   ├── data.py              ├── losses.py           ├── model.py
│   ├── model_gan.py         ├── flow.py             ├── callbacks.py
│   ├── engine.py            ├── train_flow.py       ├── train_flow_ar.py
│   ├── train_gan.py         ├── evaluate.py         └── validate_data.py
├── tests/                   ← pytest suite (see Part C)
└── jobs/                    ← PBS scripts for h200q / shortq
```

### Provenance table — what was adapted from where

| New file | Adapted from | What was taken | What changed |
|---|---|---|---|
| `src/common.py` | `Autoregressive_Model/src/common.py` | `atomic_json`, `atomic_torch_save`, `date_keys`, `selected_indices`, `contiguous_runs`, `mask_sha256`, `seed_everything` | Single-variable SST instead of 3 targets; no `log1p` transform; added RNG save/restore helpers and `consecutive_pair_starts` |
| `src/coarsen.py` | *new* | — | Mask-aware block averaging with the 50 % validity rule |
| `src/preprocess.py` | `Autoregressive_Model/src/prepare_global_sst.py` (streaming stats pattern) | Streaming mean/std, atomic NetCDF write, mask fingerprinting | Reads NETCDF3 int16 directly; produces only the coarse predictor + masks |
| `src/data.py` | `Autoregressive_Model/src/data.py` | Split-safe window construction, preload/lazy duality, worker-safe handles | `netCDF4` instead of `h5py`; single channel; mask channels; NaN→0 after normalisation |
| `src/losses.py` | `Autoregressive_Model/src/losses.py` (spectral loss) | `spectral_amplitude_loss` shape | **All losses are masked**; added hinge GAN losses and the (inert) conservation loss |
| `src/model.py` | `NZ_domain_CPM/.../models.py` + `Autoregressive_Model/src/model.py` | `ResidualBlock`+FiLM, `TimeEmbedding`, `ChannelAttention`, `SelfAttention`, `LagEncoder`, `GatedFiLM`, `group_count` | Multi-level coarse re-injection for super-resolution; zero-init output paths; `scaled_dot_product_attention` |
| `src/flow.py` | `Autoregressive_Model/src/flow.py`, `NZ_domain_CPM/.../flow.py` | Rectified-flow loss, Heun / AB samplers, rollout structure | Masked noise/state/velocity; optional coarse projection at generated-field boundaries |
| `src/consistency.py` | *new* | — | Differentiable block means, lag high-pass decomposition, and exact coarse-authority projection |
| `derived/convert_access_to_training_grid.py` | supplied workflow, repaired as standalone | Seasonal-anomaly conversion onto the established training predictor grid with atomic output |
| `src/infer_access_cm2.py` | *new* | — | Validation and resumable inference from the already-converted ACCESS-CM2 predictor; no remapping |
| `src/model_gan.py` | *new* | — | PatchGAN + spectral norm, masked logits, residual generator |
| `src/callbacks.py` | `Autoregressive_Model/src/callbacks.py`, `NZ_domain_CPM/.../callbacks.py` | Preview grid, atomic NetCDF write, metrics JSON | 5-panel layout incl. coarse input and radial power spectrum; rollout-skill plot |
| `src/engine.py` | `Autoregressive_Model/src/train.py`, `NZ_domain_CPM/.../train.py` | EMA, resumable checkpoints, wall-clock guard, status.json handshake | Factored into a shared engine used by all three trainers |
| `jobs/*.pbs` | `Autoregressive_Model/jobs/train_single_step.pbs` | `status.json` → auto-resubmit chaining, `h200q` resource block | Points at the pixi `gpu` environment |

Reference repositories were **read only** — nothing in `Autoregressive_Model`, `NZ_domain_CPM`, or `Perfect_vs_Imperfect` was modified.

---

## 6. Configuration reference

Shared keys (identical meaning in all three configs):

| Key | Default | Meaning |
|---|---|---|
| `source_path` | raw NetCDF | Untouched high-resolution input |
| `derived_path` | `derived/sst_downscaling_f16.nc` | Coarse predictor + masks |
| `normalization_cache` | `reports/normalization_f16.json` | Global mean/std + fingerprints |
| `coarsen_factor` | `16` | Linear super-resolution factor |
| `min_valid_fraction` | `0.5` | Coarse-cell validity threshold |
| `base_channels` / `levels` | `32` / `4` | Backbone width and depth |
| `batch_size` | `8` (flow), `4` (AR), `8` (GAN) | Per-step batch |
| `learning_rate` | `1e-4` | AdamW |
| `ema_decay` | `0.999` | Evaluation weights |
| `warmup_steps` / `min_learning_rate_factor` | `500` / `0.05` | Schedule |
| `gradient_clip` | `1.0` | Global norm clip |
| `max_steps` / `max_runtime_hours` | `120000` / `23.0` | Budget and PBS guard |
| `log_every` / `validation_every` | `50` / `2000` | Logging cadence |
| `preview_every` / `netcdf_every` / `checkpoint_every` | `1000` / `5000` / `2000` | **Illustrations, NetCDF, weights** |
| `sampler` / `preview_sampler_steps` | `heun` / `25` | Inference solver |
| `lambda_conservation` | `0.0` | Low→high constraint, **off** for round one |

Autoregressive-only: `horizon` (1), `lag_base_channels` (16), `lag_dropout` (0.10), `lag_path_dropout` (0.50), `lag_guidance_scale` (0.25), `enforce_coarse_consistency` (true), `rollout_days` (10), and `rollout_netcdf_every` (10000). Rollouts are diagnostics, not a training loss.

GAN-only: `noise_channels` (4), `generator_residual` (true), `discriminator_base_channels` (32), `discriminator_levels` (4), `discriminator_learning_rate` (1e-4), `critic_steps` (1), `lambda_content` (10.0), `lambda_adversarial` (1.0), `adversarial_start_step` (2000).

---

## PART A — What has been done

- [x] **Data forensics.** Format, packing, grid, calendar, gaps, land fraction, mask stationarity (40 days), ocean statistics, day-to-day persistence, per-pixel vs global spread, random-read throughput, and the coarsening-divisibility trade-off all measured directly from the 6.9 GB file.
- [x] **Reference-repository survey.** `Autoregressive_Model`, `NZ_domain_CPM`, `Perfect_vs_Imperfect` mapped; flow losses, samplers, U-Net blocks, lag encoder, dataset patterns, callbacks, checkpointing, and PBS conventions extracted. Confirmed **no existing GAN** to port.
- [x] **`pixi.toml`** with two environments — `default` (CPU, installs on a login node) and `gpu` (`platforms = [{platform = "linux-64", cuda = "12"}]`). Both resolved and installed; `pixi.lock` written.
- [x] `src/common.py` — config loading, atomic I/O, date selection, split-safe pair construction, mask hashing, RNG save/restore, normalisation helpers.
- [x] `src/coarsen.py` — mask-aware block averaging, coarse-mask construction with the 50 % rule, coordinate coarsening, nearest-neighbour upsample for figures.
- [x] `src/preprocess.py` — mask stationarity check, full-record coarsening, streaming global statistics on the training split only, atomic derived-NetCDF write, `normalization.json`.
- [x] `src/data.py` — `DerivedProduct`, worker-safe `_SourceReader`, `SuperResolutionDataset`, `AutoregressiveSuperResolutionDataset`, `build_dataset` factory; NaN→0 after normalisation; fingerprint verification.
- [x] `src/losses.py` — masked mean/MSE/L1/RMSE/bias, masked spectral loss, inert conservation loss, hinge GAN losses.
- [x] `src/model.py` — shared backbone, autoregressive variant with throttled lag pathway, `build_model` factory.
- [x] `src/flow.py` — masked rectified-flow loss, Euler/Heun/AB2 samplers, free-running diagnostic rollout, and optional coarse projection.
- [x] `src/callbacks.py` — 5-panel preview (coarse / truth / generated / error / radial spectrum), loss curves, NetCDF writers for samples and rollouts, rollout-skill plot, metrics JSON.
- [x] `src/engine.py` — EMA, warmup+cosine schedule, resumable checkpoints with RNG state, non-finite guards, wall-clock guard, `status.json`, deterministic preview batching.
- [x] `src/train_flow.py` — experiment 1 end to end.
- [x] `src/train_flow_ar.py` — experiment 2 end to end, including `skill_vs_persistence` and free-running diagnostic products.

## PART B — Remaining tasks

- [x] **B1** `src/model_gan.py` — generator (backbone + zero-init residual head) and masked PatchGAN critic.
- [x] **B2** `src/train_gan.py` — alternating critic/generator loop, single-sample masked MSE content loss, delayed adversarial start, shared callbacks.
- [x] **B3** `src/evaluate.py` — offline test-split evaluation for any run directory: RMSE/MAE/bias/spectra, sampler-step ablation, multi-day NetCDF.
- [x] **B4** `src/validate_data.py` — standalone preflight: assert no NaN reaches a batch, masks agree, statistics are sane, splits are disjoint and gapless.
- [x] **B5** `configs/flow_sr.json`, `configs/flow_ar.json`, `configs/gan_sr.json`.
- [x] **B6** `tests/` — the full matrix in Part C (including conditioning-ablation, coarse-authority, converted-ACCESS, solver, and GAN critic-freeze regressions).
- [x] **B7** `jobs/*.pbs` — `preprocess.pbs` (shortq), `cpu_tests.pbs` (shortq), `gpu_smoke.pbs` (h200q), `train_flow.pbs` / `train_flow_ar.pbs` / `train_gan.pbs` (h200q, self-resubmitting on `status == "checkpointed"`).
- [x] **B8** Real preprocessing complete: mean 19.833256 °C, std 8.699326 °C, 32 × 32 coarse grid, 716 valid coarse cells.
- [x] **B9** Full CPU suite green: 131 passed, 1 restricted-sandbox skip (2026-08-27); the two-worker DataLoader check remains verified on a PBS node.
- [x] **B10** Real-grid CPU smoke training complete for all three models (3 steps each); every run produced PNG, NetCDF, checkpoint, weights, history, and passing status. AR rollout and GAN adversarial paths were exercised.
- [x] **B11** GPU smoke passed on an NVIDIA H200 in job `6406137`: production-batch forward/backward for all models, 11.93 GiB maximum peak allocation, 25-step Heun in 1.91 s, 10-day AR rollout in 19.34 s, and a verified NetCDF product.
- [x] **B12** `README.md`, `AGENTS.md`, `.gitignore`, Git repository, and initial commit `8c0bb9c` created.
- [x] **B13** 1000-step full-grid GPU flow smoke passed in job `6406393`: 997 resumed steps in 80.8 s (~12.3 steps/s), preview + NetCDF written, and the raw model achieved 0.70 °C RMSE with 99.3 % skill versus same-noise coarse-SST ablation. The initially noisy preview was traced to expected fixed-decay EMA lag, not a broken conditioning path.
- [ ] **B14** Complete production runs. Plain flow completed 120,000 steps. The original AR job `6406391` and GAN job `6406398` were canceled on 2026-08-27 because their approaches were rejected; their outputs are retired. No replacement production job is active.
- [x] **B15** GAN adversarial recovery verified before the experiment was canceled. The step-4000 audit found generator-loss gradients contaminating critic updates; the critic was frozen during generator updates, a regression test was added, and a fresh GAN-v2 CPU smoke passed.
- [x] **B16** Coarse-guided AR replacement: focused CPU tests, full CPU suite, three CPU smokes, real-data validation, and the new H200 smoke all pass; the old job is canceled and no AR production job is active.
- [x] **B17** Full-test `flow_sr` inference completed by job `6406503`: 1,461 consecutive test days, 50-step Heun, atomic NetCDF and metrics integrity checks passed.
- [x] **B18** Retired the inference-time ACCESS remapper after receiving `derived/sst_downscaling_access_converted.nc`. The supplied conversion method is now a standalone tracked CLI at `derived/convert_access_to_training_grid.py`; inference only validates and consumes its 32×32 output.
- [x] **B19** The converted ACCESS product was audited: `sst_lr(time=51135, lat_lr=32, lon_lr=32)`, 1960-01-01 through 2099-12-31, with exactly 716 finite training-ocean cells in each probed field.
- [ ] **B20** Run and verify converted-ACCESS inference for the exact ten-year historical (1980–1989) and future (2080–2089) periods using the original `flow_sr` EMA, AB3/AM3 predictor-corrector, and 75 steps.
- [x] **B21** Verified the plain-flow test split independently from the stored derived time axis: train (1979–2008; 10,958 days), validation (2009–2010; 730 days), and test (2011–2014; 1,461 days) have zero overlap. The normalization provenance contains the training range only.
- [ ] **B22** Plain-flow continuation job `6406705` is running from step 120,000 to 220,000 in the separate `runs/flow_sr_continue_220k` run after its H200 fork smoke passed. It preserves model/EMA, AdamW state, RNG and the active 5e-6 LR; immutable 10,000-step weight snapshots begin at step 130,000.
- [x] **B23** Job `6406703` compared 100-step Heun and AB2 on the same first 30 test days and identical initial noise. Both had zero non-finite ocean pixels; AB2 was 2.04x faster and had 0.3721 °C RMSE versus Heun's 0.3759 °C. Metrics, atomic NetCDF, and the solver-difference figure passed integrity checks.
- [x] **B24** Added and validated an AB3/AM3 predictor-corrector sampler whose two-step velocity history is confined to one ODE sample. The 100-step, same-noise 30-day comparison passed: AB3-PC RMSE was 0.3749 °C versus 0.3759 °C for Heun and 0.3721 °C for AB2, with zero non-finite ocean pixels.
- [ ] **B25** Job `6406740` is running all 1,461 test days with the plain-flow EMA weights using only the AB3/AM3 predictor-corrector at 75 steps. Earlier full-test and solver-comparison outputs are preserved under solver-specific filenames.

## PART C — Exhaustive test matrix

All tests live in `tests/` and run with `pixi run test` (CPU, no GPU required). Tests that need the real 6.9 GB file are marked `@pytest.mark.slow` and are skipped automatically when it is absent, so the suite is portable.

### C1 `tests/test_coarsen.py` — coarsening correctness
| Test | Asserts |
|---|---|
| `test_all_ocean_matches_plain_mean` | With a full-ocean mask, `coarsen` equals `reshape().mean()` exactly |
| `test_nan_does_not_propagate` | A single NaN inside a block does not NaN the coarse cell |
| `test_partial_block_uses_ocean_only` | Coarse value equals the mean of the ocean cells only, hand-computed |
| `test_fifty_percent_rule_boundary` | Exactly 50 % ocean → **valid**; one cell fewer → invalid |
| `test_all_land_block_is_invalid` | Fully-land block is invalid and filled, never `0/0` |
| `test_non_divisible_grid_raises` | `ValueError` for a grid not divisible by the factor |
| `test_batch_and_single_agree` | 3-D and 2-D code paths give identical results |
| `test_coarse_coordinates` | Coordinate block means are correct and length-consistent |
| `test_upsample_nearest_roundtrip` | Upsampled shape equals the fine grid |
| `test_output_is_finite_where_valid` | No NaN/Inf anywhere the coarse mask is true |

### C2 `tests/test_common.py` — plumbing
| Test | Asserts |
|---|---|
| `test_selected_indices_ranges` | Inclusive bounds, multiple ranges, sorted output |
| `test_selected_indices_empty_raises` | Empty selection raises |
| `test_contiguous_runs` | Runs and destination offsets reconstruct the index array |
| `test_consecutive_pair_starts_excludes_boundary` | The last day of a range never starts a pair |
| `test_consecutive_pair_starts_detects_gap` | A missing day raises |
| `test_mask_sha256_sensitivity` | Digest changes when the mask *or* the grid changes |
| `test_atomic_json_and_torch_save` | No partial file remains; content round-trips |
| `test_rng_state_roundtrip` | Save/restore reproduces the identical random stream |
| `test_normalized_to_physical_restores_nan` | Land comes back as NaN, ocean round-trips to within 1e-5 |

### C3 `tests/test_data.py` — datasets (synthetic fixture)
A pytest fixture writes a tiny NETCDF3 file (64 × 64, 40 days, int16-packed, a deterministic land blob) and runs the real `preprocess.run` on it, so the dataset tests exercise the production code path.
| Test | Asserts |
|---|---|
| `test_preprocess_creates_products` | Derived NetCDF and `normalization.json` exist with the expected keys |
| `test_statistics_match_numpy` | Streaming mean/std equal a direct `np.nanmean`/`np.nanstd` to 1e-6 |
| `test_statistics_use_training_range_only` | Changing the validation range does not change the statistics |
| `test_no_nan_in_any_batch` | **Every** tensor of 40 sampled items is finite |
| `test_land_is_exactly_zero` | Target and condition are exactly 0 where the mask is 0 |
| `test_mask_channel_matches_mask` | Condition channel 1 equals the coarse mask |
| `test_mask_identical_across_items` | The mask tensor is the same object/value for every item |
| `test_shapes` | Target `(1, H, W)`, condition `(2, H/f, W/f)`, mask `(1, H, W)` |
| `test_normalized_statistics` | Ocean mean ≈ 0, std ≈ 1 within tolerance |
| `test_ar_pairs_are_consecutive` | `date_window` differs by exactly one day |
| `test_ar_pairs_do_not_cross_split` | No pair start is the last day of a range |
| `test_ar_previous_matches_previous_item_target` | `previous` of pair *t* equals `target` of pair *t−1* |
| `test_preload_matches_lazy` | Preloaded and lazy datasets return identical tensors |
| `test_fingerprint_mismatch_raises` | A tampered `normalization.json` is rejected |
| `test_dataloader_multiple_workers` | `num_workers = 2` returns finite, correct-shaped batches |

### C4 `tests/test_losses.py` — masking
| Test | Asserts |
|---|---|
| `test_masked_mse_ignores_land` | Arbitrary garbage in land pixels does not change the loss |
| `test_masked_mse_matches_manual` | Equals the hand-computed ocean-only mean |
| `test_masked_mse_full_mask_equals_mse` | With an all-ones mask, equals `F.mse_loss` |
| `test_empty_mask_is_finite` | An all-zero mask returns 0, not NaN |
| `test_masked_bias_and_rmse` | Sign and magnitude correct on a known offset |
| `test_gradients_are_zero_over_land` | `d loss / d prediction` is exactly 0 at land pixels |
| `test_spectral_loss_zero_for_identical` | Identical fields give ≈ 0 |
| `test_conservation_loss_zero_for_consistent_pair` | A field whose block mean equals the predictor scores ≈ 0 |
| `test_hinge_losses_signs` | Critic and generator hinge losses have the expected monotonicity |

### C5 `tests/test_model.py` — architectures
| Test | Asserts |
|---|---|
| `test_sr_forward_shape` | `(B,1,64,64)` in → `(B,1,64,64)` out with a `(B,2,4,4)` condition |
| `test_ar_forward_shape` | Same with the previous state supplied |
| `test_zero_init_output_at_step_zero` | A freshly built model outputs exactly 0 |
| `test_forward_is_finite_with_extreme_inputs` | ±10 σ inputs stay finite |
| `test_backward_produces_finite_grads` | All parameter grads finite; no unused-parameter surprises |
| `test_condition_actually_matters` | Changing the condition changes the output after one optimiser step |
| `test_previous_state_matters` | Same for the lag input |
| `test_lag_path_dropout_is_active_in_train_only` | Output varies in `train()`, is deterministic in `eval()` |
| `test_wrong_channel_count_raises` | Clear `ValueError` for bad condition/state channels |
| `test_wrong_mask_grid_raises` | Clear `ValueError` when the mask grid disagrees |
| `test_multiple_grid_sizes` | 64, 128, and 512 grids all run |
| `test_build_model_factory` | Both `model_kind` values build the right class; unknown raises |
| `test_parameter_count_reasonable` | Production config is within an expected order of magnitude |

### C6 `tests/test_flow.py` — objective and samplers
| Test | Asserts |
|---|---|
| `test_masked_noise_is_zero_on_land` | Exactly zero over land |
| `test_loss_is_finite_and_positive` | Both model kinds |
| `test_loss_gradient_flows` | Non-zero gradient reaches the first conv |
| `test_samplers_preserve_land_zero` | Euler / Heun / AB2 all keep land exactly 0 |
| `test_samplers_shape_and_finiteness` | Correct shape, all finite, for 1/2/5/25 steps |
| `test_sampler_determinism_with_seed` | Same generator seed → bitwise identical output |
| `test_heun_matches_analytic_linear_field` | For a constant-velocity dummy model, Heun integrates exactly |
| `test_zero_steps_raises` | `ValueError` |
| `test_unknown_sampler_raises` | `ValueError` listing the valid names |
| `test_rollout_shape_and_chaining` | `(B, L, 1, H, W)`; lead *k* depends on lead *k−1* |
| `test_single_step_rollout_loss_backward` | Gradient flows through the unrolled solver; loss finite |

### C7 `tests/test_gan.py` — adversarial components
| Test | Asserts |
|---|---|
| `test_generator_output_shape_and_mask` | Correct shape; exactly 0 over land |
| `test_generator_starts_as_bilinear_baseline` | At init, output ≈ upsampled condition over ocean |
| `test_generator_noise_changes_output` | Two noise draws differ |
| `test_discriminator_output_shape` | Patch logits with the expected downsampling |
| `test_discriminator_masks_land` | Editing land pixels does not change the logits |
| `test_one_gan_step_runs` | Critic and generator steps both produce finite grads |
| `test_content_loss_is_single_sample` | The content term equals `masked_mse` of one sample (regression guard against an ensemble mean creeping in) |

### C8 `tests/test_callbacks.py` — products
| Test | Asserts |
|---|---|
| `test_to_physical_roundtrip` | Denormalisation inverts normalisation; land is NaN |
| `test_field_metrics_ignore_nan` | Metrics match a manual `nanmean` computation |
| `test_save_preview_writes_png` | File exists and is non-trivial in size |
| `test_save_loss_curve_writes_png` | Handles 1, 10, and 5000 records |
| `test_save_netcdf_roundtrip` | Re-opened file has the right variables, coords, dtypes, units, and NaN land |
| `test_save_rollout_netcdf_roundtrip` | Lead axis and time coordinate correct |
| `test_radial_spectrum_monotone_for_smooth_field` | A smoothed field has less high-wavenumber power |
| `test_atomic_write_leaves_no_partial` | No `.partial.nc` remains |

### C9 `tests/test_engine.py` — training machinery
| Test | Asserts |
|---|---|
| `test_ema_tracks_parameters` | After *n* updates the EMA equals the analytic value |
| `test_scheduler_warmup_and_decay` | LR rises linearly then decays to the floor |
| `test_check_finite_raises_on_nan` | `FloatingPointError` with a useful message |
| `test_clip_and_step_rejects_nan_grad` | Raises instead of poisoning the weights |
| `test_checkpoint_roundtrip` | Save → new model → load reproduces identical outputs |
| `test_resume_restores_step_and_rng` | Step, history, and RNG stream all restored |
| `test_should_run_cadence` | `every = 0` never fires; `every = n` fires at multiples |
| `test_fixed_indices_deterministic` | Same indices across calls and processes |

### C10 `tests/test_train_smoke.py` — end-to-end (synthetic data)
| Test | Asserts |
|---|---|
| `test_flow_smoke_end_to_end` | 3 steps; `status.json` written; checkpoint, preview PNG, and NetCDF exist; loss finite |
| `test_flow_ar_smoke_end_to_end` | Same, plus a rollout NetCDF and a rollout-skill PNG |
| `test_gan_smoke_end_to_end` | Same, plus `adversarial_exercised == true` |
| `test_resume_continues` | Re-running with a higher `--smoke-steps` continues from the checkpoint rather than restarting |
| `test_no_nan_in_history` | Every logged loss value is finite |

### C11 Real-data checks — `src/validate_data.py` (`pixi run validate-data`)
1. Source file opens; grid, calendar, and packing match the recorded metadata.
2. Land mask is stationary across 100 evenly spaced days.
3. Derived file's mask fingerprint matches `normalization.json`.
4. Train / validation / test index sets are disjoint and each is gapless.
5. 256 randomly drawn training items contain **no** non-finite value in any tensor.
6. Normalised ocean mean is within 0.05 of 0 and std within 0.05 of 1.
7. Coarse predictor is finite everywhere inside the coarse ocean mask.
8. Reconstructed physical values round-trip to within 1e-4 °C of the source.
9. Reports the persistence baseline MSE so `skill_vs_persistence` has a reference.

### C12 GPU acceptance on `h200q` — `jobs/gpu_smoke.pbs`
1. `torch.cuda.is_available()` and the device name are logged.
2. One forward+backward at the production batch size for all three models; peak memory reported via `torch.cuda.max_memory_allocated()`.
3. 25-step Heun sample at 512 × 512 timed; NetCDF product written and re-opened.
4. Free-running 10-day rollout for the autoregressive model; peak memory reported.
5. Job fails loudly (non-zero exit) on any non-finite loss.

### C13 Acceptance criteria before launching production runs
- [x] Whole `pixi run test` suite green on CPU (120 passed, 1 restricted-sandbox skip, 2026-08-27).
- [x] `pixi run validate-data` green on the real file (all nine checks passed).
- [x] All three CPU smoke trainings produce PNG + NetCDF + checkpoint.
- [x] `jobs/gpu_smoke.pbs` completes on h200q within its walltime and fits in memory (job `6406137`, peak 11.93 GiB).
- [ ] Flow-matching validation RMSE beats bilinear upsampling of the coarse field.
- [ ] Autoregressive `skill_vs_persistence > 0` (otherwise re-tune `lag_path_dropout`).
- [ ] GAN content loss decreases and the critic logits stay bounded (no divergence).

---

## 7. Operational runbook

```bash
cd /esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling

pixi install                 # default (CPU) environment
pixi run test                # full unit-test suite
qsub jobs/preprocess.pbs     # ~15 min on shortq: derived NetCDF + statistics
pixi run validate-data       # preflight on the real file
qsub jobs/gpu_smoke.pbs      # h200q acceptance test
qsub jobs/train_flow.pbs     # experiment 1  (self-resubmits until "passed")
qsub jobs/train_flow_ar.pbs  # experiment 2
qsub jobs/train_gan.pbs      # experiment 3
pixi run evaluate-flow       # test-split metrics and NetCDF products
```

Each training job inspects `runs/<name>/status.json` on exit and re-submits itself while the status is `"checkpointed"`, exactly as `Autoregressive_Model/jobs/train_single_step.pbs` does.

## 8. Known risks

| Risk | Mitigation |
|---|---|
| AR model degenerates to persistence | Path dropout + gated FiLM starting near zero; `skill_vs_persistence` tracked every validation |
| 16× super-resolution is under-determined | Flow matching and the GAN are both generative; the spectral panel in every preview shows whether fine scales are actually produced |
| 512 × 512 activations exhaust GPU memory | `base_channels`, `batch_size`, and `levels` are config fields; `jobs/gpu_smoke.pbs` reports peak memory before any long run |
| AR degenerates to persistence | Remove lag block means, cap lag FiLM at 0.25, use 50% path dropout, enforce current coarse means, and report an evolution ratio in diagnostic sequences |
| GAN divergence | Spectral norm, hinge loss, delayed adversarial start, `λ_content = 10` dominating early training |
| NETCDF3 handles in DataLoader workers | Handles are opened lazily per process and excluded from pickling (`_SourceReader.__getstate__`) |
