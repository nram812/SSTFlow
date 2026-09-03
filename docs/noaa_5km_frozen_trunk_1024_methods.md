# NOAA 0.05° frozen-trunk 1024² transfer

## Why the first transfer was retired

The first experiment in `runs/flow_sr_noaa_5km_transfer` generated an OFAM-grid
512² field, bilinearly enlarged that *field*, and learned only a residual whose
NOAA-ocean mean was forced to zero in every 2×2 block.  At step 40,000 its
2019-01-01 validation sample had:

| Region | Bias (°C) | RMSE (°C) |
|---|---:|---:|
| All NOAA ocean | -0.17 | 1.37 |
| First NOAA ocean pixel beside land | -6.09 | 7.72 |
| Within 4 NOAA pixels of land | -3.04 | 5.48 |
| More than 8 pixels from land | -0.01 | 0.53 |

The grids are geometrically aligned: adjacent NOAA 0.05° centres average to
each OFAM 0.1° centre to numerical tolerance.  The problem is the *mask
contract*.  The target contains 15,060 NOAA-ocean pixels over OFAM-land cells
and 322 OFAM-ocean pixels that are NOAA land.  The hard partial-block correction
therefore imposed a low-resolution land/fill value on real NOAA coastal ocean.
It also prevented the second model from changing any 2×2 block mean, so the
result necessarily retained the appearance of the 0.1° sample.

Production job `6408777` was stopped after preserving its step-40,000
checkpoints and diagnostics.  Those files are retained as a failed ablation.

## Replacement architecture

The replacement predicts the complete 1024² rectified-flow velocity directly:

```text
32² SST + mask ───────────────┐
                             │
1024² flow state ── masked 2×2 mean ──> 512² state
                             │
                             v
                  frozen pretrained OFAM U-Net
                  (through final 512² decoder features)
                             │
        old output_norm/output are bypassed
                             │
                             v
                  learned Conv + PixelShuffle ×2 ──┐
                                                   ├─> residual blocks ─> 1024² velocity
1024² state + NOAA mask + resized SST condition ───┘
```

All 5,684,409 pretrained parameters are frozen and run without a backward
graph.  The new 149,201-parameter head is the only trainable component.  A
learned PixelShuffle layer produces four distinct sub-cell feature sets for
each 0.1° feature cell; bilinear interpolation is used only to present the
32² conditioning field at the head grid, not to construct a predicted SST.
The direct 1024² state branch ensures that high-wavenumber ODE state is not
discarded by the frozen 512² path.

The NOAA target mask is supplied explicitly to the new head and is the only
mask used for the loss and final velocity.  The fixed OFAM mask is used only
inside the frozen pretrained trunk.  No 2×2 conservation projection, resized
SST baseline, or OFAM land fill is added to the output.

## Objective

The training path remains the same straight conditional rectified flow used by
the pretrained model:

```text
z_t = (1 - t) ε + t y_NOAA
u   = y_NOAA - ε
L   = MSE_NOAA-ocean(vθ(z_t, SST, mask, t), u)
    + 0.5 MSE_NOAA-coast(vθ(z_t, SST, mask, t), u)
```

The coastal term covers NOAA-ocean pixels within four 0.05° cells of land.  It
does not alter targets or impose a value; it prevents the small coastal subset
from being overwhelmed by open-ocean pixels in the reduction.  Land is zeroed
throughout the flow path and excluded from both terms.

## Data and grid contract

- NOAA crop: `lat[10:1034], lon[16:1040]`, exactly 1024×1024.
- Frozen trunk grid: 512×512, exactly nested 2:1 in both dimensions.
- Predictor: daily mask-aware NOAA mean on the existing 32×32 model condition
  grid, normalized with the unchanged combined-climate OFAM statistics.
- Target mask: static NOAA finite-value mask, 734,062 ocean cells.
- Splits: train 1990–2018, validation 2019–2020, test 2021–2023.
- Missing source dates (2020-04-03 and 2023-06-11) remain absent and are never
  bridged by a sequence; this experiment is non-autoregressive in time.

## Files and callbacks

- Model: `src/model_noaa_5km_v2.py`
- Trainer: `src/train_flow_noaa_5km_v2.py`
- Configuration: `configs/flow_sr_noaa_5km_frozen_trunk_1024.json`
- H200 gate: `jobs/gpu_smoke_noaa_5km_v2.pbs`
- Production: `jobs/train_flow_noaa_5km_v2.pbs`
- Run: `runs/flow_sr_noaa_5km_frozen_trunk_1024`

Previews every 2,000 updates show the condition, NOAA truth, direct 1024²
sample, full error, four-pixel coastal error, and NOAA/OFAM mask disagreement.
NetCDF samples every 10,000 updates store the target, generated field,
diagnostic ocean-only 2× coarsening, and condition.  Validation records bias and
RMSE for coastal radii 1, 2, 4, and 8 pixels separately from the interior.

## Gates

The focused suite verifies masked downsampling, the NOAA-only coastline path,
that the pretrained output layers are truly unused, bitwise trunk freezing,
gradient flow into the learned upsampler, land-zero behavior, lack of an
implicit block projection, and finite Heun sampling.  Production is released
only after the full CPU suite, real-data validator, full-grid CPU smoke, and
H200 memory/update/sampler gate pass.

H200 gate `6408783` passed with 4.90 GiB peak allocation. Production job
`6408784` then reached its first durable checkpoint. On the fixed 2019-01-01
validation case, the step-4,000 EMA sample achieved 0.58 °C whole-ocean RMSE,
0.95 °C RMSE in the first ocean pixel beside land, and 0.77 °C RMSE within four
pixels of land. The corresponding failed-model values were 1.37, 7.72, and
5.48 °C. These early values verify removal of the coastline artifact; they are
not reported as final test skill.

## Decoder-unfrozen continuation from step 38,000

The frozen-trunk stage was stopped after its step-38,000 validation callback.
PBS delivered the stop signal after update 39,500, but the original save cadence
was 5,000 updates, so the most recent recoverable model/EMA/optimizer/RNG state
was step 35,000. The 38,000 PNG is a diagnostic and cannot be used as weights.
The reproducible bridge configuration
`configs/flow_sr_noaa_5km_head_bridge_38k.json` therefore replays 3,000
head-only updates from step 35,000 and writes a durable step-38,000 state. The
shuffle iterator restarts at the checkpoint boundary, so this reconstructed
state is not claimed to be bit-for-bit identical to the transient original
step-38,000 process.

The next stage forks weights and EMA from that step, but deliberately starts a
fresh AdamW optimizer because newly trainable parameters have no prior moments:

| Path | Parameters | Learning rate | Status |
|---|---:|---:|---|
| 1024 PixelShuffle/direct-state head | 149,201 | 5e-5 | trainable |
| 512 bottleneck + attention + all up blocks | 3,936,356 | 1e-5 | trainable |
| Time embedding + input/down encoder | 1,748,020 | — | frozen |
| Bypassed 512 output norm/convolution | 33 | — | frozen and unused |

Thus 4,085,557 of 5,833,610 parameters (70.0%) are trainable. “Entire decoder”
means `middle1`, bottleneck attention, `middle2`, and every `up` block. Frozen
encoder features and skips are computed without an autograd graph; decoder
operations retain the graph. Differential learning rates reduce catastrophic
forgetting while allowing the established high-resolution head to keep
adapting. Both groups use a fresh 500-update warmup and a 42,000-update cosine
schedule, ending at global step 80,000.

Continuation files:

- Config: `configs/flow_sr_noaa_5km_decoder_finetune_38k.json`
- H200 gate: `jobs/gpu_smoke_noaa_5km_decoder.pbs`
- Production: `jobs/train_flow_noaa_5km_decoder_38k.pbs`
- Run: `runs/flow_sr_noaa_5km_decoder_finetune_from_038000`

The stage fork records the source checkpoint SHA-256 and required source step,
refuses a checkpoint other than step 38,000, and never imports the head-only
optimizer. Focused tests assert exact optimizer coverage, decoder gradients and
updates, bitwise-frozen encoder tensors, bypassed legacy output layers, finite
masked loss, and finite Heun inference.

Operational record (2026-09-01): bridge job `6408788` exited 0 at step 38,000;
its checkpoint SHA-256 is
`ff6f288634a67dc4f5954f3b3508d4bfb60d2aa4d70c1d9b0995a21ff7241d2e`.
H200 gate `6408791` exited 0: decoder and head parameter deltas were 0.00158
and 0.01343, frozen encoder tensors were bitwise unchanged, two-step Heun was
finite, and peak allocated memory was 5.74 GiB. Production job `6408792` then
started from the checksum-verified weights-only fork.

## Low-rate continuation to step 150,000

The decoder-and-head stage completed successfully at step 80,000. Extending
the original 42,000-update cosine schedule in place would have raised both
learning rates abruptly after they had reached their floors. The 80k-to-150k
stage is therefore a checksum-locked weights/EMA fork with a fresh optimizer
and a separate 70,000-update cosine schedule:

| Path | Parameterized leaf modules | Parameters | Peak learning rate |
|---|---:|---:|---:|
| 1024 head + pretrained bottleneck/decoder | 64 | 4,085,557 | 1e-5 head / 2e-6 decoder |
| Time embedding + input/down encoder + bypassed output | 36 | 1,748,053 | frozen |

The 64/36 count refers to leaf modules that directly own parameters; no module
is partly frozen. The trainable fraction remains 70.03%. The source checkpoint
is exactly step 80,000 with SHA-256
`38cae2f6fa1451a58d533a9b80f2c2ea7f55fa40ba0d01b9ecca8b31bdd4bbd6`.
H200 gate `6408874` verified source identity, non-zero head and decoder updates,
a bitwise-frozen encoder, finite Heun sampling, and 5.74 GiB peak allocation.
Production continuation `6408875` writes to the isolated run
`runs/flow_sr_noaa_5km_decoder_continue_150k`.

Final inference uses a dedicated 1024-grid writer and validator. It evaluates
the full 2021-2023 NOAA test split and the converted ACCESS-CM2 1980-1989 and
2080-2089 periods with 75-step AB3/AM3 predictor-corrector sampling. An H200
profile found that changing attention batch shape changes floating-point ODE
trajectories despite identical per-date noise. Production therefore uses a
fixed batch of four, pads only the final short batch, discards padded outputs,
and records this numerical contract in every NetCDF. At step-80k-equivalent
weights the measured cost was 3.43 seconds per day and 5.29 GiB peak memory.
