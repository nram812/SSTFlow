# Residual-memory autoregressive flow

## Why the legacy model drifted

The legacy `runs/flow_ar` model consumed the complete previous high-resolution
SST state, used only 10% whole-path dropout, and allowed an uncapped lag FiLM
path. It was trained with observed previous states but evaluated using its own
generated states. With daily SST persistence near one, this combination made
the easiest training solution place too much weight on memory. Correcting the
rollout and enforcing daily coarse block means made inference stable, but did
not change the learned teacher-forced dependence.

## Selected design

`flow_ar_residual_memory` is a new configuration rather than an incompatible
fine-tune of the legacy checkpoint:

```text
current 32x32 SST ──> frozen successful flow_sr U-Net ───────────┐
                                                                 ├─> velocity
previous 512x512 SST ─> remove each block mean ─> lag encoder ───┘
                                               bounded FiLM <= 0.15
```

- The 5,684,409-parameter plain `flow_sr` EMA is checksum-loaded and frozen.
- Only the 382,816-parameter lag encoder and FiLM fusion path train (6.31%).
- At initialization every FiLM projection is zero, so the AR model is exactly
  the successful non-autoregressive flow for every state and condition.
- Memory receives only within-block anomalies; it cannot carry yesterday's
  block-scale SST.
- The entire memory path is removed on 50% of training batches.
- Its multiplicative/additive FiLM correction is capped at 0.15.
- Each generated day is projected to the current 32x32 ocean-block means before
  it becomes the next lag state.
- Training retains the ordinary straight-line rectified-flow velocity loss;
  there is no teacher-forced state-space rollout penalty encouraging copying.

This is functionally a residual-memory model: the trusted current-SST mapping
cannot be forgotten, and memory can learn only a bounded fine-scale correction.

## Verification and run

The focused tests require exact equality with `flow_sr` at initialization,
strict source checksums, gradients only in lag modules, and bitwise invariance
of the backbone. The complete suite has 176 passes and one restricted-sandbox
skip. H200 gate `6408884` updated all 44 lag tensors while preserving every
backbone tensor bitwise, generated a finite five-day rollout, and matched the
daily coarse boundary to `1.01e-13` normalized MSE. Production job `6408885`
trains to 120,000 updates in `runs/flow_ar_residual_memory`.

## Temporal coupling at inference

Single-day conditional flow matching identifies the distribution for each day,
but it does not identify how random draws should be coupled through time. Using
independent latent noise every day therefore creates avoidable high-frequency
flicker even when the autoregressive state remains stable. The production
rollout uses one common latent flow field (`noise_correlation = 1.0`) across
the forecast. This leaves each day's standard-normal marginal unchanged while
making stochastic texture temporally coherent.

H200 job `6408912` compared correlations 0, 0.5, 0.9, 0.95, 0.99, and 1.0 on
the same ten-day window and seed. Common noise reduced the generated/observed
daily-evolution ratio from 4.14 to 2.52; ten-day RMSE changed from 0.409 to
0.435 degC. The final one-year run therefore uses common noise together with
the learned residual memory and daily hard coarse projection. The result is
recorded as an inference choice, not misrepresented as a training loss.
