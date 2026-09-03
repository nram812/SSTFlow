# Legacy `flow_ar` training and rollout audit

## Scope

This audit applies specifically to `runs/flow_ar`, whose saved configuration
and EMA weights were produced on 2026-08-27. It is not the later
`flow_ar_coarse_guided` experiment. The production inference definition is the
saved `runs/flow_ar/config_used.json`, not the current development config.

## What the checkpoint was trained to do

For consecutive OFAM days, each training item was:

```
previous = y(t)
condition = [coarse_sst(t+1), coarse_ocean_mask]
target = y(t+1)
```

Pairs were drawn only from 1979-01-01 through 2008-12-31 and could not cross a
configured range boundary. The high-resolution and coarse SST used the same
scalar training mean and standard deviation. Land was zero only after
normalization and was excluded from the objective with the fixed ocean mask.

The velocity objective was the standard straight-line rectified-flow target:

```
z(tau) = (1 - tau) epsilon + tau y(t+1)
u = y(t+1) - epsilon
L_velocity = masked_MSE(v_theta(z, x(t+1), y(t), tau), u)
```

The legacy run also used a state-space loss from step 20,000 onward, every
fourth update, at weight 0.1. It unrolled a four-step Heun solve for one day and
compared that one generated day with truth. It did **not** train through a
multi-day autoregressive chain: the lag state for this auxiliary loss was still
the observed `y(t)`. The stored history confirms 18,500 such updates through
checkpoint step 94,000.

The usable `checkpoint.pt`, `model.pt`, and `model_ema.pt` are all exactly the
same step-94,000 state. The scheduler log continued to about step 95,550 before
the job was canceled, but those later in-memory updates were not checkpointed;
the run also never reached its configured 120,000 steps and has no passing
`status.json`. Production inference therefore records the EMA checksum rather
than inferring a completion step from the final text log.

The objective is mathematically valid conditional rectified flow. The principal
modeling weakness is exposure bias: free inference conditions on generated
states whereas both velocity and state-space training condition on a true
previous day. SST persistence is unusually strong here (one-day persistence
RMSE about 0.105 degC), and saved single-step validation was still worse than
persistence. A numerical solver cannot repair that learned-model limitation.

## Defects found in the previous inference

1. **Checkpoint semantics changed silently.** The legacy checkpoint was trained
   with the complete normalized `y(t)` entering the lag encoder and an uncapped
   sigmoid FiLM gate. Later source code high-pass filtered `y(t)` into a
   within-block anomaly and multiplied the gate by 0.25. Parameter shapes did
   not change, so strict state-dict loading succeeded while evaluating a
   different function. Missing lag-control keys now explicitly resolve to the
   legacy `full_state` and scale `1.0`; new models must explicitly record
   `within_block_anomaly` and `0.25`.

2. **The first attempted rollout had the condition axes reversed.** It passed
   `(channel, lead, lat, lon)` where the sampler requires
   `(batch, lead, channel, lat, lon)` and failed before inference.

3. **The retry requested a nonexistent item-level `date`.** Autoregressive
   dataset items expose `date_window`; the job completed expensive sampling and
   then crashed before writing the rollout.

4. **The generic evaluator ignored rollout CLI solver overrides.** Its
   teacher-forced sample used the requested CLI solver, but its free rollout
   used `rollout_sampler_steps` and `sampler` from the saved config. It would
   therefore have silently used 25-step Heun even when asked for 75-step
   AB2-PC. Solver selection is now threaded into both paths.

5. **Teacher-forced AR sample dates were one day early.** The generic date
   accessor returned the pair's previous-state date rather than the target
   date. AR evaluation now uses the final entry of `date_window`.

The old `evaluation/test_samples_heun_25step.nc` is a one-item,
teacher-forced product and is not a one-year autoregressive forecast.

## Correct one-year rollout

The production period is deliberately contained in the test split:

```
observed initial state: 2011-01-01
first generated state:  2011-01-02
last generated state:   2011-12-31
generated leads:        364
truth resets:            0
```

At physical lead `k`, the model receives independent masked Gaussian flow
noise, the coarse SST for that exact target date, and the generated field from
lead `k-1`. The lag state remains fixed throughout that day's flow ODE. Solver
velocity history is local to a single ODE and is reset before the next physical
day.

The requested AB2-PC solve uses:

```
predict: y*_(n+1) = y_n + h (3/2 f_n - 1/2 f_(n-1))
correct: y_(n+1)  = y_n + h/2 (f_n + f(t_(n+1), y*_(n+1)))
```

The first interval uses an Euler predictor and the same trapezoidal corrector
(Heun) to establish history. There are 75 fixed intervals from flow time zero
to one and two velocity evaluations per interval. This is separate from plain
AB2 and from the existing AB3/AM3 predictor-corrector.

After each daily solve, a differentiable-free deterministic projection adds one
constant per valid ocean block so that the generated 16x16 ocean mean exactly
matches that day's coarse SST. It leaves within-block fine structure unchanged.
This prevents block-scale drift from being fed into the next legacy full-state
lag input and makes the daily boundary condition authoritative.

## Gates and products

The H200 preflight runs five generated days with the complete 75-step solver,
checks physical and normalized ranges, validates exact coarse means, profiles
memory/runtime, and quantifies the difference caused by the former semantic
mismatch. Production is released only after that job passes.

The production product and companions are:

```
runs/flow_ar/evaluation/test_2011_free_rollout_ab2pc_75step.nc
runs/flow_ar/evaluation/test_2011_free_rollout_ab2pc_75step.metrics.json
runs/flow_ar/evaluation/test_2011_free_rollout_ab2pc_75step.validation.json
runs/flow_ar/evaluation/test_2011_free_rollout_ab2pc_75step_skill.png
runs/flow_ar/evaluation/test_2011_free_rollout_ab2pc_75step_snapshots.png
```

The NetCDF contains generated and target SST, daily coarse SST, the observed
initial state, both masks, exact dates/leads, checkpoint checksum, lag semantics,
solver, step count, seed, and truth-reset count.
