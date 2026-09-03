# GAN training and loss-ablation guide

## Recommended starting point

Use `configs/gan_sr_v2.json` for unconstrained conditional GAN experiments or
`configs/gan_sr_v3_hard_consistency.json` when every generated ocean-block mean
must equal the supplied coarse SST. Do not start from the retired
`configs/gan_sr.json`: its generator was trapped near bilinear interpolation by
two consecutive zero-initialized projections.

The training program is always `src/train_gan.py`. Architecture, objective,
optimizer, callbacks, data periods, and output locations are controlled by the
JSON config; there is no need to edit the Python training loop for a loss
sensitivity experiment.

## Generator objective

After `adversarial_start_step`, the default v2/v2b/v3 generator minimizes

```text
L_G = λ_content L_content
    + λ_gradient L_gradient
    + λ_spectral L_spectral
    + λ_feature_matching L_feature_matching
    + λ_adversarial L_adversarial.
```

Before that step, feature matching and adversarial loss are disabled. The five
weights are the correspondingly named `lambda_*` entries in the config.

| Component | Exact implementation | Main role | Main risk when too large |
|---|---|---|---|
| `L_content` | Ocean-masked single-realisation MSE | Day-specific fidelity and stable large scales | Regression to the mean; suppressed stochastic detail |
| `L_gradient` | Charbonnier error of adjacent ocean-only x/y finite differences | Fronts and local gradients | Noisy or over-sharpened fronts |
| `L_spectral` | MSE of `log1p(abs(rfft2(field)))` at every 2-D Fourier bin after applying the common land mask | Match scale-by-scale amplitude | Ringing or excessive texture; mask-edge leakage |
| `L_feature_matching` | Ocean-masked L1 distance between fake and detached-real intermediate critic activations at both critic scales | Stable learned perceptual structure | Generator may optimize critic features more than SST fidelity |
| `L_adversarial` | `-mean(D(fake))` for the hinge generator objective | Realistic local texture | Hallucination, mode collapse, unstable competition |

The spectral training loss is **not radially integrated**. It compares each
`(k_y, k_x)` amplitude bin and ignores phase. Radially averaged spectra are used
only in callbacks/evaluation. Because land is zeroed before the FFT, the fixed
coastline acts as a spatial window and can leak power across wavenumbers; use
the same mask for real and generated fields and interpret coastal errors
separately.

Weights are multipliers, not percentages. Always inspect the logged weighted
contribution `lambda_i * L_i`; raw losses have different units and scales.

## Critic objective and variants

The default hinge critic minimizes

```text
L_D = mean(max(0, 1 - D(real))) + mean(max(0, 1 + D(fake))).
```

GAN-v2 and v3 critics see high-resolution SST plus the resized coarse condition
and mask. GAN-v2b is the controlled image-only-critic ablation: its critics see
masked high-resolution SST alone, although the mask still removes land logits
from reductions. All use two PatchGAN scales, meaning the critic outputs a map
of local real/fake scores rather than one score for the entire domain.

GAN-v3's exact coarse consistency is an architectural projection controlled by
`enforce_coarse_consistency = true`; it is not `lambda_conservation`. This
guarantees preservation of coarse means and therefore makes coarse-scale
climate-signal preservation exact by construction. It does not guarantee that
the within-block spatial response is correct.

## Create an isolated loss experiment

Use the helper so every ablation has a unique config and run directory:

```bash
pixi run python tools/create_gan_ablation_config.py \
  --name gan_v2_spectral_x2 \
  --note "Test whether stronger Fourier-amplitude matching improves high-wavenumber PSD" \
  --parent configs/gan_sr_v2.json \
  --spectral 0.4
```

The helper refuses to overwrite an existing config or `runs/<name>`. Inspect
the generated JSON, then run a CPU smoke test directly:

```bash
PYTHONPATH=src pixi run python src/train_gan.py \
  --config configs/experiments/gan_v2_spectral_x2.json \
  --smoke-steps 3 --device cpu
```

Create a dedicated PBS launcher by copying the closest active GAN job and
changing only the config path and PBS job name. Do not point two jobs at one
`output_dir`.

## Suggested sensitivity matrix

Change one factor at a time from the same checkpoint/data split/seed:

| Experiment | Weight change from v2 | Scientific question |
|---|---|---|
| No perceptual feature matching | `lambda_feature_matching: 1 → 0` | Are learned critic features improving structure or dominating the objective? |
| No Fourier amplitude loss | `lambda_spectral: 0.2 → 0` | Does the explicit spectrum term add value beyond gradient/adversarial losses? |
| Stronger spectrum | `lambda_spectral: 0.2 → 0.4` | Does high-wavenumber power improve without ringing? |
| No gradient term | `lambda_gradient: 1 → 0` | Which fronts are attributable to local finite-difference matching? |
| Weak/strong adversarial | `lambda_adversarial: 0.05 → 0.025/0.10` | Texture-fidelity versus instability trade-off |
| Image-only critic | v2b architecture flags | Does critic access to the coarse state improve conditional consistency? |
| Hard consistency | v3 projection | Does exact block agreement improve deployment robustness? |

Minimum reporting for every ablation should include historical and future daily
RMSE/bias, climatology, ocean-interior and coastal errors, radial PSD/RALSD,
daily distribution diagnostics, ensemble dispersion, and both OFAM and
ACCESS-CM2 climate-signal metrics. A visually sharper map is not sufficient.

## Release checklist

1. Give the config, `name`, `output_dir`, and `smoke_output_dir` unique values.
2. Record the parent checkpoint/config and one scientific hypothesis.
3. Run the focused CPU tests and three-step smoke training.
4. Run the matching H200 smoke job and inspect generator/critic gradients,
   non-finite checks, memory, and a generated-minus-bilinear diagnostic.
5. Confirm loss magnitudes and weighted contributions stay bounded after the
   adversarial start.
6. Inspect callback images, spectra, and coarse consistency before production.
7. Run inference with EMA generator weights and validate dates, masks, physical
   ranges, and departure from bilinear interpolation.
8. Compare against the unchanged baseline with identical dates and seeds.
