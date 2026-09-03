# GAN objectives and autoregressive lag FiLM

This note records the exact objectives used by `gan_sr_v2`,
`gan_sr_v2b_image_only_critic`, and `gan_sr_v3_hard_consistency`, and separates
them from the lag-conditioning control in `flow_ar_residual_memory`.

## GAN generator objective

After the first 5,000 generator updates, all three current GAN variants use

```text
L_G = 5 L_content + L_gradient + 0.2 L_spectral
      + L_feature_matching + 0.05 L_adversarial
```

Before update 5,000, the adversarial and feature-matching terms are disabled.
The terms are:

- `L_content`: ocean-only, single-realisation pixel MSE in normalized SST.
- `L_gradient`: ocean-interior Charbonnier error in horizontal and vertical
  finite differences.
- `L_spectral`: MSE between `log1p` Fourier amplitudes after both fields use
  the same zero-filled land mask.
- `L_feature_matching`: ocean-masked L1 distance between fake and detached-real
  feature maps from both critic scales.
- `L_adversarial = -mean(D(fake))`: hinge generator objective.

The discriminator uses the standard hinge critic loss

```text
L_D = mean(max(0, 1 - D(real))) + mean(max(0, 1 + D(fake))).
```

Invalid land patches are removed from the critic reduction rather than being
retained as zero logits. The critic is updated once per generator update, uses
spectral-normalized convolutions, and has native and half-resolution PatchGAN
branches. The generator uses an EMA for validation and inference.

This is therefore a conditional, multi-scale, reconstruction-regularized GAN,
not a minimal unconditional GAN. A minimal GAN would normally use only the
adversarial game (often with a single image discriminator); that is usually a
poor fit for deterministic geophysical downscaling because it does not by
itself require day-specific agreement with the supplied coarse boundary.

In the final 1,000 historical-training updates, the approximate weighted
contributions to the GAN-v2 objective were: feature matching 65.5%, adversarial
13.3%, gradient 12.5%, content 8.7%, and spectrum 0.06%. GAN-v2b was similar,
with feature matching at about 60.3%. Thus the current objective is stabilized
primarily by critic feature matching rather than by the adversarial scalar
alone. The weights should not be read as percentages because the raw terms have
different scales.

## Variant-specific conditioning

- GAN-v2: the critic sees masked high-resolution SST, the upsampled two-channel
  coarse condition, and the high-resolution ocean mask.
- GAN-v2b: the critic sees only masked high-resolution SST. The mask is still
  used to zero land and exclude invalid patches, but is not an input channel.
- GAN-v3: uses the GAN-v2 critic and applies an exact differentiable projection
  to the generator output so every valid ocean-block mean equals the supplied
  coarse SST. This is a hard architectural constraint, not an extra weighted
  loss. It adds a constant within each valid block and therefore preserves the
  generated within-block anomaly.

All three generators remain strongly conditional: low-resolution SST and its
mask enter the stem and every upsampling block, and bilinearly upsampled SST is
added as a residual large-scale skip. Random noise supplies unresolved detail.

## The 0.15 lag-FiLM cap is not part of a GAN

`lag_guidance_scale = 0.15` belongs only to the residual-memory autoregressive
flow model. It does not affect GAN-v2/v2b/v3, plain Flow-SR, or the 1024x1024
NOAA model.

At each of four UNet scales, the previous-day within-block anomaly is encoded
and converted into a bounded scale and shift:

```text
g = 0.15 sigmoid(gate_logit)
main_out = main * (1 + g tanh(scale)) + g tanh(shift).
```

Consequently, 0.15 is only a hard per-channel upper bound; it is not a fixed
15% blend of yesterday and today. In the final step-120,000 EMA, the learned
mean `g` values are 0.0223, 0.0388, 0.0492, and 0.0321 across the four scales.
The observed minima/maxima are 0.0186/0.0266, 0.0214/0.0509,
0.0255/0.0678, and 0.0238/0.0516. The current low-resolution SST boundary
still enters the complete frozen Flow-SR backbone without this cap.

The cap, within-block anomaly transform, 50% lag-path dropout, and exact daily
coarse projection were introduced together because the legacy full-state lag
path overrode the current low-resolution state and drifted during rollouts.
The main risk is now the opposite one: memory can be too weak to preserve
temporal coherence. That should be decided from the one-year evolution ratio,
pointwise correlation, and spectrum—not by increasing 0.15 in isolation.

As a scale diagnostic (not a population metric), one normalized 2011 test
sample at flow time 0.5 gave lag-induced fusion changes of 2.2%, 13.0%, 26.7%,
and 15.1% of the main-feature RMS at the four scales. Replacing its previous
state by zero changed the final predicted velocity by 1.85% RMS. This confirms
both that the memory path is active and that `0.15` cannot be interpreted as a
literal 15% share of the final prediction.
