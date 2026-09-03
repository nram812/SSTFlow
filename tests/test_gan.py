import torch

from losses import (
    feature_matching_loss,
    hinge_discriminator_loss,
    hinge_generator_loss,
    masked_mse,
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
)
from consistency import coarse_consistency_mse, masked_block_mean
from model_gan import Discriminator, Generator, build_discriminator
from train_gan import (
    set_requires_grad,
    validate_resume_checksum,
    wasserstein_gradient_penalty,
)


def inputs():
    condition = torch.randn(2, 2, 4, 4); mask = torch.ones(2, 1, 32, 32); mask[..., :8, :8] = 0
    target = torch.randn(2, 1, 32, 32) * mask
    return condition, mask, target


def generator(): return Generator(base_channels=4, levels=3, noise_channels=2, attention=False, rrdb_blocks=1, growth_channels=2)
def discriminator(): return Discriminator(base_channels=4, levels=2)


def test_generator_output_shape_mask_and_bilinear_baseline():
    condition, mask, target = inputs(); output = generator()(condition, mask)
    assert output.shape == target.shape and torch.count_nonzero(output[mask == 0]) == 0
    baseline = torch.nn.functional.interpolate(condition[:, :1], size=(32, 32), mode="bilinear", align_corners=False) * mask
    torch.testing.assert_close(output, baseline)


def test_generator_noise_changes_output_after_perturbation():
    condition, mask, _ = inputs(); model = generator()
    with torch.no_grad(): model.head[-1].weight.normal_(std=0.1)
    assert not torch.equal(model(condition, mask, torch.randn(2, 2, 32, 32)), model(condition, mask, torch.randn(2, 2, 32, 32)))


def test_generator_learns_spatial_residual_instead_of_only_a_bias():
    """Guard against consecutive zero projections deadlocking the generator.

    The original production GAN had a zero-initialised U-Net output immediately
    followed by a second zero-initialised head.  Only the final bias could learn,
    so every apparent high-resolution residual was spatially constant.
    """
    torch.manual_seed(4)
    condition, mask, _ = inputs()
    model = generator()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    noise = torch.randn(2, 2, 4, 4)
    baseline = torch.nn.functional.interpolate(
        condition[:, :1], size=(32, 32), mode="bilinear", align_corners=False
    ) * mask
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij"
    )
    detail = (0.2 * torch.sin(7 * xx) * torch.cos(5 * yy))[None, None]
    target = (baseline + detail) * mask
    initial = masked_mse(model(condition, mask, noise=noise), target, mask).detach()

    for _ in range(12):
        generated = model(condition, mask, noise=noise)
        loss = masked_mse(generated, target, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    generated = model(condition, mask, noise=noise).detach()
    residual = (generated - baseline)[mask.bool()]
    final = masked_mse(generated, target, mask)
    assert final < initial
    assert residual.std() > 1.0e-3


def test_hard_constrained_generator_exactly_matches_valid_coarse_means():
    condition, mask, _ = inputs()
    condition[:, 1] = 1
    condition[:, 1, 0, 0] = 0
    model = Generator(
        base_channels=4,
        levels=3,
        noise_channels=2,
        attention=False,
        rrdb_blocks=1,
        growth_channels=2,
        enforce_coarse_consistency=True,
    )
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.1)
    output = model(condition, mask)
    means, _ = masked_block_mean(output, mask, condition.shape[-2:])
    valid = condition[:, 1:2].bool()
    torch.testing.assert_close(
        means[valid], condition[:, :1][valid], atol=2.0e-6, rtol=0
    )
    assert float(coarse_consistency_mse(output, condition, mask)) < 1.0e-12


def test_hard_constraint_preserves_generator_gradients():
    condition, mask, target = inputs()
    condition[:, 1] = 1
    model = Generator(
        base_channels=4,
        levels=3,
        noise_channels=2,
        attention=False,
        rrdb_blocks=1,
        growth_channels=2,
        enforce_coarse_consistency=True,
    )
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.1)
    loss = masked_mse(model(condition, mask), target, mask)
    loss.backward()
    assert model.stem.weight.grad is not None
    assert torch.isfinite(model.stem.weight.grad).all()
    assert float(model.stem.weight.grad.norm()) > 0


def test_discriminator_output_and_land_masking():
    condition, mask, target = inputs(); model = discriminator().eval()
    changed = target.clone(); changed[mask == 0] = 1e6
    first = model(target, condition, mask); second = model(changed, condition, mask)
    assert first.shape[0] == 2 and first.ndim == 2; torch.testing.assert_close(first, second)


def test_image_only_discriminator_has_one_input_channel_and_ignores_condition():
    condition, mask, target = inputs()
    model = Discriminator(
        base_channels=4,
        levels=2,
        condition_on_coarse=False,
        condition_on_mask=False,
    ).eval()
    first_convolution = model.critics[0].stem.conv
    assert first_convolution.in_channels == 1
    first = model(target, condition, mask)
    second = model(target, condition + 1000.0, mask)
    torch.testing.assert_close(first, second)


def test_discriminator_ablation_flags_are_read_from_config():
    model = build_discriminator({
        "discriminator_base_channels": 4,
        "discriminator_levels": 2,
        "discriminator_scales": 2,
        "condition_channels": 2,
        "target_channels": 1,
        "discriminator_condition_on_coarse": False,
        "discriminator_condition_on_mask": False,
    })
    assert model.critics[0].stem.conv.in_channels == 1
    assert model.condition_on_coarse is False
    assert model.condition_on_mask is False


def test_discriminator_excludes_invalid_patches_from_hinge_reduction():
    condition, mask, target = inputs()
    model = discriminator().eval()
    logits = model(target, condition, mask)
    full_patch_count = sum(
        (32 // (2 ** (scale + 2))) ** 2 for scale in range(2)
    )
    assert 0 < logits.shape[1] < full_patch_count


def test_multiscale_features_support_masked_feature_matching():
    condition, mask, target = inputs(); model = discriminator().eval()
    fake = target + 0.1 * mask
    _, fake_features, feature_masks = model(fake, condition, mask, return_features=True)
    _, real_features, _ = model(target, condition, mask, return_features=True)
    loss = feature_matching_loss(fake_features, real_features, feature_masks)
    assert torch.isfinite(loss) and float(loss) > 0


def test_one_gan_step_runs_and_content_is_single_sample():
    condition, mask, target = inputs(); gen, disc = generator(), discriminator()
    fake = gen(condition, mask); critic = hinge_discriminator_loss(disc(target, condition, mask), disc(fake.detach(), condition, mask))
    critic.backward(); assert torch.isfinite(critic)
    fake = gen(condition, mask); content = masked_mse(fake, target, mask)
    loss = 10 * content + hinge_generator_loss(disc(fake, condition, mask)); loss.backward()
    assert torch.isfinite(loss) and content == masked_mse(fake, target, mask)


def test_generator_step_does_not_accumulate_critic_gradients():
    condition, mask, target = inputs(); gen, disc = generator(), discriminator()
    set_requires_grad(disc, False)
    fake = gen(condition, mask)
    loss = masked_mse(fake, target, mask) + hinge_generator_loss(
        disc(fake, condition, mask)
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in gen.parameters())
    assert all(parameter.grad is None for parameter in disc.parameters())
    set_requires_grad(disc, True)


def test_wasserstein_cost_signs_reward_real_high_and_fake_low():
    real = torch.tensor([[2.0, 3.0]])
    fake = torch.tensor([[-2.0, -1.0]])
    assert float(wasserstein_discriminator_loss(real, fake)) < 0
    assert float(wasserstein_generator_loss(fake)) > 0


class UnitGradientCritic(torch.nn.Module):
    def forward(self, field, condition, mask):
        del condition
        valid = mask.expand_as(field)
        count = valid.flatten(1).sum(dim=1).sqrt().clamp_min(1)
        return ((field * valid).flatten(1).sum(dim=1) / count)[:, None]


def test_wgan_gp_uses_per_sample_unit_norm_and_ocean_mask():
    condition, mask, real = inputs()
    fake = torch.randn_like(real) * mask
    penalty, norm = wasserstein_gradient_penalty(
        UnitGradientCritic(), real, fake, condition, mask,
        generator=torch.Generator().manual_seed(30),
    )
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-6, rtol=0)
    assert float(penalty) < 1e-12


def test_wgan_gp_backpropagates_finite_critic_gradients():
    condition, mask, real = inputs()
    model = discriminator()
    fake = torch.randn_like(real) * mask
    penalty, norm = wasserstein_gradient_penalty(
        model, real, fake, condition, mask,
        generator=torch.Generator().manual_seed(31),
    )
    penalty.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert torch.isfinite(norm).all() and bool((norm > 0).all())


def test_resume_checksum_is_enforced(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"documented weights")
    provenance = validate_resume_checksum({
        "resume_from": str(checkpoint),
        "resume_from_sha256": __import__("hashlib").sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
    })
    assert provenance["path"] == str(checkpoint.resolve())
    try:
        validate_resume_checksum({
            "resume_from": str(checkpoint),
            "resume_from_sha256": "0" * 64,
        })
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("wrong resume checksum was accepted")
