import torch

from losses import feature_matching_loss, hinge_discriminator_loss, hinge_generator_loss, masked_mse
from model_gan import Discriminator, Generator
from train_gan import set_requires_grad


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


def test_discriminator_output_and_land_masking():
    condition, mask, target = inputs(); model = discriminator().eval()
    changed = target.clone(); changed[mask == 0] = 1e6
    first = model(target, condition, mask); second = model(changed, condition, mask)
    assert first.shape[0] == 2 and first.ndim == 2; torch.testing.assert_close(first, second)


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
