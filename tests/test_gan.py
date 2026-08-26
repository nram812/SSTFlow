import torch

from losses import hinge_discriminator_loss, hinge_generator_loss, masked_mse
from model_gan import Discriminator, Generator


def inputs():
    condition = torch.randn(2, 2, 4, 4); mask = torch.ones(2, 1, 32, 32); mask[..., :8, :8] = 0
    target = torch.randn(2, 1, 32, 32) * mask
    return condition, mask, target


def generator(): return Generator(base_channels=4, levels=2, noise_channels=2, attention=False)
def discriminator(): return Discriminator(base_channels=4, levels=2)


def test_generator_output_shape_mask_and_bilinear_baseline():
    condition, mask, target = inputs(); output = generator()(condition, mask)
    assert output.shape == target.shape and torch.count_nonzero(output[mask == 0]) == 0
    baseline = torch.nn.functional.interpolate(condition[:, :1], size=(32, 32), mode="bilinear", align_corners=False) * mask
    torch.testing.assert_close(output, baseline)


def test_generator_noise_changes_output_after_perturbation():
    condition, mask, _ = inputs(); model = generator()
    with torch.no_grad(): model.head.weight.normal_(std=0.1); model.backbone.output.weight.normal_(std=0.1)
    assert not torch.equal(model(condition, mask, torch.randn(2, 2, 32, 32)), model(condition, mask, torch.randn(2, 2, 32, 32)))


def test_discriminator_output_and_land_masking():
    condition, mask, target = inputs(); model = discriminator().eval()
    changed = target.clone(); changed[mask == 0] = 1e6
    first = model(target, condition, mask); second = model(changed, condition, mask)
    assert first.shape == (2, 1, 8, 8); torch.testing.assert_close(first, second)


def test_one_gan_step_runs_and_content_is_single_sample():
    condition, mask, target = inputs(); gen, disc = generator(), discriminator()
    fake = gen(condition, mask); critic = hinge_discriminator_loss(disc(target, condition, mask), disc(fake.detach(), condition, mask))
    critic.backward(); assert torch.isfinite(critic)
    fake = gen(condition, mask); content = masked_mse(fake, target, mask)
    loss = 10 * content + hinge_generator_loss(disc(fake, condition, mask)); loss.backward()
    assert torch.isfinite(loss) and content == masked_mse(fake, target, mask)
