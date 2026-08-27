"""Compact, fine-detail conditional GAN for 16x SST super-resolution.

The generator follows the ESRGAN residual-in-residual dense-block principle,
but enlarges progressively so most computation stays off the 512x512 grid. A
bilinear SST skip preserves large scales while two spectral-normalised critics
judge native and half-resolution structure.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(values, size=size, mode="bilinear", align_corners=False)


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(channels + i * growth_channels, growth_channels, 3, padding=1)
            for i in range(4)
        ])
        self.fuse = nn.Conv2d(channels + 4 * growth_channels, channels, 3, padding=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = [values]
        for layer in self.layers:
            features.append(F.leaky_relu(layer(torch.cat(features, dim=1)), 0.2))
        return values + 0.2 * self.fuse(torch.cat(features, dim=1))


class RRDB(nn.Module):
    def __init__(self, channels: int, growth_channels: int):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualDenseBlock(channels, growth_channels) for _ in range(3)
        ])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + 0.2 * self.blocks(values)


class UpsampleBlock(nn.Module):
    """Artifact-resistant nearest-neighbour 2x upsampling plus refinement."""

    def __init__(self, channels: int, condition_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Conv2d(condition_channels + 1, channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, values, condition, mask):
        size = (values.shape[-2] * 2, values.shape[-1] * 2)
        values = F.interpolate(values, size=size, mode="nearest")
        values = F.leaky_relu(self.conv(values), 0.2)
        context = torch.cat((resize(condition, size), resize(mask, size)), dim=1)
        return values + self.refine(values + self.condition(context))


class Generator(nn.Module):
    """Progressive compact RRDB generator with a physical coarse-field skip."""

    def __init__(self, base_channels=48, levels=4, condition_channels=2,
                 target_channels=1, noise_channels=4, attention=True,
                 attention_heads=4, residual=True, rrdb_blocks=4,
                 growth_channels=24):
        super().__init__()
        del attention, attention_heads
        self.noise_channels = int(noise_channels)
        self.target_channels = int(target_channels)
        self.residual = bool(residual)
        self.levels = int(levels)
        self.stem = nn.Conv2d(condition_channels + noise_channels + 1, base_channels, 3, padding=1)
        self.trunk = nn.Sequential(*[
            RRDB(base_channels, growth_channels) for _ in range(rrdb_blocks)
        ])
        self.trunk_fuse = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.upsample = nn.ModuleList([
            UpsampleBlock(base_channels, condition_channels) for _ in range(levels)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, target_channels, 3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def sample_noise(self, condition, shape=None, generator=None):
        del shape
        return torch.randn((condition.shape[0], self.noise_channels, *condition.shape[-2:]),
                           device=condition.device, dtype=condition.dtype,
                           generator=generator)

    def forward(self, condition, ocean_mask, noise=None, generator=None):
        coarse_shape = tuple(int(v) for v in condition.shape[-2:])
        target_shape = tuple(int(v) for v in ocean_mask.shape[-2:])
        factor = 2 ** self.levels
        if (coarse_shape[0] * factor, coarse_shape[1] * factor) != target_shape:
            raise ValueError(
                f"levels={self.levels} imply {coarse_shape[0]*factor}x"
                f"{coarse_shape[1]*factor}, mask is {target_shape}"
            )
        if noise is None:
            noise = self.sample_noise(condition, generator=generator)
        elif noise.shape[-2:] != coarse_shape:
            noise = resize(noise, coarse_shape)
        coarse_mask = resize(ocean_mask, coarse_shape)
        hidden = self.stem(torch.cat((condition, noise, coarse_mask), dim=1))
        hidden = hidden + self.trunk_fuse(self.trunk(hidden))
        for block in self.upsample:
            hidden = block(hidden, condition, ocean_mask)
        output = self.head(hidden)
        if self.residual:
            output = output + resize(condition[:, :self.target_channels], target_shape)
        return output * ocean_mask


class SpectralConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1)
        )

    def forward(self, values):
        return F.leaky_relu(self.conv(values), 0.2, inplace=True)


class PatchCritic(nn.Module):
    def __init__(self, input_channels: int, base_channels: int, levels: int):
        super().__init__()
        channels = [base_channels * min(2**i, 8) for i in range(levels)]
        self.stem = SpectralConvBlock(input_channels, channels[0])
        self.blocks = nn.ModuleList([
            SpectralConvBlock(channels[i - 1], channels[i]) for i in range(1, levels)
        ])
        self.output = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(channels[-1], 1, 3, padding=1)
        )

    def forward(self, values, mask):
        features = []
        hidden = self.stem(values)
        features.append(hidden)
        for block in self.blocks:
            hidden = block(hidden)
            features.append(hidden)
        logits = self.output(hidden)
        patch_mask = (resize(mask, logits.shape[-2:]) > 0.5).to(logits.dtype)
        return logits * patch_mask, features


class Discriminator(nn.Module):
    """Native/half-resolution conditional PatchGAN critics."""

    def __init__(self, base_channels=32, levels=4, condition_channels=2,
                 target_channels=1, scales=2):
        super().__init__()
        inputs = target_channels + condition_channels + 1
        self.critics = nn.ModuleList([
            PatchCritic(inputs, base_channels, levels) for _ in range(scales)
        ])

    def forward(self, field, condition, ocean_mask, return_features=False):
        values = torch.cat((field * ocean_mask, resize(condition, field.shape[-2:]), ocean_mask), dim=1)
        mask = ocean_mask
        logits, all_features, all_masks = [], [], []
        for index, critic in enumerate(self.critics):
            if index:
                values = F.avg_pool2d(values, 2)
                mask = F.avg_pool2d(mask, 2)
            score, features = critic(values, mask)
            logits.append(score.flatten(1))
            all_features.extend(features)
            all_masks.extend([resize(mask, feature.shape[-2:]) for feature in features])
        joined = torch.cat(logits, dim=1)
        return (joined, all_features, all_masks) if return_features else joined


def build_generator(config: dict) -> Generator:
    factor = int(config.get("coarsen_factor", 16))
    levels = int(config.get("generator_upsample_levels", round(math.log2(factor))))
    if 2 ** levels != factor:
        raise ValueError(f"coarsen_factor={factor} must be a power of two")
    return Generator(
        base_channels=int(config.get("generator_channels", config.get("base_channels", 48))),
        levels=levels,
        condition_channels=int(config.get("condition_channels", 2)),
        target_channels=int(config.get("target_channels", 1)),
        noise_channels=int(config.get("noise_channels", 4)),
        residual=bool(config.get("generator_residual", True)),
        rrdb_blocks=int(config.get("rrdb_blocks", 4)),
        growth_channels=int(config.get("growth_channels", 24)),
    )


def build_discriminator(config: dict) -> Discriminator:
    return Discriminator(
        base_channels=int(config.get("discriminator_base_channels", 32)),
        levels=int(config.get("discriminator_levels", 4)),
        condition_channels=int(config.get("condition_channels", 2)),
        target_channels=int(config.get("target_channels", 1)),
        scales=int(config.get("discriminator_scales", 2)),
    )
