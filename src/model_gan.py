"""PyTorch conditional GAN baseline for SST super-resolution.

The generator reuses the flow U-Net backbone with the flow time pinned to one,
so the three experiments share an identical receptive field and parameter
budget.  It maps ``(coarse condition, ocean mask, latent noise)`` directly to
the high-resolution residual around the bilinearly upsampled predictor, which is
a much easier target than the full field and keeps early training stable.

Following the request, the content loss is a **single** masked MSE between one
generated sample and the truth - not the ensemble-mean MSE used by
Harris et al. (2022) style implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SuperResolutionFlowUNet, group_count, resize_to


class Generator(nn.Module):
    """Conditional generator built on the shared super-resolution backbone."""

    def __init__(
        self,
        base_channels: int = 32,
        levels: int = 4,
        condition_channels: int = 2,
        target_channels: int = 1,
        noise_channels: int = 4,
        attention: bool = True,
        attention_heads: int = 4,
        residual: bool = True,
    ):
        super().__init__()
        self.noise_channels = int(noise_channels)
        self.target_channels = int(target_channels)
        self.condition_channels = int(condition_channels)
        self.residual = bool(residual)
        self.backbone = SuperResolutionFlowUNet(
            base_channels=base_channels,
            levels=levels,
            condition_channels=condition_channels,
            target_channels=self.noise_channels,
            attention=attention,
            attention_heads=attention_heads,
        )
        # The backbone emits `noise_channels` maps; project them to the target.
        self.head = nn.Conv2d(self.noise_channels, self.target_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def sample_noise(
        self, condition: torch.Tensor, shape: tuple[int, int], generator=None
    ) -> torch.Tensor:
        return torch.randn(
            (condition.shape[0], self.noise_channels, *shape),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )

    def forward(
        self,
        condition: torch.Tensor,
        ocean_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        generator=None,
    ) -> torch.Tensor:
        shape = tuple(int(value) for value in ocean_mask.shape[-2:])
        if noise is None:
            noise = self.sample_noise(condition, shape, generator)
        noise = noise * ocean_mask
        ones = torch.ones(
            condition.shape[0], device=condition.device, dtype=condition.dtype
        )
        hidden = self.backbone(noise, condition, ocean_mask, ones)
        output = self.head(hidden)
        if self.residual:
            baseline = resize_to(condition[:, : self.target_channels], output)
            output = output + baseline
        return output * ocean_mask


class SpectralConvBlock(nn.Module):
    """Strided, spectrally normalised convolution used by the critic."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(input_channels, output_channels, 4, stride=stride, padding=1)
        )
        self.norm = nn.GroupNorm(group_count(output_channels), output_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.norm(self.conv(values)), 0.2)


class Discriminator(nn.Module):
    """Conditional PatchGAN critic.

    The critic sees the candidate field, the bilinearly upsampled coarse
    predictor, and the ocean mask.  Its logit map is multiplied by a downsampled
    ocean mask before reduction so land patches cannot influence the score.
    """

    def __init__(
        self,
        base_channels: int = 32,
        levels: int = 4,
        condition_channels: int = 2,
        target_channels: int = 1,
    ):
        super().__init__()
        channels = [base_channels * min(2**level, 8) for level in range(levels)]
        input_channels = target_channels + condition_channels + 1
        blocks = [
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(input_channels, channels[0], 4, stride=2, padding=1)
            )
        ]
        self.stem = nn.Sequential(*blocks)
        self.blocks = nn.ModuleList(
            [
                SpectralConvBlock(channels[level - 1], channels[level])
                for level in range(1, levels)
            ]
        )
        self.output = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(channels[-1], 1, 3, padding=1)
        )
        self.downsample_factor = 2**levels

    def forward(
        self,
        field: torch.Tensor,
        condition: torch.Tensor,
        ocean_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.cat(
            (field * ocean_mask, resize_to(condition, field), ocean_mask), dim=1
        )
        hidden = F.leaky_relu(self.stem(hidden), 0.2)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output(hidden)
        patch_mask = (
            F.avg_pool2d(ocean_mask, self.downsample_factor) > 0.0
        ).to(logits.dtype)
        patch_mask = resize_to(patch_mask, logits)
        return logits * patch_mask


def build_generator(config: dict) -> Generator:
    return Generator(
        base_channels=int(config.get("base_channels", 32)),
        levels=int(config.get("levels", 4)),
        condition_channels=int(config.get("condition_channels", 2)),
        target_channels=int(config.get("target_channels", 1)),
        noise_channels=int(config.get("noise_channels", 4)),
        attention=bool(config.get("attention", True)),
        attention_heads=int(config.get("attention_heads", 4)),
        residual=bool(config.get("generator_residual", True)),
    )


def build_discriminator(config: dict) -> Discriminator:
    return Discriminator(
        base_channels=int(config.get("discriminator_base_channels", 32)),
        levels=int(config.get("discriminator_levels", 4)),
        condition_channels=int(config.get("condition_channels", 2)),
        target_channels=int(config.get("target_channels", 1)),
    )
