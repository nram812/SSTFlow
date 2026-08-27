"""Conditional velocity networks for SST super-resolution.

Two variants share one backbone:

``SuperResolutionFlowUNet``
    Plain low-to-high resolution flow matching.  Conditioning is the coarse
    field (2 channels: standardised SST and the coarse validity mask) plus the
    static high-resolution ocean mask.

``AutoregressiveSuperResolutionFlowUNet``
    Adds a separate, gated encoder for the previous day's high-resolution state.
    Sea-surface temperature is extremely persistent (lag-1 spatial correlation
    0.99993 in this dataset), so the lag pathway is deliberately throttled with
    dropout, path dropout, and a bounded gated-FiLM fusion.  Without that the
    model would simply copy yesterday and ignore the coarse predictor.

The coarse conditioning enters at *every* resolution: it is bilinearly resized
to the working grid of each stage, which is the cheapest reliable way to inject
a 32x32 predictor into a 512x512 target.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from consistency import within_block_anomaly


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the rectified-flow time in ``[0, 1]``."""

    def __init__(self, dimension: int):
        super().__init__()
        if dimension % 2:
            raise ValueError(f"Time embedding dimension must be even, got {dimension}")
        self.dimension = dimension
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=flow_time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = flow_time.reshape(-1, 1).float() * 1000.0 * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.projection(embedding.to(dtype=self.projection[0].weight.dtype))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.average = nn.AdaptiveAvgPool2d(1)
        self.maximum = nn.AdaptiveMaxPool2d(1)
        self.projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = self.projection(self.average(values))
        weights = weights + self.projection(self.maximum(values))
        return values * torch.sigmoid(weights)


class ResidualBlock(nn.Module):
    """Pre-norm residual block with FiLM conditioning on the flow time."""

    def __init__(
        self, input_channels: int, output_channels: int, time_dimension: int
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(input_channels), input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.time = nn.Linear(time_dimension, output_channels * 2)
        self.channel_attention = ChannelAttention(output_channels)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        # Start as an identity mapping: a zeroed output convolution keeps the
        # first optimiser steps small and removes a classic source of blow-ups.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(
        self, values: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        scale, shift = self.time(time_embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1.0 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        hidden = self.channel_attention(hidden)
        return hidden + self.skip(values)


class SelfAttention(nn.Module):
    """Multi-head self attention used only at the coarsest resolution."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        if channels % heads:
            raise ValueError(f"{channels} channels is not divisible by {heads} heads")
        self.heads = heads
        self.head_dimension = channels // heads
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = values.shape
        qkv = self.qkv(self.norm(values))
        qkv = qkv.reshape(batch, 3, self.heads, self.head_dimension, height * width)
        query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attended = F.scaled_dot_product_attention(
            query.transpose(-1, -2),
            key.transpose(-1, -2),
            value.transpose(-1, -2),
        ).transpose(-1, -2)
        attended = attended.reshape(batch, channels, height, width)
        return values + self.output(attended)


class LagBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(group_count(output_channels), output_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(group_count(output_channels), output_channels),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class LagEncoder(nn.Module):
    """Compress the previous high-resolution day into multiscale features."""

    def __init__(
        self,
        base_channels: int,
        dropout: float,
        path_dropout: float,
        input_channels: int = 2,
        levels: int = 4,
    ):
        super().__init__()
        self.path_dropout = float(path_dropout)
        channels = [base_channels * (2**level) for level in range(levels)]
        blocks = [LagBlock(input_channels, channels[0], dropout)]
        for level in range(1, levels):
            blocks.append(LagBlock(channels[level - 1], channels[level], dropout))
        self.blocks = nn.ModuleList(blocks)
        self.channels = channels

    def forward(self, previous_state: torch.Tensor) -> list[torch.Tensor]:
        if self.training and self.path_dropout > 0:
            keep = (
                torch.rand(
                    previous_state.shape[0],
                    1,
                    1,
                    1,
                    device=previous_state.device,
                )
                >= self.path_dropout
            ).to(previous_state.dtype)
            previous_state = previous_state * keep
        features = []
        hidden = previous_state
        for index, block in enumerate(self.blocks):
            if index:
                hidden = F.avg_pool2d(hidden, 2)
            hidden = block(hidden)
            features.append(hidden)
        return features


class GatedFiLM(nn.Module):
    """Bounded feature-wise modulation with a hard guidance-strength cap."""

    def __init__(
        self, lag_channels: int, main_channels: int, guidance_scale: float = 0.25
    ):
        super().__init__()
        if not 0.0 <= guidance_scale <= 1.0:
            raise ValueError("lag_guidance_scale must lie in [0, 1]")
        self.guidance_scale = float(guidance_scale)
        self.parameters_from_lag = nn.Conv2d(lag_channels, main_channels * 2, 1)
        self.gate_logit = nn.Parameter(torch.full((1, main_channels, 1, 1), -2.0))
        nn.init.zeros_(self.parameters_from_lag.weight)
        nn.init.zeros_(self.parameters_from_lag.bias)

    def forward(self, main: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
        if lag.shape[-2:] != main.shape[-2:]:
            lag = F.interpolate(
                lag, size=main.shape[-2:], mode="bilinear", align_corners=False
            )
        scale, shift = self.parameters_from_lag(lag).chunk(2, dim=1)
        gate = self.guidance_scale * torch.sigmoid(self.gate_logit)
        return main * (1.0 + gate * torch.tanh(scale)) + gate * torch.tanh(shift)


def resize_to(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.shape[-2:] == reference.shape[-2:]:
        return values
    return F.interpolate(
        values, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )


class SuperResolutionFlowUNet(nn.Module):
    """Velocity field ``v(state, condition, mask, t)`` on the fine grid."""

    def __init__(
        self,
        base_channels: int = 32,
        levels: int = 4,
        condition_channels: int = 2,
        target_channels: int = 1,
        attention: bool = True,
        attention_heads: int = 4,
    ):
        super().__init__()
        if levels < 2:
            raise ValueError("The U-Net needs at least two levels")
        self.levels = levels
        self.condition_channels = condition_channels
        self.target_channels = target_channels
        time_dimension = base_channels * 8
        channels = [base_channels * min(2**level, 8) for level in range(levels)]
        self.channels = channels

        self.time = TimeEmbedding(time_dimension)
        # Input: flow state + coarse condition + coarse mask + ocean mask.
        self.input_block = ResidualBlock(
            target_channels + condition_channels + 1, channels[0], time_dimension
        )
        self.down = nn.ModuleList(
            [
                ResidualBlock(
                    channels[level - 1] + condition_channels,
                    channels[level],
                    time_dimension,
                )
                for level in range(1, levels)
            ]
        )
        self.middle1 = ResidualBlock(
            channels[-1] + condition_channels, channels[-1], time_dimension
        )
        self.attention = (
            SelfAttention(channels[-1], attention_heads) if attention else nn.Identity()
        )
        self.middle2 = ResidualBlock(channels[-1], channels[-1], time_dimension)
        self.up = nn.ModuleList(
            [
                ResidualBlock(
                    channels[level] + channels[level - 1],
                    channels[level - 1],
                    time_dimension,
                )
                for level in range(levels - 1, 0, -1)
            ]
        )
        self.output_norm = nn.GroupNorm(group_count(channels[0]), channels[0])
        self.output = nn.Conv2d(channels[0], target_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _check(self, state, condition, ocean_mask):
        if state.shape[1] != self.target_channels:
            raise ValueError(
                f"Expected {self.target_channels} state channels, got {state.shape[1]}"
            )
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"Expected {self.condition_channels} condition channels, "
                f"got {condition.shape[1]}"
            )
        if ocean_mask.shape[-2:] != state.shape[-2:]:
            raise ValueError(
                f"Ocean mask grid {tuple(ocean_mask.shape[-2:])} does not match "
                f"state grid {tuple(state.shape[-2:])}"
            )

    def encode(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        ocean_mask: torch.Tensor,
        embedding: torch.Tensor,
        lag_features: list[torch.Tensor] | None = None,
        fusion: nn.ModuleList | None = None,
    ):
        skips = []
        hidden = self.input_block(
            torch.cat((state, resize_to(condition, state), ocean_mask), dim=1),
            embedding,
        )
        if fusion is not None and lag_features is not None:
            hidden = fusion[0](hidden, lag_features[0])
        skips.append(hidden)
        for index, block in enumerate(self.down):
            pooled = F.avg_pool2d(hidden, 2)
            hidden = block(
                torch.cat((pooled, resize_to(condition, pooled)), dim=1), embedding
            )
            if fusion is not None and lag_features is not None:
                hidden = fusion[index + 1](hidden, lag_features[index + 1])
            skips.append(hidden)
        return hidden, skips

    def decode(
        self, hidden: torch.Tensor, skips: list[torch.Tensor], embedding: torch.Tensor
    ) -> torch.Tensor:
        for index, block in enumerate(self.up):
            skip = skips[self.levels - 2 - index]
            hidden = block(torch.cat((resize_to(hidden, skip), skip), dim=1), embedding)
        return self.output(F.silu(self.output_norm(hidden)))

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        ocean_mask: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        self._check(state, condition, ocean_mask)
        embedding = self.time(flow_time)
        hidden, skips = self.encode(state, condition, ocean_mask, embedding)
        pooled = F.avg_pool2d(hidden, 2)
        hidden = self.middle1(
            torch.cat((pooled, resize_to(condition, pooled)), dim=1), embedding
        )
        hidden = self.attention(hidden)
        hidden = self.middle2(hidden, embedding)
        return self.decode(hidden, skips, embedding)


class AutoregressiveSuperResolutionFlowUNet(SuperResolutionFlowUNet):
    """Super-resolution velocity field conditioned on the previous day."""

    def __init__(
        self,
        base_channels: int = 32,
        levels: int = 4,
        condition_channels: int = 2,
        target_channels: int = 1,
        attention: bool = True,
        attention_heads: int = 4,
        lag_base_channels: int = 16,
        lag_dropout: float = 0.10,
        lag_path_dropout: float = 0.10,
        lag_guidance_scale: float = 0.25,
    ):
        super().__init__(
            base_channels=base_channels,
            levels=levels,
            condition_channels=condition_channels,
            target_channels=target_channels,
            attention=attention,
            attention_heads=attention_heads,
        )
        self.lag_encoder = LagEncoder(
            lag_base_channels,
            lag_dropout,
            lag_path_dropout,
            input_channels=target_channels + 1,
            levels=levels,
        )
        self.fusion = nn.ModuleList(
            [
                GatedFiLM(
                    self.lag_encoder.channels[level],
                    self.channels[level],
                    lag_guidance_scale,
                )
                for level in range(levels)
            ]
        )

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        ocean_mask: torch.Tensor,
        previous_state: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        self._check(state, condition, ocean_mask)
        if previous_state.shape != state.shape:
            raise ValueError(
                f"Previous state {tuple(previous_state.shape)} does not match "
                f"state {tuple(state.shape)}"
            )
        embedding = self.time(flow_time)
        # The current coarse field is authoritative.  Remove the previous
        # day's block means so the lag path can guide fronts and texture but
        # cannot carry yesterday's large-scale SST forward by itself.
        lag_anomaly = within_block_anomaly(
            previous_state, ocean_mask, condition.shape[-2:]
        )
        lag_features = self.lag_encoder(
            torch.cat((lag_anomaly, ocean_mask.expand_as(previous_state)), dim=1)
        )
        hidden, skips = self.encode(
            state,
            condition,
            ocean_mask,
            embedding,
            lag_features=lag_features,
            fusion=self.fusion,
        )
        pooled = F.avg_pool2d(hidden, 2)
        hidden = self.middle1(
            torch.cat((pooled, resize_to(condition, pooled)), dim=1), embedding
        )
        hidden = self.attention(hidden)
        hidden = self.middle2(hidden, embedding)
        return self.decode(hidden, skips, embedding)


def build_model(config: dict) -> SuperResolutionFlowUNet:
    """Instantiate the model described by a training config."""
    kind = config.get("model_kind", "super_resolution")
    common = {
        "base_channels": int(config.get("base_channels", 32)),
        "levels": int(config.get("levels", 4)),
        "condition_channels": int(config.get("condition_channels", 2)),
        "target_channels": int(config.get("target_channels", 1)),
        "attention": bool(config.get("attention", True)),
        "attention_heads": int(config.get("attention_heads", 4)),
    }
    if kind == "super_resolution":
        return SuperResolutionFlowUNet(**common)
    if kind == "autoregressive":
        return AutoregressiveSuperResolutionFlowUNet(
            **common,
            lag_base_channels=int(config.get("lag_base_channels", 16)),
            lag_dropout=float(config.get("lag_dropout", 0.10)),
            lag_path_dropout=float(config.get("lag_path_dropout", 0.10)),
            lag_guidance_scale=float(config.get("lag_guidance_scale", 0.25)),
        )
    raise ValueError(f"Unknown model_kind {kind!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
