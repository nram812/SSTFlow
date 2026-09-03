"""Direct 1024-square NOAA flow built on a pretrained 512 trunk.

The failed first transfer experiment sampled a complete 512-square SST field,
bilinearly enlarged that *prediction*, and restricted a second flow to zero
mean in every 2x2 ocean block.  That made the 0.05-degree product inherit the
0.1-degree appearance and, where the NOAA and OFAM coastlines differ, forced
satellite-ocean pixels towards an OFAM-land value.

The first stage uses the pretrained model as a frozen feature extractor.  A
continuation stage can unfreeze its bottleneck and complete decoder while the
time embedding and encoder remain frozen.  In both stages the final 512-square
normalisation/output convolution is bypassed.  A learned pixel-shuffle block
upsamples the last decoder feature map, fuses it with the actual 1024-square
flow state, NOAA mask, and SST condition, and predicts the full high-resolution
velocity directly.  There is no block-mean projection.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import apply_mask, masked_mse
from model import build_model, group_count, resize_to


def ocean_block_mean(
    values: torch.Tensor, mask: torch.Tensor, factor: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked block means and a one-channel valid-block mask."""
    if values.ndim != 4 or mask.ndim != 4:
        raise ValueError("values and mask must be BCHW tensors")
    if values.shape[-2:] != mask.shape[-2:]:
        raise ValueError("values and mask grids differ")
    if values.shape[-2] % factor or values.shape[-1] % factor:
        raise ValueError("high-resolution grid is not divisible by factor")
    expanded = mask.expand(values.shape[0], 1, *values.shape[-2:]).to(values.dtype)
    count = F.avg_pool2d(expanded, factor) * (factor * factor)
    total = F.avg_pool2d(values * expanded, factor) * (factor * factor)
    return total / count.clamp_min(1.0), (count > 0).to(values.dtype)


def coastline_ocean_mask(mask: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Ocean pixels within ``radius`` grid cells of NOAA land."""
    if radius < 1:
        raise ValueError("coastline radius must be positive")
    land = 1.0 - mask.to(torch.float32)
    nearby_land = F.max_pool2d(land, 2 * radius + 1, stride=1, padding=radius)
    return mask.to(torch.float32) * (nearby_land > 0).to(torch.float32)


class HighResolutionBlock(nn.Module):
    """Reflected-boundary residual block with flow-time FiLM."""

    def __init__(self, input_channels: int, output_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(input_channels), input_channels)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, padding=1, padding_mode="reflect"
        )
        self.norm2 = nn.GroupNorm(group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, 3, padding=1, padding_mode="reflect"
        )
        self.time = nn.Linear(time_dim, 2 * output_channels)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, values: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        scale, shift = self.time(embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1.0 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(values)


class Learned1024Head(nn.Module):
    """Learned 512->1024 feature upsampling and full-resolution velocity head."""

    def __init__(
        self,
        trunk_channels: int,
        time_dim: int,
        width: int = 32,
        blocks: int = 3,
        condition_channels: int = 2,
    ):
        super().__init__()
        if width < 8 or blocks < 1:
            raise ValueError("head width >= 8 and at least one block are required")
        # PixelShuffle learns four distinct sub-cell feature sets.  Unlike
        # interpolation of a generated SST image, this can represent genuine
        # structure at the NOAA grid scale.
        self.learned_upsample = nn.Sequential(
            nn.Conv2d(
                trunk_channels,
                4 * width,
                3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.PixelShuffle(2),
            nn.GroupNorm(group_count(width), width),
            nn.SiLU(),
        )
        # The direct state pathway is essential: the frozen trunk sees only a
        # masked 2x2 mean, while the rectified-flow ODE evolves at 1024 square.
        direct_channels = 1 + condition_channels + 1
        self.direct = nn.Sequential(
            nn.Conv2d(
                direct_channels,
                width // 2,
                3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.GroupNorm(group_count(width // 2), width // 2),
            nn.SiLU(),
        )
        layers = [HighResolutionBlock(width + width // 2, width, time_dim)]
        layers.extend(HighResolutionBlock(width, width, time_dim) for _ in range(blocks - 1))
        self.blocks = nn.ModuleList(layers)
        self.output_norm = nn.GroupNorm(group_count(width), width)
        self.output = nn.Conv2d(width, 1, 1)
        # Small but nonzero initialisation gives the whole new head gradients on
        # update one without an unstable random velocity field.
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        trunk: torch.Tensor,
        state: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        learned = self.learned_upsample(trunk)
        if learned.shape[-2:] != state.shape[-2:]:
            raise ValueError(
                f"learned head grid {learned.shape[-2:]} != state grid {state.shape[-2:]}"
            )
        condition_high = F.interpolate(
            condition, size=state.shape[-2:], mode="bilinear", align_corners=False
        )
        direct = self.direct(torch.cat((state, condition_high, target_mask), dim=1))
        hidden = torch.cat((learned, direct), dim=1)
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return apply_mask(self.output(F.silu(self.output_norm(hidden))), target_mask)


class NOAAFrozenTrunkFlow(nn.Module):
    """Pretrained 512 features plus a trainable 1024-square head.

    ``head_only`` reproduces the original frozen-trunk stage.
    ``decoder_and_head`` additionally trains the bottleneck and every 512-grid
    up block.  The encoder and the bypassed legacy output layers stay frozen.
    """

    def __init__(
        self,
        base: nn.Module,
        base_mask: torch.Tensor,
        head_width: int = 32,
        head_blocks: int = 3,
        trainable_policy: str = "head_only",
    ):
        super().__init__()
        if base_mask.ndim not in (2, 4):
            raise ValueError("base mask must be HW or BCHW")
        base_height, base_width = (int(value) for value in base_mask.shape[-2:])
        self.base = base
        self.register_buffer(
            "base_mask",
            base_mask.to(torch.float32).reshape(1, 1, base_height, base_width),
        )
        self.head = Learned1024Head(
            trunk_channels=int(base.channels[0]),
            time_dim=int(base.channels[0]) * 8,
            width=head_width,
            blocks=head_blocks,
            condition_channels=int(base.condition_channels),
        )
        self.set_trainable_policy(trainable_policy)

    @classmethod
    def from_pretrained(
        cls,
        config: dict,
        base_mask: torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> "NOAAFrozenTrunkFlow":
        base = build_model(config).to(device)
        state = torch.load(
            Path(config["pretrained_ema_path"]), map_location=device, weights_only=True
        )
        missing, unexpected = base.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"pretrained flow is incompatible; missing={missing}, unexpected={unexpected}"
            )
        return cls(
            base,
            base_mask.to(device),
            head_width=int(config.get("head_width", 32)),
            head_blocks=int(config.get("head_blocks", 3)),
            trainable_policy=str(config.get("trainable_policy", "head_only")),
        )

    def set_trainable_policy(self, policy: str) -> None:
        if policy not in ("head_only", "decoder_and_head"):
            raise ValueError(
                "trainable_policy must be 'head_only' or 'decoder_and_head', "
                f"got {policy!r}"
            )
        self.trainable_policy = policy
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        if policy == "decoder_and_head":
            for module in self.decoder_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        # These layers are not called by this model and must never enter an
        # optimiser merely because the decoder is unfrozen.
        for module in (self.base.output_norm, self.base.output):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.base.eval()

    def decoder_modules(self) -> tuple[nn.Module, ...]:
        """Return the bottleneck and complete active pretrained decoder."""
        return (
            self.base.middle1,
            self.base.attention,
            self.base.middle2,
            self.base.up,
        )

    def decoder_parameters(self):
        for module in self.decoder_modules():
            yield from module.parameters()

    def encoder_parameters(self):
        modules = (self.base.time, self.base.input_block, self.base.down)
        for module in modules:
            yield from module.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        # The encoder remains a deterministic, frozen feature extractor.  Put
        # only the active decoder back in train mode for the continuation.
        self.base.eval()
        if mode and self.trainable_policy == "decoder_and_head":
            for module in self.decoder_modules():
                module.train(True)
        return self

    def _decoder_features(
        self,
        high_state: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_state, _ = ocean_block_mean(high_state, target_mask, factor=2)
        base_mask = self.base_mask.expand(high_state.shape[0], -1, -1, -1)
        low_state = apply_mask(low_state, base_mask)
        # No backward graph is retained through the encoder.  Its features and
        # skip tensors are constants for the trainable decoder continuation.
        with torch.no_grad():
            embedding = self.base.time(flow_time)
            hidden, skips = self.base.encode(low_state, condition, base_mask, embedding)
            pooled = F.avg_pool2d(hidden, 2)
            middle_input = torch.cat(
                (pooled, resize_to(condition, pooled)), dim=1
            )

        def decode(values: torch.Tensor) -> torch.Tensor:
            values = self.base.middle1(values, embedding)
            values = self.base.attention(values)
            values = self.base.middle2(values, embedding)
            for index, block in enumerate(self.base.up):
                skip = skips[self.base.levels - 2 - index]
                values = block(
                    torch.cat((resize_to(values, skip), skip), dim=1), embedding
                )
            return values

        if self.trainable_policy == "head_only":
            with torch.no_grad():
                hidden = decode(middle_input)
        else:
            hidden = decode(middle_input)
        return hidden, embedding

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        expected = tuple(2 * int(value) for value in self.base_mask.shape[-2:])
        if tuple(state.shape[-2:]) != expected:
            raise ValueError(
                f"NOAA state must be exactly 2x the frozen trunk grid {expected}, "
                f"got {state.shape[-2:]}"
            )
        if target_mask.shape[-2:] != state.shape[-2:]:
            raise ValueError("NOAA state and target mask grids differ")
        target_mask = target_mask.expand(state.shape[0], -1, -1, -1).to(state.dtype)
        trunk, embedding = self._decoder_features(
            state, condition, target_mask, flow_time
        )
        return self.head(trunk, state, condition, target_mask, embedding)


def high_resolution_flow_losses(
    model: NOAAFrozenTrunkFlow,
    target: torch.Tensor,
    condition: torch.Tensor,
    mask: torch.Tensor,
    coast_radius: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinary rectified-flow loss plus a separately reported coastal loss."""
    batch = target.shape[0]
    flow_time = torch.rand(
        batch, dtype=target.dtype, device=target.device, generator=generator
    )
    noise = torch.randn(
        target.shape,
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )
    noise = apply_mask(noise, mask)
    weight = flow_time[:, None, None, None]
    state = apply_mask((1.0 - weight) * noise + weight * target, mask)
    wanted = target - noise
    predicted = model(state, condition, mask, flow_time)
    full = masked_mse(predicted, wanted, mask)
    coast = masked_mse(predicted, wanted, coastline_ocean_mask(mask, coast_radius))
    return full, coast
