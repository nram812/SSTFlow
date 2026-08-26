"""Rectified-flow objectives and ODE samplers for masked super-resolution.

The interpolant is the standard rectified flow / conditional flow matching path

``z(t) = (1 - t) * noise + t * target``,   target velocity ``u = target - noise``

with ``t ~ U(0, 1)``.  Two things differ from a generic implementation:

1.  Noise, state, and velocity are all multiplied by the ocean mask, so land
    pixels stay exactly zero along the whole trajectory and contribute no
    gradient.  Combined with the masked MSE this makes the objective a pure
    ocean problem.
2.  The autoregressive variant threads the previous high-resolution day through
    every model call, which lets the same samplers serve both model kinds.
"""

from __future__ import annotations

import torch

from losses import apply_mask, masked_mse

#: Samplers exposed to configs and the command line.
SAMPLERS = ("euler", "heun", "ab2")


def masked_noise(
    reference: torch.Tensor, mask: torch.Tensor, generator=None
) -> torch.Tensor:
    """Standard normal noise that is exactly zero over land."""
    noise = torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    return apply_mask(noise, mask)


def _call(model, state, condition, mask, flow_time, previous_state=None):
    if previous_state is None:
        return model(state, condition, mask, flow_time)
    return model(state, condition, mask, previous_state, flow_time)


def flow_matching_loss(
    model,
    target: torch.Tensor,
    condition: torch.Tensor,
    mask: torch.Tensor,
    previous_state: torch.Tensor | None = None,
    generator=None,
) -> torch.Tensor:
    """Masked conditional rectified-flow velocity loss.

    ``target`` is the normalised high-resolution field with land already zeroed;
    ``mask`` is one over ocean.  The returned scalar is the mean squared
    velocity error over ocean pixels only.
    """
    batch = target.shape[0]
    flow_time = torch.rand(
        batch, dtype=target.dtype, device=target.device, generator=generator
    )
    noise = masked_noise(target, mask, generator)
    weight = flow_time[:, None, None, None]
    state = (1.0 - weight) * noise + weight * target
    state = apply_mask(state, mask)
    wanted_velocity = target - noise
    predicted_velocity = _call(
        model, state, condition, mask, flow_time, previous_state
    )
    return masked_mse(predicted_velocity, wanted_velocity, mask)


def _time_tensor(value: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.full(
        (reference.shape[0],),
        value,
        dtype=reference.dtype,
        device=reference.device,
    )


def euler_sample(
    model,
    initial_noise: torch.Tensor,
    condition: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
    previous_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """First-order sampler: one model evaluation per step."""
    if steps < 1:
        raise ValueError("The sampler needs at least one step")
    state = apply_mask(initial_noise, mask)
    dt = 1.0 / steps
    for index in range(steps):
        velocity = _call(
            model,
            state,
            condition,
            mask,
            _time_tensor(index * dt, state),
            previous_state,
        )
        state = apply_mask(state + dt * velocity, mask)
    return state


def heun_sample(
    model,
    initial_noise: torch.Tensor,
    condition: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
    previous_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Second-order predictor/corrector sampler; two evaluations per step.

    Deliberately *not* decorated with ``no_grad`` so the same function can be
    reused inside a differentiable rollout.
    """
    if steps < 1:
        raise ValueError("The sampler needs at least one step")
    state = apply_mask(initial_noise, mask)
    dt = 1.0 / steps
    for index in range(steps):
        now = index * dt
        following = min((index + 1) * dt, 1.0)
        velocity = _call(
            model, state, condition, mask, _time_tensor(now, state), previous_state
        )
        predicted = apply_mask(state + dt * velocity, mask)
        endpoint_velocity = _call(
            model,
            predicted,
            condition,
            mask,
            _time_tensor(following, state),
            previous_state,
        )
        state = apply_mask(state + 0.5 * dt * (velocity + endpoint_velocity), mask)
    return state


def ab2_sample(
    model,
    initial_noise: torch.Tensor,
    condition: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
    previous_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adams-Bashforth 2 sampler: second-order with one evaluation per step."""
    if steps < 1:
        raise ValueError("The sampler needs at least one step")
    state = apply_mask(initial_noise, mask)
    dt = 1.0 / steps
    previous_velocity = None
    for index in range(steps):
        velocity = _call(
            model,
            state,
            condition,
            mask,
            _time_tensor(index * dt, state),
            previous_state,
        )
        if previous_velocity is None:
            state = state + dt * velocity
        else:
            state = state + dt * (1.5 * velocity - 0.5 * previous_velocity)
        state = apply_mask(state, mask)
        previous_velocity = velocity
    return state


def get_sampler(name: str):
    samplers = {"euler": euler_sample, "heun": heun_sample, "ab2": ab2_sample}
    if name not in samplers:
        raise ValueError(f"Unknown sampler {name!r}; choose from {SAMPLERS}")
    return samplers[name]


def sample(
    model,
    condition: torch.Tensor,
    mask: torch.Tensor,
    shape: tuple[int, ...],
    steps: int = 25,
    sampler: str = "heun",
    previous_state: torch.Tensor | None = None,
    generator=None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Draw one masked sample for each element of ``condition``."""
    device = device or condition.device
    reference = torch.zeros(shape, device=device, dtype=dtype)
    noise = masked_noise(reference, mask, generator)
    return get_sampler(sampler)(
        model, noise, condition, mask, steps, previous_state
    )


@torch.no_grad()
def rollout(
    model,
    initial_state: torch.Tensor,
    conditions: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 25,
    sampler: str = "heun",
    generator=None,
) -> torch.Tensor:
    """Free-running autoregressive rollout.

    ``conditions`` has shape ``(batch, lead, condition_channels, h, w)``.  Each
    lead is generated from the *previous generated* day, so errors compound
    exactly as they would at inference time.  Returns ``(batch, lead, 1, H, W)``.
    """
    if conditions.dim() != 5:
        raise ValueError(
            f"Expected (batch, lead, channel, lat, lon) conditions, got "
            f"{tuple(conditions.shape)}"
        )
    predictions = []
    state = apply_mask(initial_state, mask)
    for lead in range(conditions.shape[1]):
        noise = masked_noise(state, mask, generator)
        state = get_sampler(sampler)(
            model, noise, conditions[:, lead], mask, steps, state
        )
        predictions.append(state.clone())
    return torch.stack(predictions, dim=1)


def single_step_rollout_loss(
    model,
    previous_state: torch.Tensor,
    condition: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 4,
    generator=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable single-step rollout: sample one day, score it in space.

    Returns ``(loss, prediction)``.  The sampler runs with a small number of
    steps so the unrolled graph stays affordable; this is the "single step
    rollout" fine-tuning objective that sharpens the model in *state* space
    rather than velocity space.
    """
    noise = masked_noise(target, mask, generator)
    prediction = heun_sample(
        model, noise, condition, mask, steps, previous_state
    )
    return masked_mse(prediction, target, mask), prediction
