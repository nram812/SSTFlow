"""Tests for inference from an already-converted ACCESS-CM2 predictor."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from infer_access_cm2 import (
    _make_gan_noise,
    make_condition,
    select_time_indices,
    validate_converted_grid,
    validate_converted_values,
)


def converted_dataset():
    times = np.arange("2000-01-01", "2000-01-06", dtype="datetime64[D]")
    lat = np.array([-2.0, 0.0])
    lon = np.array([10.0, 12.0, 14.0])
    values = np.arange(30, dtype=np.float32).reshape(5, 2, 3)
    values[:, 0, 0] = np.nan
    return xr.Dataset(
        {"sst_lr": (("time", "lat_lr", "lon_lr"), values)},
        coords={"time": times, "lat_lr": lat, "lon_lr": lon},
    )


def test_time_selection_exact_dates_and_range():
    times = converted_dataset().time.values
    np.testing.assert_array_equal(
        select_time_indices(times, ["2000-01-02", "2000-01-05"]), [1, 4]
    )
    np.testing.assert_array_equal(
        select_time_indices(times, start="2000-01-02", end="2000-01-04"),
        [1, 2, 3],
    )


def test_converted_grid_must_match_training_coordinates():
    dataset = converted_dataset()
    derived = SimpleNamespace(
        lat_lr=dataset.lat_lr.values.copy(), lon_lr=dataset.lon_lr.values.copy()
    )
    assert validate_converted_grid(dataset, derived, "sst_lr").shape == (5, 2, 3)
    derived.lon_lr[1] += 0.1
    with pytest.raises(ValueError, match="not the training grid"):
        validate_converted_grid(dataset, derived, "sst_lr")


def test_converted_mask_must_match_training_mask_exactly():
    values = converted_dataset().sst_lr.values[:2]
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    validated = validate_converted_values(values, mask)
    assert validated.dtype == np.float32
    broken = values.copy(); broken[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="missing_ocean=1"):
        validate_converted_values(broken, mask)


def test_condition_normalizes_then_zero_fills_and_appends_mask():
    values = converted_dataset().sst_lr.values[:2]
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    condition = make_condition(values, mask, mean=10.0, std=2.0)
    assert condition.shape == (2, 2, 2, 3)
    assert np.isfinite(condition).all()
    assert np.all(condition[:, 0, 0, 0] == 0)
    assert np.all(condition[:, 1] == mask)
    np.testing.assert_allclose(condition[:, 0, 1, 1], (values[:, 1, 1] - 10) / 2)


def test_gan_noise_is_batch_size_independent():
    seeds = np.asarray([10, 20, 30])
    together = _make_gan_noise((3, 2, 4, 5), "cpu", torch.float32, seeds)
    separately = torch.cat([
        _make_gan_noise((1, 2, 4, 5), "cpu", torch.float32, [seed])
        for seed in seeds
    ])
    torch.testing.assert_close(together, separately)
