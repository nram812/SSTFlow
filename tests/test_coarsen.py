"""C1: mask-aware coarsening correctness."""

from __future__ import annotations

import numpy as np
import pytest

from coarsen import (
    block_valid_counts,
    coarse_coordinates,
    coarse_ocean_mask,
    coarsen,
    upsample_nearest,
)


def test_all_ocean_matches_plain_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 8, 8)).astype(np.float32)
    mask = np.ones((8, 8), dtype=bool)
    expected = values.reshape(3, 4, 2, 4, 2).mean(axis=(2, 4))
    np.testing.assert_allclose(coarsen(values, mask, 2), expected, rtol=1e-6)


def test_nan_does_not_propagate():
    values = np.ones((4, 4), dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    values[0, 0] = np.nan
    result = coarsen(values, mask, 2)
    assert np.isfinite(result).all()
    assert result[0, 0] == pytest.approx(1.0)


def test_partial_block_uses_ocean_only():
    values = np.array(
        [[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = mask[0, 1] = mask[1, 0] = True  # three of four ocean cells
    values[~mask] = np.nan
    result = coarsen(values, mask, 2, min_valid_fraction=0.5)
    assert result[0, 0] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)


def test_fifty_percent_rule_boundary():
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = mask[0, 1] = True  # exactly 50 percent
    assert bool(coarse_ocean_mask(mask, 2, 0.5)[0, 0]) is True
    mask[0, 1] = False  # 25 percent
    assert bool(coarse_ocean_mask(mask, 2, 0.5)[0, 0]) is False


def test_all_land_block_is_invalid_and_filled():
    mask = np.zeros((4, 4), dtype=bool)
    mask[2:, 2:] = True
    values = np.full((4, 4), np.nan, dtype=np.float32)
    values[2:, 2:] = 5.0
    valid = coarse_ocean_mask(mask, 2)
    assert not valid[0, 0] and valid[1, 1]
    result = coarsen(values, mask, 2, fill_value=-999.0)
    assert result[0, 0] == pytest.approx(-999.0)
    assert result[1, 1] == pytest.approx(5.0)


def test_non_divisible_grid_raises():
    with pytest.raises(ValueError, match="not divisible"):
        coarsen(np.zeros((5, 5)), np.ones((5, 5), dtype=bool), 2)


def test_batch_and_single_agree():
    rng = np.random.default_rng(1)
    mask = rng.random((8, 8)) > 0.3
    values = rng.normal(size=(8, 8)).astype(np.float32)
    values[~mask] = np.nan
    single = coarsen(values, mask, 4)
    batched = coarsen(values[None], mask, 4)[0]
    np.testing.assert_allclose(single, batched, equal_nan=True)


def test_coarse_coordinates():
    coordinate = np.arange(8, dtype=np.float64)
    result = coarse_coordinates(coordinate, 4)
    np.testing.assert_allclose(result, [1.5, 5.5])
    with pytest.raises(ValueError):
        coarse_coordinates(coordinate, 3)


def test_block_valid_counts():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = mask[1, 1] = True
    counts = block_valid_counts(mask, 2)
    assert counts[0, 0] == 2 and counts[1, 1] == 0


def test_upsample_nearest_roundtrip():
    values = np.arange(4, dtype=np.float32).reshape(2, 2)
    result = upsample_nearest(values, 3)
    assert result.shape == (6, 6)
    assert result[0, 0] == 0.0 and result[5, 5] == 3.0


def test_output_is_finite_where_valid():
    rng = np.random.default_rng(2)
    mask = rng.random((32, 32)) > 0.4
    values = rng.normal(size=(5, 32, 32)).astype(np.float32)
    values[:, ~mask] = np.nan
    valid = coarse_ocean_mask(mask, 4)
    result = coarsen(values, mask, 4)
    assert np.isfinite(result[:, valid]).all()
