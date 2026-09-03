"""NOAA grid and preprocessing tests shared by the active 1024² model."""

from __future__ import annotations

import numpy as np
import pytest

from preprocess_noaa_5km import block_mean, nested_crop_slice


def test_exact_nested_crop_centres():
    reference = np.arange(8, dtype=np.float64) * 0.1 + 107.35
    satellite = np.arange(30, dtype=np.float64) * 0.05 + 106.825
    chosen = nested_crop_slice(satellite, reference, ratio=2, atol=1e-10)
    crop = satellite[chosen]
    assert len(crop) == 16
    np.testing.assert_allclose(crop.reshape(8, 2).mean(1), reference, atol=1e-12)


def test_non_nested_crop_rejected():
    reference = np.arange(4) * 0.1
    satellite = np.arange(20) * 0.04
    with pytest.raises(ValueError, match="not nested"):
        nested_crop_slice(satellite, reference, ratio=2)


def test_noaa_block_mean_uses_only_target_ocean():
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    values[0, 0] = np.nan
    means, fractions = block_mean(values, mask, 2)
    assert means[0, 0] == pytest.approx((1 + 4 + 5) / 3)
    assert fractions[0, 0] == pytest.approx(0.75)
    assert np.isfinite(means).all()
