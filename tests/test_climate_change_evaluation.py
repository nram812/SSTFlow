"""Focused numerical tests for publication climate-change diagnostics."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis/generate_climate_change_evaluation.py"
SPEC = importlib.util.spec_from_file_location("generate_climate_change_evaluation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_coarsen_ocean_mean_uses_only_valid_ocean_cells():
    field = np.arange(16, dtype=np.float64).reshape(4, 4)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    result = MODULE.coarsen_ocean_mean(field, mask, (2, 2))
    expected = np.asarray([
        [(1.0 + 4.0 + 5.0) / 3.0, (2.0 + 3.0 + 6.0 + 7.0) / 4.0],
        [(8.0 + 9.0 + 12.0 + 13.0) / 4.0, (10.0 + 11.0 + 14.0 + 15.0) / 4.0],
    ])
    np.testing.assert_allclose(result, expected)


def test_area_weighted_signal_metrics_are_exact_for_scaled_signal():
    target = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    prediction = target * 0.8
    mask = np.ones_like(target, dtype=bool)
    result = MODULE.field_metrics(prediction, target, np.asarray([-40.0, -20.0]), mask)
    assert np.isclose(result["mean_signal_ratio"], 0.8)
    assert np.isclose(result["pattern_std_ratio"], 0.8)
    assert np.isclose(result["spatial_correlation"], 1.0)
    assert result["mean_bias_c"] < 0


def test_expand_coarse_round_trip_shape_and_values():
    coarse = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    expanded = MODULE.expand_coarse(coarse, (4, 6))
    assert expanded.shape == (4, 6)
    np.testing.assert_allclose(expanded[:2, :3], 1.0)
    np.testing.assert_allclose(expanded[2:, 3:], 4.0)
