import random

import numpy as np
import pytest
import torch

from common import (attach_ocean_mask, atomic_json, atomic_torch_save, consecutive_pair_starts,
                    contiguous_runs, mask_sha256, normalized_to_physical,
                    restore_rng_state, rng_state_payload, selected_indices)


def test_selected_indices_ranges():
    times = np.arange("2000-01-01", "2000-01-08", dtype="datetime64[D]")
    assert selected_indices(times, [["2000-01-02", "2000-01-03"], ["2000-01-06", "2000-01-06"]]).tolist() == [1, 2, 5]


def test_selected_indices_empty_raises():
    with pytest.raises(ValueError, match="No dates"):
        selected_indices(np.arange("2000-01-01", "2000-01-03", dtype="datetime64[D]"), [["2001-01-01", "2001-01-02"]])


def test_contiguous_runs():
    assert list(contiguous_runs(np.array([1, 2, 5, 6, 7]))) == [(1, 3, 0), (5, 8, 2)]


def test_consecutive_pair_starts_excludes_boundary_and_detects_gap():
    times = np.arange("2000-01-01", "2000-01-07", dtype="datetime64[D]")
    assert consecutive_pair_starts(times, [["2000-01-01", "2000-01-03"], ["2000-01-05", "2000-01-06"]]).tolist() == [0, 1, 4]
    with pytest.raises(ValueError, match="Non-consecutive"):
        consecutive_pair_starts(np.array(["2000-01-01", "2000-01-03"], dtype="datetime64[D]"), [["2000-01-01", "2000-01-03"]])


def test_mask_sha256_sensitivity():
    mask = np.ones((2, 2), bool); lat = np.arange(2); lon = np.arange(2)
    digest = mask_sha256(mask, lat, lon)
    mask[0, 0] = False
    assert digest != mask_sha256(mask, lat, lon)
    assert digest != mask_sha256(np.ones((2, 2)), lat + 1, lon)


def test_atomic_json_and_torch_save(tmp_path):
    json_path = tmp_path / "value.json"; tensor_path = tmp_path / "value.pt"
    atomic_json(json_path, {"value": 3}); atomic_torch_save({"value": torch.tensor(4)}, tensor_path)
    assert json_path.read_text().find('"value": 3') >= 0
    assert torch.load(tensor_path, weights_only=True)["value"].item() == 4
    assert not list(tmp_path.glob("*.partial"))


def test_rng_state_roundtrip():
    random.seed(4); np.random.seed(4); torch.manual_seed(4)
    state = rng_state_payload(); expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng_state(state); actual = (random.random(), np.random.rand(), torch.rand(1))
    assert expected[:2] == actual[:2]; torch.testing.assert_close(expected[2], actual[2])


def test_normalized_to_physical_restores_nan():
    values = np.array([[0.0, 1.0], [2.0, 3.0]], np.float32); mask = np.array([[1, 0], [1, 1]], bool)
    tensor = torch.from_numpy(values)[None, None]
    result = normalized_to_physical(tensor, attach_ocean_mask({"sst_mean": 10.0, "sst_std": 2.0}, mask))[0, 0].numpy()
    assert np.isnan(result[0, 1]); np.testing.assert_allclose(result[mask], [10, 14, 16])
