"""Offline evaluation selection tests."""

import numpy as np
import torch

from evaluate import evaluation_indices, rollout_inputs, target_dates


class Dataset:
    def __len__(self):
        return 11


def test_all_samples_selects_every_item_once_in_order():
    indices = evaluation_indices(Dataset(), None)
    np.testing.assert_array_equal(indices, np.arange(11))


def test_subset_remains_deterministic_and_includes_endpoints():
    indices = evaluation_indices(Dataset(), 4)
    np.testing.assert_array_equal(indices, [0, 3, 6, 10])


def test_rollout_inputs_has_batch_and_lead_axes():
    items = [
        {
            "condition": torch.zeros(2, 4, 5),
            "mask": torch.ones(1, 8, 10),
            "previous": torch.zeros(1, 8, 10),
        }
        for _ in range(3)
    ]
    previous, conditions, mask = rollout_inputs(items, torch.device("cpu"))
    assert previous.shape == (1, 1, 8, 10)
    assert conditions.shape == (1, 3, 2, 4, 5)
    assert mask.shape == (1, 1, 8, 10)


def test_target_dates_uses_target_not_previous_date_for_ar_pairs():
    class ARDataset:
        starts = np.arange(2)

        def date_window(self, item):
            return [f"2011-01-0{item + 1}", f"2011-01-0{item + 2}"]

    assert target_dates(ARDataset(), [0, 1]) == ["2011-01-02", "2011-01-03"]
