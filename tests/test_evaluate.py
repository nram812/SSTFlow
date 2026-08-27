"""Offline evaluation selection tests."""

import numpy as np

from evaluate import evaluation_indices


class Dataset:
    def __len__(self):
        return 11


def test_all_samples_selects_every_item_once_in_order():
    indices = evaluation_indices(Dataset(), None)
    np.testing.assert_array_equal(indices, np.arange(11))


def test_subset_remains_deterministic_and_includes_endpoints():
    indices = evaluation_indices(Dataset(), 4)
    np.testing.assert_array_equal(indices, [0, 3, 6, 10])
