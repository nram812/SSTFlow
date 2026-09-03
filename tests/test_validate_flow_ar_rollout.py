import numpy as np

from validate_flow_ar_rollout import physical_block_means


def test_physical_block_means_use_ocean_cells_only():
    field = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    mask = np.ones((4, 4), bool)
    mask[0, 0] = False
    field[:, 0, 0] = np.nan
    means = physical_block_means(field, mask, factor=2)
    assert means.shape == (1, 2, 2)
    assert means[0, 0, 0] == np.mean([1.0, 4.0, 5.0])
