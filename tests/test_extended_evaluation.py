"""Focused checks for the streamed post-production analysis."""

import importlib.util
from pathlib import Path

import netCDF4
import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis/generate_extended_evaluation.py"
SPEC = importlib.util.spec_from_file_location("generate_extended_evaluation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
_index_for_dates = MODULE._index_for_dates
stream_pair = MODULE.stream_pair


def product(path):
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 3)
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 2)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 2001-01-01"
        time[:] = np.arange(3)
        dataset.createVariable("lat", "f8", ("lat",))[:] = [-32.0, -22.0]
        dataset.createVariable("lon", "f8", ("lon",))[:] = [113.0, 154.0]
        target = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        generated = target + 1.0
        dataset.createVariable("sst_target", "f4", ("time", "lat", "lon"), fill_value=np.nan)[:] = target
        dataset.createVariable("sst_generated", "f4", ("time", "lat", "lon"), fill_value=np.nan)[:] = generated


def test_stream_pair_metrics_and_date_alignment(tmp_path):
    path = tmp_path / "small.nc"
    product(path)
    result = stream_pair(path, chunk=2)
    assert result["metrics"]["days"] == 3
    assert result["metrics"]["rmse_c"] == 1.0
    assert result["metrics"]["mae_c"] == 1.0
    assert result["metrics"]["bias_c"] == 1.0
    assert result["metrics"]["climatology_rmse_c"] == 1.0
    assert result["metrics"]["evolution_ratio"] == 1.0
    assert result["metrics"]["point_correlations"]["EAC"] == 1.0
    with netCDF4.Dataset(path) as dataset:
        indices = _index_for_dates(
            dataset,
            np.asarray([np.datetime64("2001-01-02"), np.datetime64("2001-01-03")]),
        )
    np.testing.assert_array_equal(indices, [1, 2])
