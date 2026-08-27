"""Focused tests for the standalone ACCESS conversion utility."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


SCRIPT = Path(__file__).resolve().parents[1] / "derived" / "convert_access_to_training_grid.py"
SPEC = importlib.util.spec_from_file_location("convert_access", SCRIPT)
convert_access = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert_access)


def test_fill_missing_2d_uses_linear_then_nearest():
    values = np.array(
        [[np.nan, 1.0, 2.0], [3.0, np.nan, 5.0], [6.0, 7.0, np.nan]],
        dtype=np.float32,
    )
    filled = convert_access.fill_missing_2d(values)
    assert np.isfinite(filled).all()
    np.testing.assert_allclose(filled[np.isfinite(values)], values[np.isfinite(values)])


def test_conversion_writer_is_atomic_and_protects_existing_output(tmp_path):
    dataset = xr.Dataset(
        {"sst_lr": (("time", "lat_lr", "lon_lr"), np.ones((1, 2, 2), np.float32))},
        coords={"time": [np.datetime64("2000-01-01")], "lat_lr": [0, 1], "lon_lr": [2, 3]},
    )
    output = tmp_path / "converted.nc"
    convert_access.write_atomic(dataset, output)
    assert output.is_file() and not output.with_suffix(".partial.nc").exists()
    with pytest.raises(FileExistsError):
        convert_access.write_atomic(dataset, output)
