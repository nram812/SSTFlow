"""Shared pytest fixtures.

The heavyweight fixture builds a *miniature replica* of the real problem: a
NETCDF3_CLASSIC file with int16 packing, a fill value, a static land blob, and a
seasonal signal.  Running the production ``preprocess.run`` on it means the
dataset, training, and callback tests exercise exactly the code that will run on
the 6.9 GB file, at a size that finishes in seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GRID = 64
COARSEN_FACTOR = 8
DAYS = 40
SCALE_FACTOR = 0.001678518
ADD_OFFSET = 45.0
FILL_VALUE = -32768


def synthetic_fields(days: int = DAYS, grid: int = GRID):
    """Smooth, seasonally varying SST with a fixed land blob (NaN)."""
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, grid), np.linspace(-1.0, 1.0, grid), indexing="ij"
    )
    land = ((x - 0.35) ** 2 + (y - 0.2) ** 2) < 0.09
    land |= (y > 0.75) & (x < -0.4)
    fields = np.empty((days, grid, grid), dtype=np.float32)
    for day in range(days):
        phase = 2.0 * np.pi * day / 30.0
        field = (
            20.0
            - 8.0 * y
            + 1.5 * np.sin(3.0 * np.pi * x + phase)
            + 0.8 * np.cos(5.0 * np.pi * y - 0.5 * phase)
        )
        fields[day] = field.astype(np.float32)
    fields[:, land] = np.nan
    return fields, ~land


def write_source(path: Path, fields: np.ndarray) -> None:
    """Write an int16-packed NETCDF3_CLASSIC file shaped like the real data."""
    import netCDF4

    days, grid, _ = fields.shape
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as dataset:
        dataset.createDimension("Time", days)
        dataset.createDimension("st_ocean", 1)
        dataset.createDimension("yt_ocean", grid)
        dataset.createDimension("xt_ocean", grid)

        time = dataset.createVariable("Time", "f8", ("Time",))
        time.units = "days since 1979-01-01 12:00:00"
        time.calendar = "standard"
        time[:] = np.arange(days, dtype=np.float64)

        depth = dataset.createVariable("st_ocean", "f8", ("st_ocean",))
        depth[:] = [2.5]

        latitude = dataset.createVariable("yt_ocean", "f8", ("yt_ocean",))
        latitude[:] = -52.95 + 0.1 * np.arange(grid)
        longitude = dataset.createVariable("xt_ocean", "f8", ("xt_ocean",))
        longitude[:] = 107.35 + 0.1 * np.arange(grid)

        variable = dataset.createVariable(
            "temp",
            "i2",
            ("Time", "st_ocean", "yt_ocean", "xt_ocean"),
            fill_value=FILL_VALUE,
        )
        variable.units = "degrees C"
        variable.standard_name = "sea_water_potential_temperature"
        variable.scale_factor = SCALE_FACTOR
        variable.add_offset = ADD_OFFSET
        # netCDF4 applies the packing and the fill value itself, so hand it
        # physical values with land masked out.
        variable[:] = np.ma.masked_invalid(fields[:, None].astype(np.float64))


def make_config(root: Path, name: str = "test_flow", **overrides) -> dict:
    config = {
        "name": name,
        "model_kind": "super_resolution",
        "source_path": str(root / "source.nc"),
        "derived_path": str(root / "derived.nc"),
        "normalization_cache": str(root / "normalization.json"),
        "output_dir": str(root / "runs" / name),
        "smoke_output_dir": str(root / "runs" / "smoke" / name),
        "coarsen_factor": COARSEN_FACTOR,
        "min_valid_fraction": 0.5,
        "train_date_ranges": [["1979-01-01", "1979-01-24"]],
        "validation_date_ranges": [["1979-01-25", "1979-01-31"]],
        "test_date_ranges": [["1979-02-01", "1979-02-09"]],
        "smoke_date_ranges": [["1979-01-01", "1979-01-10"]],
        "base_channels": 8,
        "levels": 3,
        "condition_channels": 2,
        "target_channels": 1,
        "attention": True,
        "attention_heads": 2,
        "batch_size": 2,
        "num_workers": 0,
        "preload": False,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-5,
        "ema_decay": 0.9,
        "warmup_steps": 2,
        "min_learning_rate_factor": 0.1,
        "gradient_clip": 1.0,
        "seed": 7,
        "max_steps": 4,
        "max_runtime_hours": 0.25,
        "log_every": 1,
        "validation_every": 0,
        "preview_every": 0,
        "netcdf_every": 0,
        "checkpoint_every": 0,
        "sampler": "heun",
        "validation_sampler_steps": 2,
        "preview_sampler_steps": 2,
        "validation_samples": 2,
        "preview_samples": 2,
        "lambda_conservation": 0.0,
    }
    config.update(overrides)
    return config


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """Build the miniature dataset once and run real preprocessing on it."""
    import preprocess

    root = tmp_path_factory.mktemp("sst")
    fields, ocean = synthetic_fields()
    write_source(root / "source.nc", fields)
    config = make_config(root)
    preprocess.run(config, chunk=16, probe_days=8)
    (root / "config.json").write_text(json.dumps(config, indent=2))
    np.save(root / "fields.npy", fields)
    np.save(root / "ocean.npy", ocean)
    return root


@pytest.fixture
def config(synthetic_root: Path) -> dict:
    return json.loads((synthetic_root / "config.json").read_text())


@pytest.fixture
def normalization(synthetic_root: Path) -> dict:
    return json.loads((synthetic_root / "normalization.json").read_text())


@pytest.fixture
def truth(synthetic_root: Path):
    return (
        np.load(synthetic_root / "fields.npy"),
        np.load(synthetic_root / "ocean.npy"),
    )


@pytest.fixture
def derived(config):
    from data import DerivedProduct

    return DerivedProduct(config["derived_path"])
