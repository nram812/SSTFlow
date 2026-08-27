#!/usr/bin/env python3
"""Convert ACCESS-CM2 SST to the trained 32 x 32 predictor grid.

The conversion preserves ACCESS-CM2 seasonal anomalies relative to the OFAM
training period and adds them to the OFAM seasonal coarse-grid climatology.
Missing coastal anomaly cells are filled linearly and then, outside the convex
hull, by nearest neighbour. The established training coarse mask is restored
before an atomic NetCDF write.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
import xarray as xr


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCESS = Path(
    "/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/"
    "Autoregressive_Model/data/global_sst/"
    "ACCESS-CM2_1960-2099_sst_global_2deg_raw.nc"
)
DEFAULT_TRAINING = REPOSITORY_ROOT / "derived" / "sst_downscaling_f16.nc"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "derived" / "sst_downscaling_access_converted.nc"


def fill_missing_2d(values: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation, then nearest outside the hull."""
    values = np.asarray(values)
    ny, nx = values.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    valid = np.isfinite(values)
    if valid.all() or not valid.any():
        return values
    points = np.column_stack((yy[valid], xx[valid]))
    grid_points = np.column_stack((yy.ravel(), xx.ravel()))
    filled = griddata(
        points, values[valid], grid_points, method="linear"
    ).reshape(ny, nx)
    outside = ~np.isfinite(filled)
    if outside.any():
        nearest = griddata(
            points, values[valid], grid_points, method="nearest"
        ).reshape(ny, nx)
        filled[outside] = nearest[outside]
    return filled


def convert_access_to_training_grid(
    access_path: Path,
    training_path: Path,
) -> xr.Dataset:
    """Build the converted ACCESS predictor lazily as an xarray Dataset."""
    training = xr.open_dataset(training_path, chunks={"time": 365})
    access = xr.open_dataset(access_path, chunks={"time": 365})
    required_training = {"sst_lr", "ocean_mask_lr", "lat_lr", "lon_lr"}
    required_access = {"sst_raw", "lat", "lon"}
    if not required_training.issubset(training.variables):
        raise ValueError(f"Training product lacks {required_training - set(training.variables)}")
    if not required_access.issubset(access.variables):
        raise ValueError(f"ACCESS product lacks {required_access - set(access.variables)}")

    training_climatology = training.sst_lr.groupby("time.season").mean("time")
    access_on_grid = access.sst_raw.interp(
        lat=training.lat_lr,
        lon=training.lon_lr,
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )
    access_training = access_on_grid.sel(time=training.time)
    access_climatology = access_training.groupby("time.season").mean("time")
    anomalies = access_on_grid.groupby("time.season") - access_climatology
    filled_anomalies = xr.apply_ufunc(
        fill_missing_2d,
        anomalies,
        input_core_dims=[["lat_lr", "lon_lr"]],
        output_core_dims=[["lat_lr", "lon_lr"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    converted = filled_anomalies.groupby("time.season") + training_climatology
    converted = converted.where(training.ocean_mask_lr.astype(bool)).astype(np.float32)
    converted.name = "sst_lr"
    converted.attrs.update(
        units="degC",
        long_name="ACCESS-CM2 seasonal anomalies on the OFAM training coarse grid",
    )
    result = converted.to_dataset()
    result.attrs.update(
        source_file=str(access_path.resolve()),
        training_grid_file=str(training_path.resolve()),
        conversion=(
            "ACCESS seasonal anomalies relative to the OFAM training dates, "
            "added to the OFAM seasonal coarse-grid climatology"
        ),
        missing_value_fill="linear griddata followed by nearest griddata",
    )
    return result


def write_atomic(dataset: xr.Dataset, output: Path, overwrite: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    temporary = output.with_suffix(".partial.nc")
    dataset.to_netcdf(
        temporary,
        engine="h5netcdf",
        encoding={"sst_lr": {"dtype": "float32", "zlib": True, "complevel": 4}},
    )
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access", type=Path, default=DEFAULT_ACCESS)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    dataset = convert_access_to_training_grid(arguments.access, arguments.training)
    try:
        write_atomic(dataset, arguments.output, arguments.overwrite)
    finally:
        dataset.close()
    print(f"[ok] wrote {arguments.output}")


if __name__ == "__main__":
    main()
