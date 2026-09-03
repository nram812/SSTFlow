"""Plot matched SRDN pilot predictions for a quick visual audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np


def read_sample(path: Path, index: int):
    with netCDF4.Dataset(path, "r") as dataset:
        dates = netCDF4.num2date(
            dataset.variables["time"][:],
            dataset.variables["time"].units,
            calendar=dataset.variables["time"].calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        return {
            "date": str(dates[index])[:10],
            "target": np.asarray(dataset.variables["sst_target"][index]),
            "bilinear": np.asarray(dataset.variables["sst_bilinear"][index]),
            "prediction": np.asarray(dataset.variables["sst_generated"][index]),
            "lat": np.asarray(dataset.variables["lat"][:]),
            "lon": np.asarray(dataset.variables["lon"][:]),
        }


def plot(srdcnn_path: Path, resafno_path: Path, output: Path, index: int = 0):
    cnn = read_sample(srdcnn_path, index)
    afno = read_sample(resafno_path, index)
    if cnn["date"] != afno["date"]:
        raise ValueError("sample NetCDF files do not contain matched dates")
    target = cnn["target"]
    fields = [target, cnn["bilinear"], cnn["prediction"], afno["prediction"]]
    names = ["Target", "Bilinear", "SRDCNN", "ResAFNO"]
    ocean = np.isfinite(target)
    values = np.concatenate([field[ocean] for field in fields])
    vmin, vmax = np.quantile(values, [0.01, 0.99])
    errors = [field - target for field in fields[1:]]
    error_values = np.concatenate([error[ocean] for error in errors])
    error_limit = max(float(np.quantile(np.abs(error_values), 0.99)), 0.05)

    figure, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    extent = [cnn["lon"].min(), cnn["lon"].max(), cnn["lat"].min(), cnn["lat"].max()]
    for axis, field, name in zip(axes[0], fields, names):
        image = axis.imshow(
            np.ma.masked_invalid(field), origin="lower", extent=extent,
            cmap="turbo", vmin=vmin, vmax=vmax,
        )
        axis.set_title(name)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
    figure.colorbar(image, ax=axes[0].tolist(), label="SST (°C)", shrink=0.8)

    error_names = ["Bilinear − target", "SRDCNN − target", "ResAFNO − target", "ResAFNO − SRDCNN"]
    error_fields = errors + [afno["prediction"] - cnn["prediction"]]
    for axis, field, name in zip(axes[1], error_fields, error_names):
        image = axis.imshow(
            np.ma.masked_invalid(field), origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-error_limit, vmax=error_limit,
        )
        axis.set_title(name)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
    figure.colorbar(image, ax=axes[1].tolist(), label="difference (°C)", shrink=0.8)
    figure.suptitle(f"SRDN 10k pilot example: {cnn['date']} (held-out test period)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srdcnn", type=Path, required=True)
    parser.add_argument("--resafno", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    plot(args.srdcnn, args.resafno, args.output, args.index)


if __name__ == "__main__":
    main()
