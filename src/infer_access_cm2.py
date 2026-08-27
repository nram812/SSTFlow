#!/usr/bin/env python3
"""Downscale an already-converted ACCESS-CM2 predictor with ``flow_sr``.

This program does no spatial interpolation. Its input must already use the
32 x 32 predictor grid and static ocean mask stored in the training-derived
product. Conversion is a separate, auditable step implemented by
``derived/convert_access_to_training_grid.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import netCDF4
import numpy as np
import torch
import xarray as xr

import engine
from common import REPOSITORY_ROOT, load_json
from data import DerivedProduct
from flow import SAMPLERS, get_sampler
from model import build_model


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "access_cm2_inference.json"


def select_time_indices(
    source_times: np.ndarray,
    dates: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> np.ndarray:
    """Select exact dates or one inclusive range from a daily time axis."""
    days = np.asarray(source_times).astype("datetime64[D]")
    if dates:
        requested = np.asarray(dates, dtype="datetime64[D]")
        lookup = {value: index for index, value in enumerate(days)}
        missing = [str(value) for value in requested if value not in lookup]
        if missing:
            raise ValueError(f"Dates absent from converted ACCESS-CM2 file: {missing}")
        indices = np.asarray([lookup[value] for value in requested], dtype=np.int64)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate dates were requested")
        return np.sort(indices)
    if start is None or end is None:
        raise ValueError("Provide dates or both a start and end date")
    first, last = np.datetime64(start, "D"), np.datetime64(end, "D")
    if first > last:
        raise ValueError("A period start must not be later than its end")
    indices = np.flatnonzero((days >= first) & (days <= last))
    if not len(indices):
        raise ValueError("Requested period does not intersect the converted input")
    return indices.astype(np.int64)


def validate_converted_grid(
    dataset: xr.Dataset, derived: DerivedProduct, variable: str
) -> xr.DataArray:
    """Require the converted predictor to obey the trained-grid contract."""
    if variable not in dataset:
        raise ValueError(f"Converted ACCESS-CM2 input lacks variable {variable!r}")
    field = dataset[variable]
    if field.dims != ("time", "lat_lr", "lon_lr"):
        raise ValueError(
            f"{variable} must have dimensions (time, lat_lr, lon_lr), got {field.dims}"
        )
    if "time" not in dataset.coords:
        raise ValueError("Converted ACCESS-CM2 input has no time coordinate")
    for name, wanted in (("lat_lr", derived.lat_lr), ("lon_lr", derived.lon_lr)):
        if name not in dataset.coords:
            raise ValueError(f"Converted ACCESS-CM2 input has no {name} coordinate")
        actual = np.asarray(dataset[name].values)
        if actual.shape != wanted.shape or not np.allclose(
            actual, wanted, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"Converted ACCESS-CM2 {name} is not the training grid")
    return field


def validate_converted_values(values: np.ndarray, ocean_mask_lr: np.ndarray) -> np.ndarray:
    """Check the converted static mask and return float32 physical SST."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    expected_shape = (len(values), *ocean_mask_lr.shape)
    if values.shape != expected_shape:
        raise ValueError(f"Expected converted batch shape {expected_shape}, got {values.shape}")
    finite = np.isfinite(values)
    wanted = np.broadcast_to(np.asarray(ocean_mask_lr, dtype=bool), values.shape)
    if not np.array_equal(finite, wanted):
        missing = int(np.count_nonzero(wanted & ~finite))
        unexpected = int(np.count_nonzero(~wanted & finite))
        raise ValueError(
            "Converted ACCESS-CM2 mask differs from the training predictor mask: "
            f"missing_ocean={missing}, finite_training_land={unexpected}"
        )
    return values


def make_condition(
    values: np.ndarray, ocean_mask_lr: np.ndarray, mean: float, std: float
) -> np.ndarray:
    """Normalize first, fill missing cells with zero, and append the mask."""
    values = validate_converted_values(values, ocean_mask_lr)
    normalized = (values - float(mean)) / float(std)
    normalized = np.where(ocean_mask_lr[None], normalized, 0.0).astype(np.float32)
    mask = np.broadcast_to(ocean_mask_lr, normalized.shape).astype(np.float32)
    condition = np.stack((normalized, mask), axis=1)
    if not np.isfinite(condition).all():
        raise ValueError("Non-finite value remained after ACCESS normalization")
    return condition


def load_flow_run(run_dir: Path, device: torch.device):
    config = load_json(run_dir / "config_used.json")
    if config.get("model_kind") != "super_resolution":
        raise ValueError("Converted ACCESS inference requires the plain flow_sr model")
    normalization = load_json(run_dir / "normalization.json")
    derived = DerivedProduct(config["derived_path"])
    derived.verify(normalization)
    weights = run_dir / "model_ema.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing EMA weights: {weights}")
    model = build_model(config)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return config, normalization, derived, model.to(device).eval(), weights


def _selected_time_hash(times: np.ndarray) -> str:
    encoded = np.asarray(times).astype("datetime64[ns]").view("i8").tobytes()
    return hashlib.sha256(encoded).hexdigest()


def _make_noise(shape, device, dtype, mask, seeds) -> torch.Tensor:
    """Make batch-size-independent noise from absolute source time indices."""
    draws = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        draws.append(
            torch.randn(
                (1, *shape[1:]), device=device, dtype=dtype, generator=generator
            )
        )
    return torch.cat(draws) * mask


class OperationalWriter:
    """Resumable NetCDF writer atomically promoted only when complete."""

    def __init__(self, output: Path, times: np.ndarray, derived, attrs: dict):
        self.output = output
        self.partial = output.with_suffix(".partial.nc")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        expected_hash = _selected_time_hash(times)
        if self.partial.exists():
            self.dataset = netCDF4.Dataset(self.partial, "r+")
            if self.dataset.getncattr("selected_time_sha256") != expected_hash:
                self.dataset.close()
                raise ValueError(f"Partial output dates do not match: {self.partial}")
            for name in ("converted_input_path", "sampler", "sampler_steps"):
                if str(self.dataset.getncattr(name)) != str(attrs[name]):
                    self.dataset.close()
                    raise ValueError(f"Partial output {name} does not match this run")
            self.completed = int(self.dataset.getncattr("completed"))
            return

        self.dataset = netCDF4.Dataset(self.partial, "w", format="NETCDF4")
        ds = self.dataset
        ds.createDimension("time", len(times))
        ds.createDimension("lat", len(derived.lat))
        ds.createDimension("lon", len(derived.lon))
        ds.createDimension("lat_lr", len(derived.lat_lr))
        ds.createDimension("lon_lr", len(derived.lon_lr))
        time_variable = ds.createVariable("time", "f8", ("time",))
        time_variable.units = "days since 1850-01-01 00:00:00"
        time_variable.calendar = "proleptic_gregorian"
        time_variable[:] = (
            times.astype("datetime64[ns]") - np.datetime64("1850-01-01")
        ) / np.timedelta64(1, "D")
        ds.createVariable("lat", "f8", ("lat",))[:] = derived.lat
        ds.createVariable("lon", "f8", ("lon",))[:] = derived.lon
        ds.createVariable("lat_lr", "f8", ("lat_lr",))[:] = derived.lat_lr
        ds.createVariable("lon_lr", "f8", ("lon_lr",))[:] = derived.lon_lr
        ds.createVariable(
            "sst_downscaled", "f4", ("time", "lat", "lon"),
            zlib=True, complevel=4,
            chunksizes=(1, len(derived.lat), len(derived.lon)),
            fill_value=np.float32(np.nan),
        ).units = "degrees C"
        ds.createVariable(
            "sst_coarse", "f4", ("time", "lat_lr", "lon_lr"),
            zlib=True, complevel=4,
            chunksizes=(1, len(derived.lat_lr), len(derived.lon_lr)),
            fill_value=np.float32(np.nan),
        ).units = "degrees C"
        ds.createVariable("ocean_mask", "u1", ("lat", "lon"))[:] = derived.ocean_mask
        ds.createVariable("ocean_mask_lr", "u1", ("lat_lr", "lon_lr"))[:] = (
            derived.ocean_mask_lr
        )
        for name, value in attrs.items():
            ds.setncattr(name, value)
        ds.setncattr("selected_time_sha256", expected_hash)
        ds.setncattr("completed", 0)
        ds.sync()
        self.completed = 0

    def write(self, start: int, downscaled: np.ndarray, coarse: np.ndarray) -> None:
        stop = start + len(downscaled)
        self.dataset.variables["sst_downscaled"][start:stop] = downscaled
        self.dataset.variables["sst_coarse"][start:stop] = coarse
        self.completed = stop
        self.dataset.setncattr("completed", stop)
        self.dataset.sync()

    def finish(self, mean: float, std: float) -> None:
        expected = len(self.dataset.dimensions["time"])
        if self.completed != expected:
            raise ValueError(f"Cannot finish incomplete output: {self.completed}/{expected}")
        coarse = self.dataset.variables["sst_coarse"][:]
        valid = self.dataset.variables["ocean_mask_lr"][:].astype(bool)
        values = np.asarray(coarse[:, valid])
        self.dataset.setncattr("coarse_sst_min", float(np.min(values)))
        self.dataset.setncattr("coarse_sst_max", float(np.max(values)))
        self.dataset.setncattr(
            "coarse_fraction_outside_training_3sigma",
            float(np.mean(np.abs((values - mean) / std) > 3.0)),
        )
        self.dataset.close()
        os.replace(self.partial, self.output)

    def close(self) -> None:
        if getattr(self, "dataset", None) is not None and self.dataset.isopen():
            self.dataset.close()


def load_settings(config_path: Path, period_name: str) -> dict:
    config = load_json(config_path)
    if period_name not in config["periods"]:
        raise ValueError(f"Unknown period {period_name!r}; choose from {config['periods']}")
    result = dict(config)
    result.update(config["periods"][period_name])
    for name in ("input_path", "run_dir", "output"):
        value = Path(result[name])
        result[name] = value if value.is_absolute() else REPOSITORY_ROOT / value
    result["period_name"] = period_name
    return result


@torch.no_grad()
def run(settings: dict, device_name: str | None = None) -> Path:
    device = engine.resolve_device(device_name)
    run_dir = Path(settings["run_dir"])
    input_path = Path(settings["input_path"])
    output = Path(settings["output"])
    sampler_name = str(settings["sampler"])
    steps = int(settings["sampler_steps"])
    batch_size = int(settings["batch_size"])
    seed = int(settings["seed"])
    variable = str(settings.get("variable", "sst_lr"))
    if sampler_name not in SAMPLERS or steps < 1 or batch_size < 1:
        raise ValueError("Invalid sampler, sampler_steps, or batch_size")

    model_config, normalization, derived, model, weights = load_flow_run(
        run_dir, device
    )
    with xr.open_dataset(input_path, engine="h5netcdf") as source:
        field = validate_converted_grid(source, derived, variable)
        indices = select_time_indices(
            source.time.values, start=settings["start"], end=settings["end"]
        )
        times = source.time.values[indices]
        days = times.astype("datetime64[D]")
        expected_days = int(settings.get("expected_days", len(indices)))
        if len(indices) != expected_days:
            raise ValueError(
                f"Period {settings['period_name']} has {len(indices)} days, "
                f"expected {expected_days}"
            )
        attrs = {
            "product": "ACCESS-CM2 SST downscaled by flow_sr",
            "period_name": settings["period_name"],
            "converted_input_path": str(input_path.resolve()),
            "converted_input_variable": variable,
            "conversion_script": "derived/convert_access_to_training_grid.py",
            "spatial_interpolation_in_inference": "none",
            "model_run": str(run_dir.resolve()),
            "model_experiment": str(model_config["name"]),
            "weights": weights.name,
            "sampler": sampler_name,
            "sampler_steps": steps,
            "seed": seed,
            "training_sst_mean": float(normalization["sst_mean"]),
            "training_sst_std": float(normalization["sst_std"]),
        }
        writer = OperationalWriter(output, times, derived, attrs)
        mask = torch.from_numpy(
            derived.ocean_mask[None, None].astype(np.float32)
        ).to(device)
        try:
            for output_start in range(writer.completed, len(indices), batch_size):
                batch_indices = indices[output_start:output_start + batch_size]
                coarse = validate_converted_values(
                    field.isel(time=batch_indices).values, derived.ocean_mask_lr
                )
                condition = torch.from_numpy(
                    make_condition(
                        coarse,
                        derived.ocean_mask_lr,
                        normalization["sst_mean"],
                        normalization["sst_std"],
                    )
                ).to(device)
                batch_mask = mask.expand(len(batch_indices), -1, -1, -1)
                noise = _make_noise(
                    (len(batch_indices), 1, *derived.shape),
                    device,
                    condition.dtype,
                    batch_mask,
                    seed + batch_indices,
                )
                generated = get_sampler(sampler_name)(
                    model, noise, condition, batch_mask, steps
                )
                generated = generated.cpu().numpy()[:, 0]
                generated = (
                    generated * float(normalization["sst_std"])
                    + float(normalization["sst_mean"])
                )
                generated = np.where(
                    derived.ocean_mask[None], generated, np.nan
                ).astype(np.float32)
                writer.write(output_start, generated, coarse)
                if writer.completed == len(indices) or writer.completed % 100 == 0:
                    print(
                        f"[inference] {settings['period_name']} "
                        f"completed={writer.completed}/{len(indices)} "
                        f"date={days[writer.completed - 1]}",
                        flush=True,
                    )
            writer.finish(
                float(normalization["sst_mean"]), float(normalization["sst_std"])
            )
        except BaseException:
            writer.close()
            raise
    print(f"[ok] wrote {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--period-name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    arguments = parser.parse_args()
    run(load_settings(arguments.config, arguments.period_name), arguments.device)


if __name__ == "__main__":
    main()
