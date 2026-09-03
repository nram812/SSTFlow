#!/usr/bin/env python3
"""Run the trained NOAA 0.05-degree flow on NOAA test or ACCESS predictors.

The model always consumes the established 32x32 physical SST predictor plus
its fixed ocean mask.  NOAA test inference writes both the satellite truth and
the generated 1024x1024 field.  ACCESS inference changes only the 32x32 SST
boundary; it retains the NOAA target grid and target-ocean mask learned during
fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import torch
import xarray as xr

import engine
from common import REPOSITORY_ROOT, atomic_json, load_json
from data_noaa_5km import NOAATransferDataset, NOAATransferProduct
from flow import SAMPLERS, get_sampler
from infer_access_cm2 import (
    _make_noise,
    make_condition,
    select_time_indices,
    validate_converted_values,
)
from model_noaa_5km_v2 import NOAAFrozenTrunkFlow
from train_flow_noaa_5km_v2 import configure_paths


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "noaa_5km_inference_150k.json"


def time_sha256(times: np.ndarray) -> str:
    encoded = np.asarray(times).astype("datetime64[ns]").view("i8").tobytes()
    return hashlib.sha256(encoded).hexdigest()


def validate_access_grid(
    dataset: xr.Dataset, product: NOAATransferProduct, variable: str
) -> xr.DataArray:
    """Require the ACCESS product to match the model's 32x32 condition grid."""
    if variable not in dataset:
        raise ValueError(f"converted ACCESS input lacks {variable!r}")
    field = dataset[variable]
    if field.dims != ("time", "lat_lr", "lon_lr"):
        raise ValueError(f"unexpected ACCESS dimensions {field.dims}")
    for name, wanted in (
        ("lat_lr", product.coarse_lat),
        ("lon_lr", product.coarse_lon),
    ):
        if name not in dataset.coords:
            raise ValueError(f"converted ACCESS input has no {name} coordinate")
        actual = np.asarray(dataset[name].values)
        if actual.shape != wanted.shape or not np.allclose(
            actual, wanted, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"converted ACCESS {name} is not the training grid")
    return field


def pad_fixed_batch(
    condition: torch.Tensor, seeds: np.ndarray, batch_size: int
) -> tuple[torch.Tensor, np.ndarray, int]:
    """Pad only a final short batch so every GPU solve has one fixed shape.

    Scaled-dot-product attention can select different floating-point kernels for
    different batch shapes.  Keeping the shape fixed makes a configured
    inference reproducible, including its final dates.  Padded members are
    discarded and never written.
    """
    count = len(condition)
    if count < 1 or count > int(batch_size):
        raise ValueError(f"invalid inference batch length {count}/{batch_size}")
    seeds = np.asarray(seeds, dtype=np.int64)
    if seeds.shape != (count,):
        raise ValueError(f"seed shape {seeds.shape} does not match batch length {count}")
    if count == batch_size:
        return condition, seeds, count
    padding = int(batch_size) - count
    padded_condition = torch.cat(
        (condition, condition[-1:].expand(padding, -1, -1, -1)), dim=0
    )
    # Dummy seeds cannot collide with real per-date seeds in this experiment.
    padded_seeds = np.concatenate(
        (seeds, np.arange(padding, dtype=np.int64) + 9_000_000_000)
    )
    return padded_condition, padded_seeds, count


class NOAAInferenceWriter:
    """Resumable daily writer, atomically promoted only after all dates exist."""

    def __init__(
        self,
        output: Path,
        times: np.ndarray,
        product: NOAATransferProduct,
        include_target: bool,
        attrs: dict,
    ):
        self.output = Path(output)
        self.partial = self.output.with_suffix(".partial.nc")
        self.include_target = bool(include_target)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise FileExistsError(f"output already exists: {self.output}")
        expected_time_hash = time_sha256(times)
        if self.partial.exists():
            self.dataset = netCDF4.Dataset(self.partial, "r+")
            if self.dataset.getncattr("selected_time_sha256") != expected_time_hash:
                self.close()
                raise ValueError("partial output dates differ from this request")
            if bool(int(self.dataset.getncattr("includes_target"))) != self.include_target:
                self.close()
                raise ValueError("partial target schema differs from this request")
            for name in ("model_run", "weights_sha256", "sampler", "sampler_steps"):
                if str(self.dataset.getncattr(name)) != str(attrs[name]):
                    self.close()
                    raise ValueError(f"partial output {name} differs from this request")
            self.completed = int(self.dataset.getncattr("completed"))
            return

        ds = netCDF4.Dataset(self.partial, "w", format="NETCDF4")
        self.dataset = ds
        ds.createDimension("time", len(times))
        ds.createDimension("lat_target", len(product.target_lat))
        ds.createDimension("lon_target", len(product.target_lon))
        ds.createDimension("lat_lr", len(product.coarse_lat))
        ds.createDimension("lon_lr", len(product.coarse_lon))
        time_variable = ds.createVariable("time", "f8", ("time",))
        time_variable.units = "days since 1850-01-01 00:00:00"
        time_variable.calendar = "proleptic_gregorian"
        time_variable[:] = (
            times.astype("datetime64[ns]") - np.datetime64("1850-01-01")
        ) / np.timedelta64(1, "D")
        ds.createVariable("lat_target", "f8", ("lat_target",))[:] = product.target_lat
        ds.createVariable("lon_target", "f8", ("lon_target",))[:] = product.target_lon
        ds.createVariable("lat_lr", "f8", ("lat_lr",))[:] = product.coarse_lat
        ds.createVariable("lon_lr", "f8", ("lon_lr",))[:] = product.coarse_lon
        generated_name = "sst_generated" if include_target else "sst_downscaled"
        generated = ds.createVariable(
            generated_name,
            "f4",
            ("time", "lat_target", "lon_target"),
            zlib=True,
            complevel=4,
            chunksizes=(1, len(product.target_lat), len(product.target_lon)),
            fill_value=np.float32(np.nan),
        )
        generated.units = "degrees C"
        if include_target:
            target = ds.createVariable(
                "sst_target",
                "f4",
                ("time", "lat_target", "lon_target"),
                zlib=True,
                complevel=4,
                chunksizes=(1, len(product.target_lat), len(product.target_lon)),
                fill_value=np.float32(np.nan),
            )
            target.units = "degrees C"
        coarse = ds.createVariable(
            "sst_coarse",
            "f4",
            ("time", "lat_lr", "lon_lr"),
            zlib=True,
            complevel=4,
            chunksizes=(1, len(product.coarse_lat), len(product.coarse_lon)),
            fill_value=np.float32(np.nan),
        )
        coarse.units = "degrees C"
        ds.createVariable("ocean_mask", "u1", ("lat_target", "lon_target"))[:] = (
            product.target_mask
        )
        ds.createVariable("ocean_mask_lr", "u1", ("lat_lr", "lon_lr"))[:] = (
            product.coarse_mask
        )
        for name, value in attrs.items():
            ds.setncattr(name, value)
        ds.setncattr("selected_time_sha256", expected_time_hash)
        ds.setncattr("includes_target", int(include_target))
        ds.setncattr("completed", 0)
        ds.sync()
        self.completed = 0

    def write(
        self,
        start: int,
        generated: np.ndarray,
        coarse: np.ndarray,
        target: np.ndarray | None = None,
    ) -> None:
        stop = start + len(generated)
        generated_name = "sst_generated" if self.include_target else "sst_downscaled"
        self.dataset.variables[generated_name][start:stop] = generated
        self.dataset.variables["sst_coarse"][start:stop] = coarse
        if self.include_target:
            if target is None:
                raise ValueError("test writer requires target fields")
            self.dataset.variables["sst_target"][start:stop] = target
        self.completed = stop
        self.dataset.setncattr("completed", stop)
        self.dataset.sync()

    def finish(self) -> None:
        expected = len(self.dataset.dimensions["time"])
        if self.completed != expected:
            raise ValueError(f"cannot finish incomplete output {self.completed}/{expected}")
        self.dataset.close()
        os.replace(self.partial, self.output)

    def close(self) -> None:
        if getattr(self, "dataset", None) is not None and self.dataset.isopen():
            self.dataset.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(run_dir: Path, device: torch.device):
    config = configure_paths(load_json(run_dir / "config_used.json"))
    normalization = load_json(run_dir / "normalization.json")
    product = NOAATransferProduct(config["derived_path"])
    product.verify(normalization)
    weights = run_dir / "model_ema.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"missing final EMA weights: {weights}")
    status = load_json(run_dir / "status.json")
    if status.get("status") != "passed" or int(status.get("step", -1)) != int(
        config["max_steps"]
    ):
        raise ValueError(f"run is not complete: {status}")
    model = NOAAFrozenTrunkFlow.from_pretrained(
        config,
        torch.from_numpy(product.base_mask.astype(np.float32)),
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return config, normalization, product, model.eval(), weights


def _settings(config_path: Path, period_name: str | None = None) -> dict:
    settings = load_json(config_path)
    if period_name is not None:
        if period_name not in settings["periods"]:
            raise ValueError(f"unknown period {period_name!r}")
        settings = {**settings, **settings["periods"][period_name]}
        settings["period_name"] = period_name
    for name in ("run_dir", "input_path", "output"):
        if name not in settings:
            continue
        value = Path(settings[name])
        settings[name] = value if value.is_absolute() else REPOSITORY_ROOT / value
    return settings


def _attrs(settings: dict, config: dict, weights: Path, product_name: str) -> dict:
    return {
        "product": product_name,
        "model_run": str(Path(settings["run_dir"]).resolve()),
        "model_experiment": str(config["name"]),
        "weights": weights.name,
        "weights_sha256": _file_sha256(weights),
        "sampler": str(settings["sampler"]),
        "sampler_steps": int(settings["sampler_steps"]),
        "batch_size": int(settings["batch_size"]),
        "fixed_batch_padding": "final short batch padded to configured batch_size",
        "seed": int(settings["seed"]),
        "target_grid": "NOAA 0.05-degree nested Australian crop",
        "target_mask": "fixed NOAA training ocean mask",
    }


def _physical(values: torch.Tensor, mean: float, std: float, mask: np.ndarray):
    result = values.detach().cpu().numpy()[:, 0] * std + mean
    return np.where(mask[None], result, np.nan).astype(np.float32)


@torch.no_grad()
def run_test(settings: dict, device_name: str | None = None) -> Path:
    device = engine.resolve_device(device_name)
    run_dir = Path(settings["run_dir"])
    config, normalization, product, model, weights = load_run(run_dir, device)
    if settings["sampler"] not in SAMPLERS:
        raise ValueError(f"unknown flow sampler {settings['sampler']!r}")
    dataset = NOAATransferDataset(
        config, normalization, config["test_date_ranges"], product
    )
    times = product.times[dataset.indices]
    expected = int(settings.get("expected_test_days", len(dataset)))
    if len(dataset) != expected:
        raise ValueError(f"test split has {len(dataset)} days, expected {expected}")
    attrs = _attrs(settings, config, weights, "NOAA test SST downscaled by rectified flow")
    attrs["selection"] = "full_test"
    writer = NOAAInferenceWriter(
        Path(settings["output"]), times, product, include_target=True, attrs=attrs
    )
    mean, std = float(normalization["sst_mean"]), float(normalization["sst_std"])
    batch_size = int(settings["batch_size"])
    sampler = get_sampler(str(settings["sampler"]))
    mask = torch.from_numpy(product.target_mask[None, None].astype(np.float32)).to(device)
    try:
        for start in range(writer.completed, len(dataset), batch_size):
            items = np.arange(start, min(start + batch_size, len(dataset)))
            batch = engine.batch_to_device(engine.collate_indices(dataset, items), device)
            absolute_indices = np.asarray(batch["index"], dtype=np.int64)
            condition, seeds, count = pad_fixed_batch(
                batch["condition"],
                int(settings["seed"]) + absolute_indices,
                batch_size,
            )
            batch_mask = mask.expand(batch_size, -1, -1, -1)
            noise = _make_noise(
                (batch_size, 1, *product.target_shape),
                device,
                condition.dtype,
                batch_mask,
                seeds,
            )
            generated = sampler(
                model,
                noise,
                condition,
                batch_mask,
                int(settings["sampler_steps"]),
            )[:count]
            generated_p = _physical(generated, mean, std, product.target_mask)
            target_p = _physical(batch["target"], mean, std, product.target_mask)
            coarse = batch["condition"][:, 0].cpu().numpy() * std + mean
            coarse = np.where(product.coarse_mask[None], coarse, np.nan).astype(np.float32)
            writer.write(start, generated_p, coarse, target_p)
            if writer.completed == len(dataset) or writer.completed % 100 == 0:
                print(
                    f"[NOAA test] {writer.completed}/{len(dataset)} "
                    f"date={str(times[writer.completed - 1])[:10]}",
                    flush=True,
                )
        writer.finish()
    except BaseException:
        writer.close()
        raise
    dataset.close()
    return Path(settings["output"])


@torch.no_grad()
def run_access(settings: dict, device_name: str | None = None) -> Path:
    device = engine.resolve_device(device_name)
    run_dir = Path(settings["run_dir"])
    config, normalization, product, model, weights = load_run(run_dir, device)
    if settings["sampler"] not in SAMPLERS:
        raise ValueError(f"unknown flow sampler {settings['sampler']!r}")
    input_path = Path(settings["input_path"])
    variable = str(settings.get("variable", "sst_lr"))
    with xr.open_dataset(input_path, engine="h5netcdf") as source:
        field = validate_access_grid(source, product, variable)
        indices = select_time_indices(
            source.time.values, start=settings["start"], end=settings["end"]
        )
        times = source.time.values[indices]
        if len(indices) != int(settings["expected_days"]):
            raise ValueError(
                f"{settings['period_name']} has {len(indices)} days, "
                f"expected {settings['expected_days']}"
            )
        attrs = _attrs(
            settings, config, weights, "ACCESS-CM2 SST downscaled to NOAA 0.05-degree grid"
        )
        attrs.update(
            {
                "period_name": settings["period_name"],
                "converted_input_path": str(input_path.resolve()),
                "converted_input_variable": variable,
                "spatial_interpolation_in_inference": "none",
            }
        )
        writer = NOAAInferenceWriter(
            Path(settings["output"]), times, product, include_target=False, attrs=attrs
        )
        mean, std = float(normalization["sst_mean"]), float(normalization["sst_std"])
        batch_size = int(settings["batch_size"])
        sampler = get_sampler(str(settings["sampler"]))
        mask = torch.from_numpy(product.target_mask[None, None].astype(np.float32)).to(device)
        try:
            for start in range(writer.completed, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                coarse = validate_converted_values(
                    field.isel(time=batch_indices).values, product.coarse_mask
                )
                condition = torch.from_numpy(
                    make_condition(coarse, product.coarse_mask, mean, std)
                ).to(device)
                condition, seeds, count = pad_fixed_batch(
                    condition,
                    int(settings["seed"]) + batch_indices,
                    batch_size,
                )
                batch_mask = mask.expand(batch_size, -1, -1, -1)
                noise = _make_noise(
                    (batch_size, 1, *product.target_shape),
                    device,
                    condition.dtype,
                    batch_mask,
                    seeds,
                )
                generated = sampler(
                    model,
                    noise,
                    condition,
                    batch_mask,
                    int(settings["sampler_steps"]),
                )[:count]
                writer.write(start, _physical(generated, mean, std, product.target_mask), coarse)
                if writer.completed == len(indices) or writer.completed % 100 == 0:
                    print(
                        f"[NOAA ACCESS {settings['period_name']}] "
                        f"{writer.completed}/{len(indices)} "
                        f"date={str(times[writer.completed - 1])[:10]}",
                        flush=True,
                    )
            writer.finish()
        except BaseException:
            writer.close()
            raise
    return Path(settings["output"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test", "access"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--period-name", choices=("historical", "future"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sampler-steps", type=int)
    args = parser.parse_args()
    settings = _settings(args.config, args.period_name if args.mode == "access" else None)
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.sampler_steps is not None:
        settings["sampler_steps"] = args.sampler_steps
    if args.mode == "test":
        output = run_test(settings, args.device)
    else:
        if args.period_name is None:
            parser.error("access mode requires --period-name")
        output = run_access(settings, args.device)
    atomic_json(
        output.with_suffix(".manifest.json"),
        {"status": "completed", "mode": args.mode, "output": str(output.resolve())},
    )
    print(json.dumps({"status": "completed", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
