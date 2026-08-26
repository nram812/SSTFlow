#!/usr/bin/env python3
"""Preflight validation of the real data before any long training run.

Nine checks, each of which raises with an explanatory message rather than
silently producing a NaN thousands of optimiser steps later.  Run it with

    pixi run validate-data

after ``pixi run preprocess`` and before submitting a training job.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common import (
    date_keys,
    load_config,
    load_json,
    mask_sha256,
    selected_indices,
)
from data import DerivedProduct, build_dataset
from losses import masked_mse
from preprocess import open_source, read_days, source_times

CHECKS: list[tuple[str, str]] = []


def report(name: str, detail: str) -> None:
    CHECKS.append((name, detail))
    print(f"[ok] {name}: {detail}", flush=True)


def check_source(config: dict) -> dict:
    dataset = open_source(config["source_path"])
    try:
        times = source_times(dataset)
        variable = dataset.variables["temp"]
        shape = tuple(int(value) for value in variable.shape)
        days = date_keys(times)
        deltas = np.diff(np.asarray(days, dtype="datetime64[D]"))
        if not np.all(deltas == np.timedelta64(1, "D")):
            gaps = int((deltas != np.timedelta64(1, "D")).sum())
            raise ValueError(f"Source calendar has {gaps} non-daily gaps")
        report(
            "source",
            f"shape={shape} days={len(times)} {days[0]}..{days[-1]} no gaps",
        )
        return {"times": times, "shape": shape}
    finally:
        dataset.close()


def check_mask_stationary(config: dict, probes: int = 100) -> np.ndarray:
    dataset = open_source(config["source_path"])
    try:
        total = dataset.variables["temp"].shape[0]
        indices = np.unique(np.linspace(0, total - 1, probes).astype(int))
        reference = None
        for index in indices:
            mask = np.isfinite(read_days(dataset, int(index), int(index) + 1)[0])
            if reference is None:
                reference = mask
            elif not np.array_equal(mask, reference):
                raise ValueError(
                    f"Land mask changes at time index {index}: "
                    f"{int((mask != reference).sum())} cells differ"
                )
        report(
            "mask_stationary",
            f"{len(indices)} probes identical; ocean fraction "
            f"{float(reference.mean()):.4f}",
        )
        return reference
    finally:
        dataset.close()


def check_fingerprint(derived: DerivedProduct, normalization: dict) -> None:
    digest = mask_sha256(derived.ocean_mask, derived.lat, derived.lon)
    if digest != normalization["ocean_mask_sha256"]:
        raise ValueError(
            f"Fingerprint mismatch: derived={digest} "
            f"statistics={normalization['ocean_mask_sha256']}"
        )
    report("fingerprint", f"{digest[:16]}... matches the statistics file")


def check_splits(config: dict, derived: DerivedProduct) -> None:
    sets = {}
    for name in ("train", "validation", "test"):
        indices = selected_indices(derived.times, config[f"{name}_date_ranges"])
        gaps = np.diff(indices)
        if not np.all(gaps == 1):
            raise ValueError(f"{name} split is not contiguous in time")
        sets[name] = set(int(value) for value in indices)
    for left in sets:
        for right in sets:
            if left < right and sets[left] & sets[right]:
                raise ValueError(
                    f"{left} and {right} splits overlap by "
                    f"{len(sets[left] & sets[right])} days"
                )
    report(
        "splits",
        ", ".join(f"{name}={len(values)}" for name, values in sets.items())
        + " (disjoint, gapless)",
    )


def check_batches(config: dict, normalization: dict, derived: DerivedProduct,
                  samples: int = 256) -> dict:
    dataset = build_dataset(
        config,
        normalization,
        config["train_date_ranges"],
        "super_resolution",
        derived=derived,
        preload=False,
    )
    try:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(dataset), size=min(samples, len(dataset)),
                             replace=False)
        ocean_values = []
        for index in indices:
            item = dataset[int(index)]
            for key in ("target", "condition", "mask"):
                tensor = item[key]
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"Non-finite value in {key!r} at dataset index {index}"
                    )
            land = item["mask"] == 0
            if float(item["target"][land].abs().max() if land.any() else 0.0) != 0.0:
                raise ValueError(f"Land is not exactly zero at index {index}")
            ocean_values.append(
                item["target"][item["mask"] > 0].numpy().astype(np.float64)
            )
        stacked = np.concatenate(ocean_values)
        report(
            "batches",
            f"{len(indices)} random items all finite; land exactly zero",
        )
        mean, std = float(stacked.mean()), float(stacked.std())
        if abs(mean) > 0.05 or abs(std - 1.0) > 0.05:
            raise ValueError(
                f"Normalised ocean statistics are off: mean={mean:.4f} std={std:.4f}"
            )
        report("normalisation", f"ocean mean={mean:.4f} std={std:.4f}")
        return {"normalized_mean": mean, "normalized_std": std}
    finally:
        dataset.close()


def check_coarse(derived: DerivedProduct) -> None:
    values = derived.sst_lr[:, derived.ocean_mask_lr]
    if not np.isfinite(values).all():
        raise ValueError(
            f"{int((~np.isfinite(values)).sum())} non-finite coarse values "
            "inside the coarse ocean mask"
        )
    report(
        "coarse_predictor",
        f"grid={derived.coarse_shape} valid cells="
        f"{int(derived.ocean_mask_lr.sum())}/{derived.ocean_mask_lr.size} all finite",
    )


def check_roundtrip(config: dict, normalization: dict,
                    derived: DerivedProduct) -> None:
    dataset = build_dataset(
        config,
        normalization,
        config["validation_date_ranges"],
        "super_resolution",
        derived=derived,
        preload=False,
    )
    source = open_source(config["source_path"])
    try:
        item = dataset[0]
        index = item["index"]
        truth = read_days(source, index, index + 1)[0]
        recovered = (
            item["target"][0].numpy() * float(normalization["sst_std"])
            + float(normalization["sst_mean"])
        )
        ocean = derived.ocean_mask
        error = float(np.abs(recovered[ocean] - truth[ocean]).max())
        if error > 1.0e-3:
            raise ValueError(f"Round-trip error {error:.6f} degC is too large")
        report("roundtrip", f"max |error| = {error:.2e} degC over ocean")
    finally:
        dataset.close()
        source.close()


def check_persistence(config: dict, normalization: dict,
                      derived: DerivedProduct) -> dict:
    dataset = build_dataset(
        config,
        normalization,
        config["validation_date_ranges"],
        "autoregressive",
        derived=derived,
        preload=False,
    )
    try:
        indices = np.linspace(0, len(dataset) - 1, min(32, len(dataset))).astype(int)
        losses = []
        for index in indices:
            item = dataset[int(index)]
            losses.append(
                float(
                    masked_mse(
                        item["previous"][None],
                        item["target"][None],
                        item["mask"][None],
                    )
                )
            )
        baseline = float(np.mean(losses))
        physical = baseline * float(normalization["sst_std"]) ** 2
        report(
            "persistence_baseline",
            f"normalised MSE={baseline:.6f} (RMSE={np.sqrt(physical):.4f} degC) "
            f"over {len(indices)} validation pairs",
        )
        return {"persistence_mse_normalized": baseline}
    finally:
        dataset.close()


def run(config: dict) -> dict:
    normalization = load_json(config["normalization_cache"])
    derived = DerivedProduct(config["derived_path"])

    check_source(config)
    mask = check_mask_stationary(config)
    if not np.array_equal(mask, derived.ocean_mask):
        raise ValueError("Source mask differs from the derived ocean mask")
    check_fingerprint(derived, normalization)
    check_splits(config, derived)
    check_coarse(derived)
    statistics = check_batches(config, normalization, derived)
    check_roundtrip(config, normalization, derived)
    persistence = check_persistence(config, normalization, derived)

    summary = {
        "config": config["name"],
        "checks_passed": len(CHECKS),
        "details": dict(CHECKS),
        **statistics,
        **persistence,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run(load_config(arguments.config))


if __name__ == "__main__":
    main()
