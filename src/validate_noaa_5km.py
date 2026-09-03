#!/usr/bin/env python3
"""Real-data preflight for the NOAA 0.05-degree transfer experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, load_config, load_json, selected_indices
from data_noaa_5km import NOAATransferDataset, NOAATransferProduct
from model_noaa_5km_v2 import coastline_ocean_mask, ocean_block_mean
from preprocess_noaa_5km import block_mean
from train_flow_noaa_5km_v2 import configure_paths


def run(config: dict, samples: int = 12) -> dict:
    config = configure_paths(config)
    normalization = load_json(config["normalization_cache"])
    if normalization.get("normalization_policy") != "fixed_pretrained_ofam_statistics":
        raise ValueError("NOAA transfer must retain the pretrained OFAM normalization")
    product = NOAATransferProduct(config["derived_path"])
    product.verify(normalization)
    split_indices = {
        name: selected_indices(product.times, config[f"{name}_date_ranges"])
        for name in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if np.intersect1d(split_indices[left], split_indices[right]).size:
            raise ValueError(f"NOAA {left}/{right} splits overlap")
    expected_counts = {"train": 10592, "validation": 730, "test": 1094}
    actual_counts = {name: int(len(values)) for name, values in split_indices.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected NOAA split counts {actual_counts}")

    dataset = NOAATransferDataset(
        config, normalization, config["train_date_ranges"], product
    )
    positions = np.linspace(0, len(dataset) - 1, min(samples, len(dataset))).astype(int)
    maximum_predictor_error = 0.0
    maximum_downsample_roundtrip = 0.0
    target_minimum, target_maximum = np.inf, -np.inf
    for position in positions:
        item = dataset[int(position)]
        for key in ("target", "base_target", "condition", "target_mask", "base_mask"):
            if not torch.isfinite(item[key]).all():
                raise FloatingPointError(f"Non-finite NOAA tensor {key} at {position}")
        physical = item["target"][0].numpy() * dataset.std + dataset.mean
        physical = np.where(product.target_mask, physical, np.nan)
        coarse, _ = block_mean(physical, product.target_mask, 32)
        condition = item["condition"][0].numpy() * dataset.std + dataset.mean
        predictor_error = float(
            np.max(np.abs(coarse[product.coarse_mask] - condition[product.coarse_mask]))
        )
        maximum_predictor_error = max(maximum_predictor_error, predictor_error)
        means, valid = ocean_block_mean(
            item["target"][None], item["target_mask"][None], 2
        )
        comparison = valid.bool() & item["base_mask"][None].bool()
        downsample_error = torch.abs(means - item["base_target"][None])
        maximum_downsample_roundtrip = max(
            maximum_downsample_roundtrip,
            float(torch.max(downsample_error[comparison])),
        )
        target_minimum = min(target_minimum, float(np.nanmin(physical)))
        target_maximum = max(target_maximum, float(np.nanmax(physical)))
    if maximum_predictor_error > 2e-5:
        raise AssertionError(f"Stored NOAA predictor mismatch {maximum_predictor_error}")
    if maximum_downsample_roundtrip > 2e-6:
        raise AssertionError(
            f"NOAA target-to-512 masked mean mismatch {maximum_downsample_roundtrip}"
        )
    target_mask_tensor = torch.from_numpy(product.target_mask.astype(np.float32))[None, None]
    coast_cells = int(coastline_ocean_mask(target_mask_tensor, 4).sum())
    repeated_base = np.repeat(np.repeat(product.base_mask, 2, 0), 2, 1)
    report = {
        "status": "passed",
        "source_dates": [str(product.times[0])[:10], str(product.times[-1])[:10]],
        "days": int(len(product.times)),
        "split_counts": actual_counts,
        "sampled_fields": int(len(positions)),
        "target_shape": list(product.target_shape),
        "target_ocean_cells": int(product.target_mask.sum()),
        "base_ocean_cells": int(product.base_mask.sum()),
        "coarse_ocean_cells": int(product.coarse_mask.sum()),
        "normalization_policy": normalization["normalization_policy"],
        "maximum_predictor_roundtrip_c": maximum_predictor_error,
        "maximum_target_to_base_roundtrip_normalized": maximum_downsample_roundtrip,
        "coastal_ocean_cells_within_4px": coast_cells,
        "noaa_only_target_cells": int((product.target_mask & ~repeated_base).sum()),
        "ofam_only_target_cells": int((~product.target_mask & repeated_base).sum()),
        "sampled_physical_range_c": [target_minimum, target_maximum],
        "missing_dates": normalization["source_missing_dates"],
    }
    output = Path(config["derived_path"]).with_suffix(".validation.json")
    atomic_json(output, report)
    dataset.close()
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    run(load_config(args.config), args.samples)


if __name__ == "__main__":
    main()
