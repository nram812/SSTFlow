"""C3: preprocessing and datasets on a miniature replica of the real file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import make_config, synthetic_fields, write_source
from data import (
    AutoregressiveSuperResolutionDataset,
    DerivedProduct,
    SuperResolutionDataset,
    build_dataset,
)


def test_preprocess_creates_products(config, normalization, synthetic_root):
    assert Path(config["derived_path"]).is_file()
    assert Path(config["normalization_cache"]).is_file()
    for key in (
        "sst_mean",
        "sst_std",
        "ocean_mask_sha256",
        "coarsen_factor",
        "grid_shape",
        "coarse_grid_shape",
        "training_days",
    ):
        assert key in normalization, key
    assert normalization["coarse_grid_shape"] == [8, 8]


def test_statistics_match_numpy(config, normalization, truth):
    fields, ocean = truth
    from common import selected_indices

    derived = DerivedProduct(config["derived_path"])
    indices = selected_indices(derived.times, config["train_date_ranges"])
    values = fields[indices][:, ocean]
    assert normalization["sst_mean"] == pytest.approx(float(values.mean()), abs=1e-3)
    assert normalization["sst_std"] == pytest.approx(float(values.std()), abs=1e-3)
    assert normalization["training_days"] == len(indices)


def test_statistics_use_training_range_only(config, normalization, truth):
    fields, ocean = truth
    everything = fields[:, ocean]
    # The full record has a different mean than the training window, so the
    # statistics genuinely exclude validation and test days.
    assert normalization["sst_mean"] != pytest.approx(
        float(everything.mean()), abs=1e-6
    )


def test_no_nan_in_any_batch(config, normalization, derived):
    dataset = SuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    for index in range(len(dataset)):
        item = dataset[index]
        for key in ("target", "condition", "mask"):
            assert torch.isfinite(item[key]).all(), (key, index)
    dataset.close()


def test_land_is_exactly_zero(config, normalization, derived):
    dataset = SuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    item = dataset[0]
    land = item["mask"] == 0
    assert land.any()
    assert float(item["target"][land].abs().max()) == 0.0
    coarse_land = torch.from_numpy(~derived.ocean_mask_lr)
    assert float(item["condition"][0][coarse_land].abs().max()) == 0.0
    dataset.close()


def test_mask_channel_matches_mask(config, normalization, derived):
    dataset = SuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    item = dataset[3]
    expected = torch.from_numpy(derived.ocean_mask_lr.astype(np.float32))
    torch.testing.assert_close(item["condition"][1], expected)
    dataset.close()


def test_mask_identical_across_items(config, normalization, derived):
    dataset = SuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    reference = dataset[0]["mask"]
    for index in (1, 5, len(dataset) - 1):
        torch.testing.assert_close(dataset[index]["mask"], reference)
    dataset.close()


def test_shapes(config, normalization, derived):
    dataset = SuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    item = dataset[0]
    factor = int(config["coarsen_factor"])
    height, width = derived.shape
    assert item["target"].shape == (1, height, width)
    assert item["condition"].shape == (2, height // factor, width // factor)
    assert item["mask"].shape == (1, height, width)
    dataset.close()


def test_normalized_statistics(config, normalization, derived):
    dataset = SuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    values = np.concatenate(
        [
            dataset[index]["target"][dataset[index]["mask"] > 0].numpy()
            for index in range(len(dataset))
        ]
    )
    assert abs(float(values.mean())) < 0.05
    assert abs(float(values.std()) - 1.0) < 0.05
    dataset.close()


def test_ar_pairs_are_consecutive(config, normalization, derived):
    dataset = AutoregressiveSuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    window = dataset.date_window(0)
    assert len(window) == 2
    first, second = np.asarray(window, dtype="datetime64[D]")
    assert second - first == np.timedelta64(1, "D")
    dataset.close()


def test_ar_pairs_do_not_cross_split(config, normalization, derived):
    from common import selected_indices

    dataset = AutoregressiveSuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    indices = selected_indices(derived.times, config["train_date_ranges"])
    assert len(dataset) == len(indices) - 1
    assert int(indices[-1]) not in set(int(v) for v in dataset.starts)
    dataset.close()


def test_ar_previous_matches_previous_item_target(config, normalization, derived):
    dataset = AutoregressiveSuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    torch.testing.assert_close(dataset[1]["previous"], dataset[0]["target"])
    dataset.close()


def test_preload_matches_lazy(config, normalization, derived):
    lazy = SuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    eager = SuperResolutionDataset(
        config,
        normalization,
        config["train_date_ranges"],
        derived=derived,
        preload=True,
    )
    for index in (0, 4, len(lazy) - 1):
        torch.testing.assert_close(lazy[index]["target"], eager[index]["target"])
    lazy.close()
    eager.close()


def test_fingerprint_mismatch_raises(config, normalization, derived):
    tampered = {**normalization, "ocean_mask_sha256": "0" * 64}
    with pytest.raises(ValueError, match="fingerprint"):
        SuperResolutionDataset(
            config, tampered, config["train_date_ranges"], derived=derived
        )


def test_dataloader_multiple_workers(config, normalization, derived):
    # PyTorch's worker result queue needs a local AF_UNIX socket.  The Codex
    # filesystem sandbox denies socket creation; normal login/PBS nodes do not.
    import socket
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.bind("")
        probe.close()
    except PermissionError:
        pytest.skip("sandbox forbids multiprocessing resource-sharer sockets")
    dataset = SuperResolutionDataset(
        config, normalization, config["train_date_ranges"], derived=derived
    )
    loader = DataLoader(
        dataset, batch_size=2, num_workers=2, multiprocessing_context="spawn"
    )
    # Exhaust the iterator so PyTorch joins both workers cleanly; abandoning an
    # iterator after one batch can emit a spurious worker-aborted warning at
    # interpreter shutdown even though the batch itself succeeded.
    batches = list(loader)
    batch = batches[0]
    assert batch["target"].shape[0] == 2
    assert torch.isfinite(batch["target"]).all()
    assert torch.isfinite(batch["condition"]).all()
    dataset.close()


def test_build_dataset_factory(config, normalization, derived):
    plain = build_dataset(
        config, normalization, config["train_date_ranges"], "super_resolution",
        derived=derived,
    )
    auto = build_dataset(
        config, normalization, config["train_date_ranges"], "autoregressive",
        derived=derived,
    )
    assert isinstance(plain, SuperResolutionDataset)
    assert isinstance(auto, AutoregressiveSuperResolutionDataset)
    with pytest.raises(ValueError, match="Unknown dataset kind"):
        build_dataset(
            config, normalization, config["train_date_ranges"], "nonsense",
            derived=derived,
        )
    plain.close()
    auto.close()


def test_combined_preprocess_and_reader_map_unique_days_to_two_sources(
    tmp_path, normalization
):
    import netCDF4
    from preprocess_combined import run

    fields, _ = synthetic_fields()
    first = tmp_path / "historical.nc"
    second = tmp_path / "future.nc"
    write_source(first, fields[:20])
    write_source(second, fields[20:])
    with netCDF4.Dataset(second, "a") as source:
        source.variables["Time"][:] = np.arange(20, 40, dtype=np.float64)

    combined = make_config(tmp_path, "combined")
    combined.update(
        source_paths=[str(first), str(second)],
        source_date_ranges=[
            ["1979-01-01", "1979-01-20"],
            ["1979-01-21", "1979-02-09"],
        ],
        derived_path=str(tmp_path / "combined_derived.nc"),
        train_date_ranges=[["1979-01-01", "1979-02-09"]],
    )
    summary = run(combined, chunk=7, probe_days=4)
    derived = DerivedProduct(combined["derived_path"])
    dataset = SuperResolutionDataset(
        combined, normalization, combined["train_date_ranges"], derived=derived
    )

    assert summary["days"] == 40
    assert np.array_equal(derived.source_id[:20], np.zeros(20, dtype=np.int16))
    assert np.array_equal(derived.source_id[20:], np.ones(20, dtype=np.int16))
    assert np.array_equal(derived.source_index[:20], np.arange(20))
    assert np.array_equal(derived.source_index[20:], np.arange(20))
    expected = (fields[20] - normalization["sst_mean"]) / normalization["sst_std"]
    ocean = dataset[20]["mask"][0].bool()
    torch.testing.assert_close(
        dataset[20]["target"][0][ocean],
        torch.from_numpy(expected.astype(np.float32))[ocean],
        atol=2.0e-3,
        rtol=0,
    )
    dataset.close()
