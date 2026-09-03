"""Schema and restart tests for NOAA-grid test and ACCESS inference."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import torch

from infer_noaa_5km import (
    NOAAInferenceWriter,
    pad_fixed_batch,
    time_sha256,
    validate_access_grid,
)


def tiny_product():
    return SimpleNamespace(
        target_lat=np.asarray([-1.0, -0.5, 0.0, 0.5]),
        target_lon=np.asarray([100.0, 100.5, 101.0, 101.5, 102.0, 102.5]),
        coarse_lat=np.asarray([-0.5, 0.5]),
        coarse_lon=np.asarray([100.5, 102.0, 103.5]),
        target_mask=np.asarray(
            [
                [0, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
            ],
            dtype=bool,
        ),
        coarse_mask=np.asarray([[0, 1, 1], [1, 1, 0]], dtype=bool),
    )


def writer_attrs():
    return {
        "model_run": "/run",
        "weights_sha256": "a" * 64,
        "sampler": "ab3_pc",
        "sampler_steps": 75,
    }


def test_noaa_writer_resumes_and_atomically_promotes(tmp_path):
    product = tiny_product()
    times = np.arange(
        np.datetime64("2021-01-01"), np.datetime64("2021-01-04")
    ).astype("datetime64[ns]")
    output = tmp_path / "test.nc"
    target = np.broadcast_to(product.target_mask, (3, 4, 6)).astype(np.float32)
    target = np.where(product.target_mask[None], target + 20, np.nan)
    generated = np.where(product.target_mask[None], target + 0.1, np.nan)
    coarse = np.broadcast_to(product.coarse_mask, (3, 2, 3)).astype(np.float32)
    coarse = np.where(product.coarse_mask[None], coarse + 20, np.nan)

    writer = NOAAInferenceWriter(output, times, product, True, writer_attrs())
    writer.write(0, generated[:2], coarse[:2], target[:2])
    writer.close()
    assert not output.exists() and output.with_suffix(".partial.nc").exists()

    writer = NOAAInferenceWriter(output, times, product, True, writer_attrs())
    assert writer.completed == 2
    writer.write(2, generated[2:], coarse[2:], target[2:])
    writer.finish()
    assert output.exists() and not output.with_suffix(".partial.nc").exists()
    with xr.open_dataset(output) as dataset:
        assert set(dataset.data_vars) == {
            "sst_generated",
            "sst_target",
            "sst_coarse",
            "ocean_mask",
            "ocean_mask_lr",
        }
        assert dataset.sizes["time"] == 3
        assert dataset.attrs["completed"] == 3
        assert dataset.attrs["selected_time_sha256"] == time_sha256(times)
        np.testing.assert_array_equal(dataset.ocean_mask.values, product.target_mask)
        np.testing.assert_array_equal(dataset.ocean_mask_lr.values, product.coarse_mask)


def test_noaa_writer_refuses_incompatible_partial(tmp_path):
    product = tiny_product()
    times = np.asarray(["2021-01-01", "2021-01-02"], dtype="datetime64[ns]")
    output = tmp_path / "access.nc"
    writer = NOAAInferenceWriter(output, times, product, False, writer_attrs())
    writer.close()
    with pytest.raises(ValueError, match="dates"):
        NOAAInferenceWriter(
            output,
            np.asarray(["2021-01-01", "2021-01-03"], dtype="datetime64[ns]"),
            product,
            False,
            writer_attrs(),
        )


def test_access_grid_must_match_noaa_condition_grid():
    product = tiny_product()
    values = np.ones((2, 2, 3), dtype=np.float32)
    dataset = xr.Dataset(
        {"sst_lr": (("time", "lat_lr", "lon_lr"), values)},
        coords={
            "time": np.asarray(["1980-01-01", "1980-01-02"], dtype="datetime64[D]"),
            "lat_lr": product.coarse_lat,
            "lon_lr": product.coarse_lon,
        },
    )
    assert validate_access_grid(dataset, product, "sst_lr").shape == (2, 2, 3)
    shifted = dataset.assign_coords(lon_lr=dataset.lon_lr + 0.1)
    with pytest.raises(ValueError, match="training grid"):
        validate_access_grid(shifted, product, "sst_lr")


def test_fixed_batch_padding_preserves_real_members_and_uses_dummy_seeds():
    condition = torch.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2).float()
    padded, seeds, count = pad_fixed_batch(condition, np.asarray([10, 11, 12]), 4)
    assert count == 3 and padded.shape == (4, 2, 2, 2)
    torch.testing.assert_close(padded[:3], condition, rtol=0, atol=0)
    torch.testing.assert_close(padded[3], condition[-1], rtol=0, atol=0)
    np.testing.assert_array_equal(seeds[:3], [10, 11, 12])
    assert seeds[3] > 1_000_000


def test_fixed_batch_padding_rejects_bad_seed_shape():
    with pytest.raises(ValueError, match="seed shape"):
        pad_fixed_batch(torch.ones(2, 2, 2, 2), np.asarray([1]), 4)
