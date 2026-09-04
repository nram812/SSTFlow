"""Small real-data contract test for the 32x32 -> 512x512 OFAM product."""

from pathlib import Path

import netCDF4
import numpy as np
import tensorflow as tf

from model_srdn_advanced import SRDCNN_SST_v3, SRDN_ResAFNO_v4
from srdn_data import DerivedProduct, SRDNData


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sst_10km_OFAM_historical_Australia.nc"
DERIVED = ROOT / "derived" / "sst_downscaling_f16.nc"
NORMALIZATION = ROOT / "reports" / "normalization_f16.json"


def _assert_projection(output, inputs, derived):
    output = output.numpy()[0, ..., 0]
    fine_mask = inputs["fine_mask"][0, ..., 0].astype(bool)
    coarse = inputs["coarse_sst"][0, ..., 0]
    coarse_mask = inputs["coarse_mask"][0, ..., 0].astype(bool)
    assert np.all(output[~fine_mask] == 0.0)
    blocks = output.reshape(
        derived.coarse_shape[0], derived.coarsen_factor,
        derived.coarse_shape[1], derived.coarsen_factor,
    )
    masks = fine_mask.reshape(
        derived.coarse_shape[0], derived.coarsen_factor,
        derived.coarse_shape[1], derived.coarsen_factor,
    )
    means = (blocks * masks).sum(axis=(1, 3)) / np.maximum(masks.sum(axis=(1, 3)), 1)
    valid = coarse_mask & (masks.sum(axis=(1, 3)) > 0)
    np.testing.assert_allclose(means[valid], coarse[valid], atol=2e-5)


def test_real_data_contract_and_forward():
    assert SOURCE.exists(), SOURCE
    assert DERIVED.exists(), DERIVED
    assert NORMALIZATION.exists(), NORMALIZATION
    derived = DerivedProduct(DERIVED)
    assert derived.fine_shape == (512, 512)
    assert derived.coarse_shape == (32, 32)
    data = SRDNData(
        SOURCE,
        derived,
        NORMALIZATION,
        [["2011-01-01", "2011-01-02"]],
    )
    inputs, target = data.batch([0, 1])
    assert inputs["coarse_sst"].shape == (2, 32, 32, 1)
    assert inputs["coarse_mask"].shape == (2, 32, 32, 1)
    assert inputs["fine_mask"].shape == (2, 512, 512, 1)
    assert target.shape == (2, 512, 512, 1)
    assert np.isfinite(target).all()
    assert np.all(target[inputs["fine_mask"] == 0.0] == 0.0)

    # Confirm the loader uses the decoded/packed NetCDF values rather than raw
    # int16 codes for one ocean point.
    with netCDF4.Dataset(SOURCE, "r") as source:
        raw = np.ma.filled(source.variables["temp"][data.indices[0], 0], np.nan)
    ocean = np.argwhere(derived.ocean_mask)[0]
    expected = (float(raw[tuple(ocean)]) - data.mean) / data.std
    np.testing.assert_allclose(target[0, ocean[0], ocean[1], 0], expected, atol=1e-5)

    for builder in (
        lambda: SRDCNN_SST_v3(numHiddenUnits=16),
        lambda: SRDN_ResAFNO_v4(
            numHiddenUnits=16, trunk_blocks=1, num_freq_blocks=4
        ),
    ):
        model = builder()
        output = model(inputs, training=False)
        assert output.shape == (2, 512, 512, 1)
        assert np.isfinite(output.numpy()).all()
        _assert_projection(output, inputs, derived)
    data.close()


if __name__ == "__main__":
    test_real_data_contract_and_forward()
    print("REAL-DATA CONTRACT PASSED")
