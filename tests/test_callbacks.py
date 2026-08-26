import numpy as np
import torch
import xarray as xr

from callbacks import (field_metrics, radial_spectrum, save_loss_curve,
                       save_netcdf, save_preview, save_rollout_netcdf,
                       to_physical)


def fields():
    target = np.arange(64, dtype=np.float32).reshape(8, 8); target[:2, :2] = np.nan
    generated = target + 1; coarse = np.nanmean(target.reshape(4, 2, 4, 2), axis=(1, 3))
    return coarse, target, generated


def test_to_physical_roundtrip():
    mask = np.ones((8, 8), bool); mask[:2, :2] = False
    normal = {"sst_mean": 10, "sst_std": 2}; values = torch.ones(1, 1, 8, 8)
    result = to_physical(values, normal, mask)
    assert np.isnan(result[0, 0, 0, 0]) and np.nanmax(result) == 12


def test_field_metrics_ignore_nan():
    _, target, generated = fields(); metrics = field_metrics(generated, target)
    assert metrics["rmse"] == 1 and metrics["mae"] == 1 and metrics["bias"] == 1
    assert metrics["nonfinite_ocean_pixels"] == 0


def test_save_preview_and_loss_curves(tmp_path):
    coarse, target, generated = fields(); preview = tmp_path / "preview.png"
    save_preview(coarse, target, generated, preview, "test", 2); assert preview.stat().st_size > 1000
    for count in (1, 10, 100):
        path = tmp_path / f"loss_{count}.png"; save_loss_curve([{"step": i + 1, "total": 1 / (i + 1)} for i in range(count)], path)
        assert path.stat().st_size > 1000


def test_save_netcdf_roundtrip_and_atomic(tmp_path):
    coarse, target, generated = fields(); path = tmp_path / "sample.nc"
    save_netcdf(generated[None], target[None], coarse[None], ["2000-01-01"],
                np.arange(8), np.arange(8), np.arange(4), np.arange(4), path, {"step": 1})
    with xr.open_dataset(path) as dataset:
        assert set(dataset.data_vars) == {"sst_generated", "sst_target", "sst_coarse"}
        assert dataset.sst_generated.dtype == np.float32 and dataset.attrs["units"] == "degrees C"
    assert not list(tmp_path.glob("*.partial.nc"))


def test_save_rollout_netcdf_roundtrip(tmp_path):
    _, target, generated = fields(); path = tmp_path / "rollout.nc"
    save_rollout_netcdf(np.stack([generated] * 2), np.stack([target] * 2),
                        ["2000-01-01", "2000-01-02"], np.arange(8), np.arange(8), path, {})
    with xr.open_dataset(path) as dataset:
        assert dataset.sizes["lead"] == 2 and str(dataset.time.values[1])[:10] == "2000-01-02"


def test_radial_spectrum_smooth_has_less_high_power():
    rng = np.random.default_rng(3); rough = rng.normal(size=(64, 64))
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(rough, 4); _, rp = radial_spectrum(rough); _, sp = radial_spectrum(smooth)
    assert np.mean(sp[-10:]) < np.mean(rp[-10:])
