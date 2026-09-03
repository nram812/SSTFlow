#!/usr/bin/env python3
"""Execute multi-model evaluation across all 7 super-resolution models.

Processes each model sequentially to maintain low peak memory and high performance.
Computes:
1. Part 1: Test-Set (OFAM Holdout 2011-2014) Climatology, TXx, MHW, and Spatial Errors.
2. Part 2: Historical (ACCESS-CM2 1980-1989) Climatology & Added Value vs OFAM Ground Truth.
3. Part 3: Climate Change Signals (SSP585 2080-2089 vs Historical 1980-1989) & MHW Projections.

Saves high-resolution figures to figures/ and outputs structured summary tables.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import xarray as xr

# Style configuration
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.facecolor"] = "#f8f9fa"
plt.rcParams["axes.facecolor"] = "#edf0f2"

BASE_DIR = Path("/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling")
RUNS_DIR = BASE_DIR / "runs"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "flow_sr": "Flow-SR (AB3-PC 75st)",
    "gan_sr_v2": "GAN-SR v2",
    "gan_sr_v2_hist_rcp85_continue_220k": "GAN-SR v2 (Hist/RCP85 220k)",
    "gan_sr_v2b_image_only_critic": "GAN-SR v2b (Image Critic)",
    "gan_sr_v2b_hist_rcp85_continue_220k": "GAN-SR v2b (Hist/RCP85 220k)",
    "gan_sr_v3_hard_consistency": "GAN-SR v3 (Hard Cons.)",
    "gan_sr_v3_hist_rcp85_continue_220k": "GAN-SR v3 (Hard Cons. 220k)",
}
MODEL_KEYS = list(MODEL_REGISTRY.keys())
SEASONS = ["DJF", "MAM", "JJA", "SON"]
OFAM_FILE = BASE_DIR / "sst_10km_OFAM_historical_Australia.nc"


def spatial_rmse(pred: xr.DataArray, truth: xr.DataArray) -> float:
    diff = pred - truth
    return float(np.sqrt(np.nanmean(diff.values ** 2)))


def spatial_bias(pred: xr.DataArray, truth: xr.DataArray) -> float:
    return float(np.nanmean((pred - truth).values))


def spatial_mae(pred: xr.DataArray, truth: xr.DataArray) -> float:
    return float(np.nanmean(np.abs((pred - truth).values)))


def spatial_corr(pred: xr.DataArray, truth: xr.DataArray) -> float:
    p = pred.values.ravel()
    t = truth.values.ravel()
    mask = np.isfinite(p) & np.isfinite(t)
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(p[mask], t[mask])[0, 1])


def interp_coarse_to_fine(da_coarse: xr.DataArray, fine_lat: np.ndarray, fine_lon: np.ndarray) -> xr.DataArray:
    lat_dim = "lat_lr" if "lat_lr" in da_coarse.dims else ("lat" if "lat" in da_coarse.dims else da_coarse.dims[-2])
    lon_dim = "lon_lr" if "lon_lr" in da_coarse.dims else ("lon" if "lon" in da_coarse.dims else da_coarse.dims[-1])
    return da_coarse.interp({lat_dim: fine_lat, lon_dim: fine_lon}, method="linear").rename({lat_dim: "lat", lon_dim: "lon"})


def prepare_ofam(ds: xr.Dataset) -> xr.DataArray:
    da = ds["temp"] if "temp" in ds.data_vars else ds[list(ds.data_vars)[0]]
    if "st_ocean" in da.dims:
        da = da.isel(st_ocean=0, drop=True)
    rename_map = {}
    for d in da.dims:
        if d == "Time":
            rename_map[d] = "time"
        elif d == "yt_ocean":
            rename_map[d] = "lat"
        elif d == "xt_ocean":
            rename_map[d] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    return da


def run_part1() -> pd.DataFrame:
    print("\n" + "=" * 80, flush=True)
    print("RUNNING PART 1: Multi-Model Test-Set Evaluation (OFAM Holdout 2011-2014)", flush=True)
    print("=" * 80, flush=True)

    # First load ground truth and coarse input from reference test file
    ref_path = RUNS_DIR / "flow_sr" / "evaluation" / "full_test_samples.nc"
    if not ref_path.exists():
        ref_path = RUNS_DIR / "flow_sr" / "evaluation" / "full_test_samples_ab3_pc_75step.nc"

    # Test set holdout years: strictly 2011-2014 (1,461 daily time steps)
    TEST_YEARS = [2011, 2012, 2013, 2014]

    with xr.open_dataset(ref_path) as ds_ref:
        da_tgt = ds_ref["sst_target"].sel(time=ds_ref.time.dt.year.isin(TEST_YEARS)).load()
        da_coarse_lr = ds_ref["sst_coarse"].sel(time=ds_ref.time.dt.year.isin(TEST_YEARS)).load()
        fine_lat = da_tgt.lat.values
        fine_lon = da_tgt.lon.values

    # Compute Ground Truth reference climatologies
    clim_tgt = da_tgt.mean(dim="time").compute()
    clim_tgt_season = da_tgt.groupby("time.season").mean(dim="time").compute()
    txx_tgt = da_tgt.groupby("time.year").max(dim="time").mean(dim="year").compute()
    thresh_tgt = da_tgt.quantile(0.90, dim="time").compute()
    mhw_tgt = (da_tgt > thresh_tgt).mean(dim="time").compute()

    # Compute Coarse baseline metrics
    clim_coarse_lr = da_coarse_lr.mean(dim="time").compute()
    clim_coarse = interp_coarse_to_fine(clim_coarse_lr, fine_lat, fine_lon)
    clim_coarse_season = interp_coarse_to_fine(da_coarse_lr.groupby("time.season").mean(dim="time").compute(), fine_lat, fine_lon)
    txx_coarse = interp_coarse_to_fine(da_coarse_lr.groupby("time.year").max(dim="time").mean(dim="year").compute(), fine_lat, fine_lon)
    thresh_coarse_lr = da_coarse_lr.quantile(0.90, dim="time").compute()
    mhw_coarse = interp_coarse_to_fine((da_coarse_lr > thresh_coarse_lr).mean(dim="time").compute(), fine_lat, fine_lon)

    metrics_dict = {
        "Coarse Input": {
            "Annual RMSE (°C)": spatial_rmse(clim_coarse, clim_tgt),
            "Annual Bias (°C)": spatial_bias(clim_coarse, clim_tgt),
            "Annual MAE (°C)": spatial_mae(clim_coarse, clim_tgt),
            "Pattern Corr": spatial_corr(clim_coarse, clim_tgt),
            "TXx Error (°C)": spatial_mae(txx_coarse, txx_tgt),
            "MHW Freq RMSE": spatial_rmse(mhw_coarse, mhw_tgt),
            "DJF RMSE (°C)": spatial_rmse(clim_coarse_season.sel(season="DJF"), clim_tgt_season.sel(season="DJF")),
            "MAM RMSE (°C)": spatial_rmse(clim_coarse_season.sel(season="MAM"), clim_tgt_season.sel(season="MAM")),
            "JJA RMSE (°C)": spatial_rmse(clim_coarse_season.sel(season="JJA"), clim_tgt_season.sel(season="JJA")),
            "SON RMSE (°C)": spatial_rmse(clim_coarse_season.sel(season="SON"), clim_tgt_season.sel(season="SON")),
        }
    }

    model_clims = {}
    model_clims_season = {}
    valid_test_models = []

    for model in MODEL_KEYS:
        test_path = RUNS_DIR / model / "evaluation" / "full_test_samples.nc"
        if not test_path.exists():
            test_path = RUNS_DIR / model / "evaluation" / "full_test_samples_ab3_pc_75step.nc"
        if test_path.exists():
            print(f"  • Evaluating {model:<36} ...", end=" ", flush=True)
            with xr.open_dataset(test_path) as ds_m:
                da_gen = ds_m["sst_generated"].sel(time=ds_m.time.dt.year.isin(TEST_YEARS)).load()
                clim_m = da_gen.mean(dim="time").compute()
                clim_m_s = da_gen.groupby("time.season").mean(dim="time").compute()
                txx_m = da_gen.groupby("time.year").max(dim="time").mean(dim="year").compute()
                mhw_m = (da_gen > thresh_tgt).mean(dim="time").compute()

                model_clims[model] = clim_m
                model_clims_season[model] = clim_m_s
                valid_test_models.append(model)

                m_name = MODEL_REGISTRY.get(model, model)
                metrics_dict[m_name] = {
                    "Annual RMSE (°C)": spatial_rmse(clim_m, clim_tgt),
                    "Annual Bias (°C)": spatial_bias(clim_m, clim_tgt),
                    "Annual MAE (°C)": spatial_mae(clim_m, clim_tgt),
                    "Pattern Corr": spatial_corr(clim_m, clim_tgt),
                    "TXx Error (°C)": spatial_mae(txx_m, txx_tgt),
                    "MHW Freq RMSE": spatial_rmse(mhw_m, mhw_tgt),
                    "DJF RMSE (°C)": spatial_rmse(clim_m_s.sel(season="DJF"), clim_tgt_season.sel(season="DJF")),
                    "MAM RMSE (°C)": spatial_rmse(clim_m_s.sel(season="MAM"), clim_tgt_season.sel(season="MAM")),
                    "JJA RMSE (°C)": spatial_rmse(clim_m_s.sel(season="JJA"), clim_tgt_season.sel(season="JJA")),
                    "SON RMSE (°C)": spatial_rmse(clim_m_s.sel(season="SON"), clim_tgt_season.sel(season="SON")),
                }
                print(f"RMSE={metrics_dict[m_name]['Annual RMSE (°C)']:.4f}°C", flush=True)

    # Plot Annual Climatology Biases
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    bias_coarse = clim_coarse - clim_tgt
    norm_bias = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)

    im0 = bias_coarse.plot(ax=axes_flat[0], cmap="RdBu_r", norm=norm_bias, add_colorbar=False)
    axes_flat[0].set_title(f"Coarse Input (Bilinear)\nRMSE = {metrics_dict['Coarse Input']['Annual RMSE (°C)']:.4f}°C | Bias = {metrics_dict['Coarse Input']['Annual Bias (°C)']:.4f}°C", weight="bold")

    for i, model in enumerate(valid_test_models):
        ax = axes_flat[i + 1]
        m_name = MODEL_REGISTRY.get(model, model)
        bias_m = model_clims[model] - clim_tgt
        bias_m.plot(ax=ax, cmap="RdBu_r", norm=norm_bias, add_colorbar=False)
        ax.set_title(f"{m_name}\nRMSE = {metrics_dict[m_name]['Annual RMSE (°C)']:.4f}°C | Bias = {metrics_dict[m_name]['Annual Bias (°C)']:.4f}°C", weight="bold")

    for ax in axes_flat:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("SST Climatology Bias (°C) relative to OFAM Ground Truth", fontsize=11)
    plt.suptitle("Test Set (OFAM Holdout 2011–2014): Multi-Model Climatology Bias Maps", fontsize=16, weight="bold", y=0.98)
    plt.subplots_adjust(bottom=0.10, top=0.92, hspace=0.25, wspace=0.10)
    fig_path = FIGURES_DIR / "part1_multimodel_test_bias_maps.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig_path}", flush=True)

    # Plot Seasonal Biases
    fig, axes = plt.subplots(4, len(valid_test_models) + 1, figsize=(3 * (len(valid_test_models) + 1), 16), sharex=True, sharey=True)
    norm_s = TwoSlopeNorm(vmin=-2.0, vcenter=0, vmax=2.0)

    for r, s in enumerate(SEASONS):
        b_c_s = clim_coarse_season.sel(season=s) - clim_tgt_season.sel(season=s)
        b_c_s.plot(ax=axes[r, 0], cmap="RdBu_r", norm=norm_s, add_colorbar=False)
        if r == 0:
            axes[r, 0].set_title("Coarse Input", weight="bold")
        axes[r, 0].set_ylabel(f"{s}\nLatitude (°N)", weight="bold")

        for c, model in enumerate(valid_test_models):
            ax = axes[r, c + 1]
            b_m_s = model_clims_season[model].sel(season=s) - clim_tgt_season.sel(season=s)
            im = b_m_s.plot(ax=ax, cmap="RdBu_r", norm=norm_s, add_colorbar=False)
            if r == 0:
                ax.set_title(MODEL_REGISTRY.get(model, model), fontsize=9.5, weight="bold")
            if r == 3:
                ax.set_xlabel("Longitude (°E)")
            else:
                ax.set_xlabel("")

    cbar_ax = fig.add_axes([0.25, 0.03, 0.50, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Seasonal SST Bias (°C) vs OFAM Target", fontsize=11)
    plt.suptitle("Test Set: Multi-Model Seasonal Climatology Bias Breakdown", fontsize=16, weight="bold", y=0.99)
    plt.subplots_adjust(bottom=0.07, top=0.95, hspace=0.18, wspace=0.08)
    fig_path = FIGURES_DIR / "part1_multimodel_seasonal_biases.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig_path}", flush=True)

    df_p1 = pd.DataFrame(metrics_dict).T.round(4)
    print("\nPART 1 METRICS TABLE:", flush=True)
    print(df_p1.to_string(), flush=True)
    return df_p1


def run_part2() -> pd.DataFrame:
    print("\n" + "=" * 80, flush=True)
    print("RUNNING PART 2: Historical Climatology Validation (ACCESS-CM2 1980-1989 vs OFAM)", flush=True)
    print("=" * 80, flush=True)

    with xr.open_dataset(OFAM_FILE) as ds_ofam_raw:
        da_ofam = prepare_ofam(ds_ofam_raw)
        years_ofam = np.unique(da_ofam.time.dt.year.values)
        common_years = np.array([y for y in range(1980, 1990) if y in years_ofam])
        da_ofam_sub = da_ofam.sel(time=da_ofam.time.dt.year.isin(common_years)).load()

    clim_ofam = da_ofam_sub.mean(dim="time").compute()
    clim_ofam_season = da_ofam_sub.groupby("time.season").mean(dim="time").compute()
    fine_lat = clim_ofam.lat.values
    fine_lon = clim_ofam.lon.values

    # Load and evaluate coarse GCM baseline
    ref_hist_pattern = str(RUNS_DIR / "flow_sr" / "access_cm2_converted" / "historical_*.nc")
    with xr.open_dataset(glob.glob(ref_hist_pattern)[0]) as ds_h_ref:
        da_coarse_lr = ds_h_ref["sst_coarse"].sel(time=ds_h_ref.time.dt.year.isin(common_years)).load()
        clim_coarse_lr = da_coarse_lr.mean(dim="time").compute()
        clim_coarse = interp_coarse_to_fine(clim_coarse_lr, fine_lat, fine_lon)
        clim_coarse_season = interp_coarse_to_fine(da_coarse_lr.groupby("time.season").mean(dim="time").compute(), fine_lat, fine_lon)

    rmse_base = spatial_rmse(clim_coarse, clim_ofam)
    hist_metrics = {
        "Raw Coarse ACCESS-CM2": {
            "Annual RMSE (°C)": rmse_base,
            "Annual Bias (°C)": spatial_bias(clim_coarse, clim_ofam),
            "Pattern Corr": spatial_corr(clim_coarse, clim_ofam),
            "Error Reduction (%)": 0.0,
            "DJF RMSE (°C)": spatial_rmse(clim_coarse_season.sel(season="DJF"), clim_ofam_season.sel(season="DJF")),
            "MAM RMSE (°C)": spatial_rmse(clim_coarse_season.sel(season="MAM"), clim_ofam_season.sel(season="MAM")),
            "JJA RMSE (°C)": spatial_rmse(clim_coarse_season.sel(season="JJA"), clim_ofam_season.sel(season="JJA")),
            "SON RMSE (°C)": spatial_rmse(clim_coarse_season.sel(season="SON"), clim_ofam_season.sel(season="SON")),
        }
    }

    hist_clims = {}
    valid_hist_models = []

    for model in MODEL_KEYS:
        pattern = str(RUNS_DIR / model / "access_cm2_converted" / "historical_*.nc")
        matching = glob.glob(pattern)
        if matching:
            print(f"  • Evaluating historical {model:<36} ...", end=" ", flush=True)
            with xr.open_dataset(matching[0]) as ds_h:
                da_down = ds_h["sst_downscaled"].sel(time=ds_h.time.dt.year.isin(common_years)).load()
                clim_m = da_down.mean(dim="time").compute()
                clim_m_s = da_down.groupby("time.season").mean(dim="time").compute()
                hist_clims[model] = clim_m
                valid_hist_models.append(model)

                m_name = MODEL_REGISTRY.get(model, model)
                rmse_m = spatial_rmse(clim_m, clim_ofam)
                err_reduc = ((rmse_base - rmse_m) / rmse_base) * 100.0

                hist_metrics[m_name] = {
                    "Annual RMSE (°C)": rmse_m,
                    "Annual Bias (°C)": spatial_bias(clim_m, clim_ofam),
                    "Pattern Corr": spatial_corr(clim_m, clim_ofam),
                    "Error Reduction (%)": err_reduc,
                    "DJF RMSE (°C)": spatial_rmse(clim_m_s.sel(season="DJF"), clim_ofam_season.sel(season="DJF")),
                    "MAM RMSE (°C)": spatial_rmse(clim_m_s.sel(season="MAM"), clim_ofam_season.sel(season="MAM")),
                    "JJA RMSE (°C)": spatial_rmse(clim_m_s.sel(season="JJA"), clim_ofam_season.sel(season="JJA")),
                    "SON RMSE (°C)": spatial_rmse(clim_m_s.sel(season="SON"), clim_ofam_season.sel(season="SON")),
                }
                print(f"RMSE={rmse_m:.4f}°C (Reduction: {err_reduc:+.1f}%)", flush=True)

    # Plot Historical Biases
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    bias_hist_coarse = clim_coarse - clim_ofam
    norm_hist_bias = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)

    im0 = bias_hist_coarse.plot(ax=axes_flat[0], cmap="RdBu_r", norm=norm_hist_bias, add_colorbar=False)
    axes_flat[0].set_title(f"Raw Coarse ACCESS-CM2\nRMSE = {rmse_base:.4f}°C | Bias = {spatial_bias(clim_coarse, clim_ofam):.4f}°C", weight="bold")

    for i, model in enumerate(valid_hist_models):
        ax = axes_flat[i + 1]
        m_name = MODEL_REGISTRY.get(model, model)
        bias_h_m = hist_clims[model] - clim_ofam
        bias_h_m.plot(ax=ax, cmap="RdBu_r", norm=norm_hist_bias, add_colorbar=False)
        ax.set_title(f"{m_name}\nRMSE = {hist_metrics[m_name]['Annual RMSE (°C)']:.4f}°C (Reduc: {hist_metrics[m_name]['Error Reduction (%)']:+.1f}%)", weight="bold")

    for ax in axes_flat:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Historical SST Bias (°C) relative to OFAM Ground Truth (1980–1989)", fontsize=11)
    plt.suptitle("Historical Climatology Validation (ACCESS-CM2 1980–1989 vs OFAM)", fontsize=16, weight="bold", y=0.98)
    plt.subplots_adjust(bottom=0.10, top=0.92, hspace=0.25, wspace=0.10)
    fig_path = FIGURES_DIR / "part2_multimodel_historical_biases.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig_path}", flush=True)

    df_p2 = pd.DataFrame(hist_metrics).T.round(4)
    print("\nPART 2 METRICS TABLE:", flush=True)
    print(df_p2.to_string(), flush=True)
    return df_p2


def run_part3() -> pd.DataFrame:
    print("\n" + "=" * 80, flush=True)
    print("RUNNING PART 3: Climate Change Signals & MHW Projections (SSP585 2080-2089 vs Historical)", flush=True)
    print("=" * 80, flush=True)

    # First load coarse GCM future & historical reference
    ref_f_pattern = str(RUNS_DIR / "flow_sr" / "access_cm2_converted" / "future_*.nc")
    ref_h_pattern = str(RUNS_DIR / "flow_sr" / "access_cm2_converted" / "historical_*.nc")

    with xr.open_dataset(glob.glob(ref_f_pattern)[0]) as ds_f_ref, xr.open_dataset(glob.glob(ref_h_pattern)[0]) as ds_h_ref:
        da_coarse_f = ds_f_ref["sst_coarse"].load()
        da_coarse_h = ds_h_ref["sst_coarse"].load()
        fine_lat = ds_f_ref["sst_downscaled"].lat.values
        fine_lon = ds_f_ref["sst_downscaled"].lon.values

    clim_coarse_f = interp_coarse_to_fine(da_coarse_f.mean(dim="time").compute(), fine_lat, fine_lon)
    clim_coarse_h = interp_coarse_to_fine(da_coarse_h.mean(dim="time").compute(), fine_lat, fine_lon)
    delta_coarse = clim_coarse_f - clim_coarse_h

    clim_coarse_f_s = interp_coarse_to_fine(da_coarse_f.groupby("time.season").mean(dim="time").compute(), fine_lat, fine_lon)
    clim_coarse_h_s = interp_coarse_to_fine(da_coarse_h.groupby("time.season").mean(dim="time").compute(), fine_lat, fine_lon)
    delta_coarse_s = clim_coarse_f_s - clim_coarse_h_s

    thresh_coarse_h = da_coarse_h.quantile(0.90, dim="time").compute()
    mhw_coarse_h = interp_coarse_to_fine((da_coarse_h > thresh_coarse_h).mean(dim="time").compute(), fine_lat, fine_lon)
    mhw_coarse_f = interp_coarse_to_fine((da_coarse_f > thresh_coarse_h).mean(dim="time").compute(), fine_lat, fine_lon)
    mhw_delta_coarse = mhw_coarse_f - mhw_coarse_h

    mean_c_w = float(np.nanmean(delta_coarse))
    cc_summary = {
        "Raw Coarse ACCESS-CM2": {
            "Annual Mean ΔT (°C)": mean_c_w,
            "Max Pixel ΔT (°C)": float(np.nanmax(delta_coarse)),
            "DJF ΔT (°C)": float(np.nanmean(delta_coarse_s.sel(season="DJF"))),
            "MAM ΔT (°C)": float(np.nanmean(delta_coarse_s.sel(season="MAM"))),
            "JJA ΔT (°C)": float(np.nanmean(delta_coarse_s.sel(season="JJA"))),
            "SON ΔT (°C)": float(np.nanmean(delta_coarse_s.sel(season="SON"))),
            "Warming Diff (°C)": 0.0,
            "Underestimation (%)": 0.0,
            "Mean ΔMHW Fraction": float(np.nanmean(mhw_delta_coarse)),
        }
    }

    deltas_dict = {}
    diffs_dict = {}
    mhw_deltas_dict = {}
    valid_fut_models = []

    for model in MODEL_KEYS:
        f_pattern = str(RUNS_DIR / model / "access_cm2_converted" / "future_*.nc")
        h_pattern = str(RUNS_DIR / model / "access_cm2_converted" / "historical_*.nc")
        f_matching = glob.glob(f_pattern)
        h_matching = glob.glob(h_pattern)

        if f_matching and h_matching:
            print(f"  • Evaluating climate signals {model:<36} ...", end=" ", flush=True)
            with xr.open_dataset(f_matching[0]) as ds_f, xr.open_dataset(h_matching[0]) as ds_h:
                da_f = ds_f["sst_downscaled"].load()
                da_h = ds_h["sst_downscaled"].load()

                clim_f = da_f.mean(dim="time").compute()
                clim_h = da_h.mean(dim="time").compute()
                delta_m = clim_f - clim_h
                diff_m = delta_m - delta_coarse

                clim_f_s = da_f.groupby("time.season").mean(dim="time").compute()
                clim_h_s = da_h.groupby("time.season").mean(dim="time").compute()
                delta_m_s = clim_f_s - clim_h_s

                thresh_h = da_h.quantile(0.90, dim="time").compute()
                mhw_h = (da_h > thresh_h).mean(dim="time").compute()
                mhw_f = (da_f > thresh_h).mean(dim="time").compute()
                mhw_delta_m = mhw_f - mhw_h

                deltas_dict[model] = delta_m
                diffs_dict[model] = diff_m
                mhw_deltas_dict[model] = mhw_delta_m
                valid_fut_models.append(model)

                m_name = MODEL_REGISTRY.get(model, model)
                mean_m_w = float(np.nanmean(delta_m))
                mean_diff = float(np.nanmean(diff_m))
                underest = (mean_diff / mean_c_w) * 100.0

                cc_summary[m_name] = {
                    "Annual Mean ΔT (°C)": mean_m_w,
                    "Max Pixel ΔT (°C)": float(np.nanmax(delta_m)),
                    "DJF ΔT (°C)": float(np.nanmean(delta_m_s.sel(season="DJF"))),
                    "MAM ΔT (°C)": float(np.nanmean(delta_m_s.sel(season="MAM"))),
                    "JJA ΔT (°C)": float(np.nanmean(delta_m_s.sel(season="JJA"))),
                    "SON ΔT (°C)": float(np.nanmean(delta_m_s.sel(season="SON"))),
                    "Warming Diff (°C)": mean_diff,
                    "Underestimation (%)": underest,
                    "Mean ΔMHW Fraction": float(np.nanmean(mhw_delta_m)),
                }
                print(f"Mean ΔT={mean_m_w:.3f}°C (Diff={mean_diff:+.3f}°C, Underest={underest:+.1f}%)", flush=True)

    # Figure 1: Warming Signals ΔT
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    vmin_w = 1.0
    vmax_w = 4.5

    im0 = delta_coarse.plot(ax=axes_flat[0], cmap="RdYlBu_r", vmin=vmin_w, vmax=vmax_w, add_colorbar=False)
    axes_flat[0].set_title(f"Raw Coarse ACCESS-CM2 ΔT\nMean = {mean_c_w:.3f}°C", weight="bold")

    for i, model in enumerate(valid_fut_models):
        ax = axes_flat[i + 1]
        m_name = MODEL_REGISTRY.get(model, model)
        deltas_dict[model].plot(ax=ax, cmap="RdYlBu_r", vmin=vmin_w, vmax=vmax_w, add_colorbar=False)
        ax.set_title(f"{m_name} ΔT\nMean = {cc_summary[m_name]['Annual Mean ΔT (°C)']:.3f}°C (Diff = {cc_summary[m_name]['Warming Diff (°C)']:+.3f}°C)", weight="bold")

    for ax in axes_flat:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Climate Change Warming Signal ΔT (°C) [2080–2089 minus 1980–1989]", fontsize=11)
    plt.suptitle("Multi-Model Climate Change Projections: SSP585 (2080–2089) − Historical (1980–1989)", fontsize=16, weight="bold", y=0.98)
    plt.subplots_adjust(bottom=0.10, top=0.92, hspace=0.25, wspace=0.10)
    fig_path = FIGURES_DIR / "part3_multimodel_warming_signals.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig_path}", flush=True)

    # Figure 2: Warming Differences
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    axes_flat[0].axis("off")
    axes_flat[0].text(
        0.5,
        0.5,
        "Warming Discrepancy\nΔT_downscaled − ΔT_coarse\n\nRed = Amplified Warming\nBlue = Dampened Warming",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        color="#102a43",
    )
    norm_diff = TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8)

    for i, model in enumerate(valid_fut_models):
        ax = axes_flat[i + 1]
        m_name = MODEL_REGISTRY.get(model, model)
        im = diffs_dict[model].plot(ax=ax, cmap="RdBu_r", norm=norm_diff, add_colorbar=False)
        ax.set_title(f"{m_name}\nMean Diff = {cc_summary[m_name]['Warming Diff (°C)']:+.3f}°C ({cc_summary[m_name]['Underestimation (%)']:+.1f}%)", weight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Warming Difference (Downscaled − Coarse ΔT, °C)", fontsize=11)
    plt.suptitle("Multi-Model Warming Discrepancy Maps (Downscaled AI vs Raw Coarse ACCESS-CM2)", fontsize=16, weight="bold", y=0.98)
    plt.subplots_adjust(bottom=0.10, top=0.92, hspace=0.25, wspace=0.10)
    fig_path = FIGURES_DIR / "part3_multimodel_warming_differences.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig_path}", flush=True)

    # Figure 3: MHW Projections
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    im0 = mhw_delta_coarse.plot(ax=axes_flat[0], cmap="YlOrRd", vmin=0, vmax=0.9, add_colorbar=False)
    axes_flat[0].set_title(f"Raw Coarse ACCESS-CM2 ΔMHW\nMean Δ = +{float(np.nanmean(mhw_delta_coarse)):.3f}", weight="bold")

    for i, model in enumerate(valid_fut_models):
        ax = axes_flat[i + 1]
        m_name = MODEL_REGISTRY.get(model, model)
        mhw_deltas_dict[model].plot(ax=ax, cmap="YlOrRd", vmin=0, vmax=0.9, add_colorbar=False)
        ax.set_title(f"{m_name}\nMean Δ = +{cc_summary[m_name]['Mean ΔMHW Fraction']:.3f}", weight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    axes_flat[0].set_xlabel("Longitude (°E)")
    axes_flat[0].set_ylabel("Latitude (°N)")

    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
    cbar = fig.colorbar(im0, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Change in MHW Exceedance Frequency (Δ Fraction of Days exceeding Historical 90th %ile)", fontsize=11)
    plt.suptitle("Multi-Model Marine Heatwave (MHW) Projections: End-of-Century Frequency Shift", fontsize=16, weight="bold", y=0.98)
    plt.subplots_adjust(bottom=0.10, top=0.92, hspace=0.25, wspace=0.10)
    fig_path = FIGURES_DIR / "part3_multimodel_mhw_projections.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig_path}", flush=True)

    df_p3 = pd.DataFrame(cc_summary).T.round(3)
    print("\nPART 3 METRICS TABLE:", flush=True)
    print(df_p3.to_string(), flush=True)
    return df_p3


def main():
    df_p1 = run_part1()
    df_p2 = run_part2()
    df_p3 = run_part3()

    # Save summary tables to CSV
    df_p1.to_csv(FIGURES_DIR / "part1_multimodel_testset_metrics.csv")
    df_p2.to_csv(FIGURES_DIR / "part2_multimodel_historical_metrics.csv")
    df_p3.to_csv(FIGURES_DIR / "part3_multimodel_climate_change_metrics.csv")

    print("\n" + "=" * 80, flush=True)
    print("ALL FIGURES AND EVALUATION TABLES SUCCESSFULLY GENERATED AND SAVED TO figures/!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
