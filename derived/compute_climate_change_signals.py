#!/usr/bin/env python3
"""Fast computation of climate change signals across all models using xarray."""

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

def interp_coarse_to_fine(da_coarse, fine_lat, fine_lon):
    lat_dim = "lat_lr" if "lat_lr" in da_coarse.dims else ("lat" if "lat" in da_coarse.dims else da_coarse.dims[-2])
    lon_dim = "lon_lr" if "lon_lr" in da_coarse.dims else ("lon" if "lon" in da_coarse.dims else da_coarse.dims[-1])
    return da_coarse.interp({lat_dim: fine_lat, lon_dim: fine_lon}, method="linear").rename({lat_dim: "lat", lon_dim: "lon"})

print("Computing Coarse GCM baseline climate change signal...", flush=True)
ref_f = glob.glob(str(RUNS_DIR / "flow_sr" / "access_cm2_converted" / "future_*.nc"))[0]
ref_h = glob.glob(str(RUNS_DIR / "flow_sr" / "access_cm2_converted" / "historical_*.nc"))[0]

with xr.open_dataset(ref_f) as ds_f, xr.open_dataset(ref_h) as ds_h:
    fine_lat = ds_f["sst_downscaled"].lat.values
    fine_lon = ds_f["sst_downscaled"].lon.values
    
    # Compute Coarse mean
    c_f_mean = ds_f["sst_coarse"].mean(dim="time").compute()
    c_h_mean = ds_h["sst_coarse"].mean(dim="time").compute()
    delta_coarse = interp_coarse_to_fine(c_f_mean - c_h_mean, fine_lat, fine_lon)
    
    c_f_s = ds_f["sst_coarse"].groupby("time.season").mean(dim="time").compute()
    c_h_s = ds_h["sst_coarse"].groupby("time.season").mean(dim="time").compute()
    delta_coarse_s = interp_coarse_to_fine(c_f_s - c_h_s, fine_lat, fine_lon)

mean_c_w = float(np.nanmean(delta_coarse))
print(f"Coarse ACCESS-CM2 Annual Mean ΔT: {mean_c_w:.3f}°C", flush=True)

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
    }
}

deltas_dict = {}
diffs_dict = {}
valid_models = []

for model in MODEL_KEYS:
    f_files = glob.glob(str(RUNS_DIR / model / "access_cm2_converted" / "future_*.nc"))
    h_files = glob.glob(str(RUNS_DIR / model / "access_cm2_converted" / "historical_*.nc"))
    if f_files and h_files:
        print(f"Processing model: {model} ...", end=" ", flush=True)
        with xr.open_dataset(f_files[0]) as ds_f, xr.open_dataset(h_files[0]) as ds_h:
            clim_f = ds_f["sst_downscaled"].mean(dim="time").compute()
            clim_h = ds_h["sst_downscaled"].mean(dim="time").compute()
            delta_m = clim_f - clim_h
            diff_m = delta_m - delta_coarse

            clim_f_s = ds_f["sst_downscaled"].groupby("time.season").mean(dim="time").compute()
            clim_h_s = ds_h["sst_downscaled"].groupby("time.season").mean(dim="time").compute()
            delta_m_s = clim_f_s - clim_h_s

            deltas_dict[model] = delta_m
            diffs_dict[model] = diff_m
            valid_models.append(model)

            mean_m_w = float(np.nanmean(delta_m))
            mean_diff = float(np.nanmean(diff_m))
            underest = (mean_diff / mean_c_w) * 100.0

            m_name = MODEL_REGISTRY.get(model, model)
            cc_summary[m_name] = {
                "Annual Mean ΔT (°C)": mean_m_w,
                "Max Pixel ΔT (°C)": float(np.nanmax(delta_m)),
                "DJF ΔT (°C)": float(np.nanmean(delta_m_s.sel(season="DJF"))),
                "MAM ΔT (°C)": float(np.nanmean(delta_m_s.sel(season="MAM"))),
                "JJA ΔT (°C)": float(np.nanmean(delta_m_s.sel(season="JJA"))),
                "SON ΔT (°C)": float(np.nanmean(delta_m_s.sel(season="SON"))),
                "Warming Diff (°C)": mean_diff,
                "Underestimation (%)": underest,
            }
            print(f"Mean ΔT={mean_m_w:.3f}°C (Diff={mean_diff:+.3f}°C, Underest={underest:+.1f}%)", flush=True)

df_cc = pd.DataFrame(cc_summary).T.round(3)
csv_path = FIGURES_DIR / "climate_change_signals_summary.csv"
df_cc.to_csv(csv_path)

print("\n" + "=" * 85, flush=True)
print("CLIMATE CHANGE SIGNAL (SSP585 2080-2089 vs Historical 1980-1989) SYNTHESIS TABLE:")
print("=" * 85, flush=True)
print(df_cc.to_string(), flush=True)

# Plot Figure: Multi-Model Warming Signals ΔT
fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)
axes_flat = axes.flatten()
vmin_w = 1.0
vmax_w = 4.5

im0 = delta_coarse.plot(ax=axes_flat[0], cmap="RdYlBu_r", vmin=vmin_w, vmax=vmax_w, add_colorbar=False)
axes_flat[0].set_title(f"Raw Coarse ACCESS-CM2 ΔT\nMean = {mean_c_w:.3f}°C", weight="bold")

for i, model in enumerate(valid_models):
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
fig_path1 = FIGURES_DIR / "part3_multimodel_warming_signals.png"
plt.savefig(fig_path1, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved Figure: {fig_path1}", flush=True)

# Plot Figure: Warming Differences
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

for i, model in enumerate(valid_models):
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
fig_path2 = FIGURES_DIR / "part3_multimodel_warming_differences.png"
plt.savefig(fig_path2, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved Figure: {fig_path2}", flush=True)

print("Done!", flush=True)
