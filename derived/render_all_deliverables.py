#!/usr/bin/env python3
"""Run all figure updates and render web-optimized MP4 and GIF animations."""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling")
PYTHON = BASE_DIR / ".pixi" / "envs" / "default" / "bin" / "python"

print("=================================================================")
print("1. Updating testset_evaluation.ipynb with 2011-2014 filter...")
subprocess.run([str(PYTHON), "derived/update_notebook.py"], cwd=str(BASE_DIR), check=True)

print("\n=================================================================")
print("2. Re-generating Multi-Model Evaluation Figures (Parts 1, 2, 3)...")
subprocess.run([str(PYTHON), "derived/generate_multimodel_evaluation_figures.py"], cwd=str(BASE_DIR), check=True)

print("\n=================================================================")
print("3. Rendering Autoregressive Flow-AR Animation (MP4 & compact preview GIF)...")
# MP4 (Full 364 frames, 10 fps)
subprocess.run([
    str(PYTHON), "figures/animate_flow_ar_sst.py",
    "--output", "figures/flow_ar_sst_comparison.mp4",
    "--fps", "10",
    "--stride", "1",
    "--dpi", "100"
], cwd=str(BASE_DIR), check=True)

# Preview GIF (60 frames, 6 fps, 80 dpi, ~8MB)
subprocess.run([
    str(PYTHON), "figures/animate_flow_ar_sst.py",
    "--output", "figures/flow_ar_sst_preview.gif",
    "--fps", "6",
    "--stride", "6",
    "--dpi", "80"
], cwd=str(BASE_DIR), check=True)

print("\n=================================================================")
print("4. Rendering GAN-SR Animation (MP4 & compact preview GIF)...")
# MP4 (100 frames, 8 fps)
subprocess.run([
    str(PYTHON), "figures/animate_gan_sr_sst.py",
    "--output", "figures/gan_sr_sst_comparison.mp4",
    "--fps", "8",
    "--stride", "1",
    "--max-frames", "120",
    "--dpi", "100"
], cwd=str(BASE_DIR), check=True)

# Preview GIF (50 frames, 5 fps, 80 dpi)
subprocess.run([
    str(PYTHON), "figures/animate_gan_sr_sst.py",
    "--output", "figures/gan_sr_sst_preview.gif",
    "--fps", "5",
    "--stride", "2",
    "--max-frames", "50",
    "--dpi", "80"
], cwd=str(BASE_DIR), check=True)

print("\n=================================================================")
print("5. Rendering Multi-Model Intercomparison Animation (MP4 & compact preview GIF)...")
# MP4 (100 frames, 8 fps)
subprocess.run([
    str(PYTHON), "figures/animate_flow_gan_sr_sst.py",
    "--output", "figures/flow_gan_sr_sst_comparison.mp4",
    "--fps", "8",
    "--stride", "1",
    "--max-frames", "120",
    "--dpi", "100"
], cwd=str(BASE_DIR), check=True)

# Preview GIF (50 frames, 5 fps, 80 dpi)
subprocess.run([
    str(PYTHON), "figures/animate_flow_gan_sr_sst.py",
    "--output", "figures/flow_gan_sr_sst_preview.gif",
    "--fps", "5",
    "--stride", "2",
    "--max-frames", "50",
    "--dpi", "80"
], cwd=str(BASE_DIR), check=True)

print("\n=================================================================")
print("All Figures and Fast-Loading Animations Successfully Rendered!")
print("=================================================================")
