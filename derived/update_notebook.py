#!/usr/bin/env python3
"""Update testset_evaluation.ipynb with exact 2011-2014 filtering in Part 1."""

import json
from pathlib import Path

nb_path = Path("/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling/testset_evaluation.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("id") == "part1_load_data":
        cell["source"] = [
            "# Load holdout test-set evaluation files across all registered models\n",
            "# Test set holdout years: strictly 2011-2014 (1,461 daily time steps)\n",
            "TEST_YEARS = [2011, 2012, 2013, 2014]\n",
            "test_gen_list = []\n",
            "valid_test_models = []\n",
            "ds_ref = None\n",
            "\n",
            "for model in MODEL_KEYS:\n",
            "    test_path = os.path.join(RUNS_DIR, model, 'evaluation', 'full_test_samples.nc')\n",
            "    if not os.path.exists(test_path):\n",
            "        test_path = os.path.join(RUNS_DIR, model, 'evaluation', 'full_test_samples_ab3_pc_75step.nc')\n",
            "    \n",
            "    if os.path.exists(test_path):\n",
            "        ds_m = xr.open_dataset(test_path)\n",
            "        da_m = ds_m['sst_generated'].sel(time=ds_m.time.dt.year.isin(TEST_YEARS))\n",
            "        test_gen_list.append(da_m)\n",
            "        valid_test_models.append(model)\n",
            "        if ds_ref is None:\n",
            "            ds_ref = ds_m\n",
            "        print(f'Loaded test samples for: {model} (time={da_m.sizes[\"time\"]})')\n",
            "    else:\n",
            "        print(f'Warning: Test samples file not found for: {model}')\n",
            "\n",
            "# Concatenate all model predictions along a new \"model_name\" dimension\n",
            "da_test_models = xr.concat(test_gen_list, dim='model_name')\n",
            "da_test_models['model_name'] = valid_test_models\n",
            "\n",
            "da_test_tgt = ds_ref['sst_target'].sel(time=ds_ref.time.dt.year.isin(TEST_YEARS))\n",
            "da_test_coarse_lr = ds_ref['sst_coarse'].sel(time=ds_ref.time.dt.year.isin(TEST_YEARS))\n",
            "fine_lat = da_test_tgt.lat.values\n",
            "fine_lon = da_test_tgt.lon.values\n",
            "\n",
            "print(f'\\nCombined Test-Set DataArray shape: {da_test_models.shape} (models, time, lat, lon)')"
        ]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully updated {nb_path} with strict 2011-2014 filter!")
