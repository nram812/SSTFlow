# ACCESS-CM2 conversion and downscaling

## Data-flow contract

ACCESS-CM2 processing is deliberately split into two programs:

1. `derived/convert_access_to_training_grid.py` creates
   `derived/sst_downscaling_access_converted.nc` on the model's established
   32×32 predictor grid.
2. `src/infer_access_cm2.py` validates that grid and mask, normalizes the
   converted SST with the stored OFAM training statistics, and samples the
   completed `flow_sr` EMA model. It performs **no spatial interpolation**.

The NetCDF inputs and all inference products are excluded from Git. The
conversion and inference programs, JSON configuration, environment lock file,
tests, and PBS launcher are tracked.

## Conversion method

The standalone conversion script implements the supplied ACCESS workflow:

- interpolate ACCESS `sst_raw` to `lat_lr`/`lon_lr` by nearest neighbour;
- calculate ACCESS seasonal climatology over the OFAM training dates;
- calculate the full ACCESS seasonal anomalies relative to that climatology;
- fill coastal anomaly gaps using linear `scipy.interpolate.griddata`, followed
  by nearest neighbour outside the valid-point convex hull;
- add the OFAM seasonal 32×32 SST climatology;
- restore the exact static `ocean_mask_lr`; and
- atomically write float32 `sst_lr(time, lat_lr, lon_lr)`.

The supplied converted file has 51,135 daily fields from 1960-01-01 through
2099-12-31, a 32×32 grid, and exactly 716 finite training-ocean cells per
probed day.

To recreate it (the existing file is protected unless `--overwrite` is given):

```bash
pixi run python derived/convert_access_to_training_grid.py \
  --access /path/to/ACCESS-CM2_1960-2099_sst_global_2deg_raw.nc \
  --training derived/sst_downscaling_f16.nc \
  --output derived/sst_downscaling_access_converted.nc
```

## Inference configuration

`configs/access_cm2_inference.json` is the authoritative operational record:

- input: `derived/sst_downscaling_access_converted.nc`, variable `sst_lr`;
- model: `runs/flow_sr/model_ema.pt`;
- solver: AB3/AM3 predictor-corrector (`ab3_pc`), 75 integration steps;
- historical period: 1980-01-01 through 1989-12-31 (3,653 days);
- future period: 2080-01-01 through 2089-12-31 (3,653 days); and
- deterministic seed: 42 plus the absolute converted-file time index.

These are two literal ten-calendar-year windows. In particular, an inclusive
1980–1990 interval would contain eleven years, while 2080–2099 would contain
twenty.

Run both periods on the H200 queue:

```bash
qsub jobs/infer_access_cm2_periods.pbs
```

Or run one period directly:

```bash
PYTHONPATH=src pixi run -e gpu python src/infer_access_cm2.py \
  --config configs/access_cm2_inference.json \
  --period-name historical --device cuda
```

## Validation, output, and restart behavior

Before inference, the program requires coordinate equality with the training
grid to `1e-6` degrees and exact agreement with the static 716-cell coarse
ocean mask for every batch. Normalization follows the training contract:
normalize with the training-only scalar mean/std, fill invalid cells with zero,
and provide the coarse mask as the second condition channel.

Each output contains `sst_downscaled(time, lat, lon)`, `sst_coarse(time,
lat_lr, lon_lr)`, both static masks, the exact dates, model/weight/solver
provenance, and coarse distribution-shift diagnostics. Writes are resumable in
`.partial.nc`; the file is atomically renamed only after every requested day is
complete. Noise depends on the absolute source index, so results do not change
with batch size or resumption.
