# Documentation map

Start with the top-level [README](../README.md) for the shortest reproducible
workflow.  Use this page to find the detailed method or operational note that
matches the experiment you are changing.

| Topic | Document | Use it for |
|---|---|---|
| GAN formulation and ablations | [GAN experiment guide](gan_experiment_guide.md) | Loss equations, configurable weights, safe sensitivity experiments, critic choices, and launch checklist |
| Exact GAN/FiLM interpretation | [GAN losses and lag FiLM](gan_losses_and_lag_film.md) | Current implementation details and measured loss/gate contributions |
| ACCESS-CM2 deployment | [ACCESS-CM2 operational inference](access_cm2_operational_inference.md) | Grid conversion, masks, periods, inference, and validation |
| NOAA 0.05° transfer | [NOAA transfer methods](noaa_5km_frozen_trunk_1024_methods.md) | Failed coastline design, replacement architecture, trainable layers, and inference |
| Residual-memory AR | [Residual-memory Flow-AR](flow_ar_residual_memory.md) | Memory pathway, FiLM cap, coarse projection, and training |
| Legacy AR audit | [Flow-AR rollout audit](flow_ar_rollout_audit.md) | Historical bug, corrected date pairing, and one-year rollout contract |
| Full engineering ledger | [Project plan](../plan.md) | Data forensics, decisions, completed jobs, and exhaustive test matrix |
| PBS launchers | [Job catalogue](../jobs/README.md) | Which job is active, diagnostic, continuation, inference, or rendering |

Generated scientific products live outside this documentation tree:

- `runs/<experiment>/`: weights, checkpoints, callbacks, and inference files;
- `figures/`: comparison figures and animations;
- `figures/climate_change/`: verified historical/future signal figures and CSV tables;
- `unsuccessful_experiments/`: quarantined provenance for scientifically rejected runs;
- `reports/`: machine-readable validation and evaluation reports; and
- `presentation/`: editable slide decks and their generation manifests;
- `SST_Downscaling_Skeleton_Paper.docx`: editable Word manuscript; and
- `paper/`: the manuscript source, reproducible builder, and validation report.

Raw NetCDF data, model weights, and production outputs are intentionally not
tracked by Git.  A clone reproduces code and environment, not those large
artifacts.
