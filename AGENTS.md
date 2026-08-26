# Repository maintenance notes

- Read `plan.md` before changing model, masking, normalization, split, or operational behavior. Keep its Part A/B task status synchronized with verified work.
- The raw 6.9 GB NetCDF file is immutable. Do not rewrite or add it to version control.
- Preserve the static ocean-mask contract: normalize first, fill missing cells with zero, explicitly provide mask channels, and use masked reductions.
- Training statistics must use only `train_date_ranges`; autoregressive pairs must never cross a range boundary.
- Keep all paths configurable. Resolve repository-relative paths through `common.load_config`.
- Use atomic writes for checkpoints, JSON, and NetCDF products. Preserve resumability and RNG state.
- New behavior needs a focused CPU test. DataLoader multiprocessing also needs verification on a normal login/PBS node because restricted sandboxes may forbid PyTorch resource-sharing sockets.
- Run `pixi run test`, the three CPU smoke tasks, `pixi run validate-data`, and the H200 smoke job before production training.
- Do not modify the reference repositories listed in `plan.md`.
