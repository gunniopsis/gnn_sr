# Burnett River full-domain training and evaluation

Report Sections 3-4 and Appendices C-D. Train fresh weights on all 780,785 fine cells with width 64, two coarse/two fine layers, T=0.3 m and lambda=0.01. Fold 1 and seed 42 retain 9,475 training and 1,053 validation timesteps. Validation RMSE controls checkpoint selection, scheduling and patience 30.

`configs/c2_f2.toml` sets a hard maximum of 100 epochs. The original run selected epoch 95 at validation depth RMSE 0.2503278380152749 m. A reproduction selects its own minimum-RMSE checkpoint, without forcing that epoch or score.

Preflight, training and inference use `scripts/run_full_domain_experiment.py`; common-mask comparison uses `scripts/compare_full_domain_lsg.py`. The original deployment name `c2_f2_8h` is consolidated to `c2_f2` in this standalone release. Outputs and caches are isolated from historical runs.

Follow [reproduction stages 4-5](../../../docs/reproduction.md#4-fresh-full-domain-training). All 18 test events have 383 evaluated timesteps. Raw full-domain diagnostics use all cells; the report benchmark uses the supplied 769,709-cell fold-1 mask. Keep the exact training cache for LF parent projection and all timestep predictions for subsequent maps/timing analyses.
