# AOI composite-loss study

Report Appendix B, Figure 21. Eight temperatures `{0.001,0.003,0.01,0.03,0.1,0.3,0.5,1.0}` m cross five weights `{0.001,0.01,0.1,1,10}`. Include one fresh MSE control: 41 runs in `candidates.json`.

The initial 30 combinations are calibrated on training data only. All are retained and trained, plus the control; the ten higher-temperature runs extend that grid. Every run is evaluated on the same 18 held-out events using its validation-selected checkpoint. There is no finalist-only reduction.

All models use width 64, two coarse/three fine layers, the original AOI and training settings. The adopted setting is T=0.3 m and lambda=0.01 (original validation RMSE 0.23513 m, epoch 174).

Follow [reproduction stage 2](../../../docs/reproduction.md#2-loss-calibration-and-all-41-training-runs). Plot all eight temperatures using `python scripts/report_loss_experiments.py --stage all` followed by `python scripts/plot_loss_heatmaps.py --stage all`.
