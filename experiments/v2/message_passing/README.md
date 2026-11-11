# AOI message-passing depth study

Report Appendix B, Figure 22. The complete grid crosses coarse layers `{1,2,3,4}` with fine layers `{1,2,3,4,6}`: 20 runs, including a fresh `c2_f3` reference. Width 64, T=0.3 m, lambda=0.01 and one cross-scale fusion stage remain fixed.

All candidates use the same four-coarse/six-fine-hop graph: 362 coarse nodes, 3,271 fine nodes and 2,314 unchanged AOI targets. Use fold 1 and seed 42. Evaluate all 20 validation-selected checkpoints on the same held-out events.

The original `c4_f4` trial had the lowest validation RMSE (0.22989 m, epoch 252). The adopted `c2_f2` model had 140,481 parameters and validation RMSE 0.23817 m at epoch 107, with lower computational demand.

Follow [reproduction stage 3](../../../docs/reproduction.md#3-all-20-message-passing-configurations). Plot with `python scripts/plot_depth_heatmaps.py` after all evaluations finish.
