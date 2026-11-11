# AOI width comparison

Report Appendix B, Table 8 and Figure 20. Three MSE-only models use widths 64, 128 and 256 with two coarse and three fine message-passing layers. Width changes latent vectors and MLP hidden layers together. The original 1 km AOI, 2,314 target cells, fold 1 and seed 42 remain fixed.

The original width-64 model had 165,889 parameters, selected epoch 174, validation RMSE 0.23890 m and 204 total epochs. These are reference outcomes, not hard-coded selection targets. Width 64 was adopted.

Follow [reproduction stage 1](../../../docs/reproduction.md#1-aoi-width-comparison) and the [Gadi guide](../../../gadi/README.md). The runner evaluates the selected checkpoint when training completes. Keep its `width_evaluation.json`, normalization and checkpoint for loss calibration. Generate comparisons with `python scripts/compare_hidden_widths.py` after all three runs finish.
