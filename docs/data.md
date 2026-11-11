# External data and model inputs

Obtain the exact Burnett River research dataset and, for baseline comparisons, the supplied LSG artifacts from the project/data owner. This package does not provide a verified public download URL. The training split, geometry and array order must match the supplied inputs; substituting another Burnett dataset is not an exact reproduction.

Place the files under the repository root as follows:

```text
reference_files/BurnettRV/
  BurnettRV_event_summary.csv
  Geometry_data/
    HECRAS_Geometry_data.npz
    Tuflow_Geometry_data.npz
  HD_model_data/
    High-fidelity/                 Paradise_<event>_002.npz
    Low-fidelity/                  BurnettRV_LFmodelB.<plan>.hdf
  Train_test_split_data/
    Train_test_split_ValidateOnGrp_1.npz
  HF_EOF_analysis/
    Categories_HFdata_ValidateOnGrp_1.npz
  Result_data/
    Validation_results.npz
```

The first four groups are required for GNN training/evaluation. The category mask and saved validation results are additionally required for common-mask comparison. Saved-LSG inference needs further LF/HF EOF and trained model files: the exact relative paths, sizes and SHA-256 hashes are enumerated in [the LSG input manifest](../experiments/v2/lsg_full_domain/input_manifest.json).

The catalogue identifies all 74 paired simulations and their groups. Preserve its row order and the saved 90/10 timestep partition. Do not regenerate the split from a seed. The code uses group/fold 1, giving 9,475 training, 1,053 validation and 18 held-out events. There are 383 evaluated hourly timesteps per held-out event after warm-up, spanning hours 48 through 430.

The catalogue's event identifier `e1971_his` occupies held-out position 15; the thesis Table 6 labels that position `e1971_real_slmax`. Keep the supplied machine identifier and catalogue order. Do not rename data files to match the display label.

HF NPZ events must retain their original uncompressed embedded NPY arrays, which the loader memory-maps. Recompressing the source events prevents that access method. Paths in TOML files resolve relative to the TOML file, not the shell working directory. The default files resolve to the layout above.

Inspect the setup before compute:

```bash
gnnsr inspect --config experiments/v2/hidden_width/configs/width_64.toml
```

Before saved-LSG inference, hash-verify its inputs in the LSG environment:

```bash
python scripts/verify_lsg_full_domain_inputs.py
```

That command writes a receipt under `cache/`; the LSG runner rejects missing or changed inputs. No datasets or original GNN checkpoint files are included. New GNN training produces its own weights and normalization. To evaluate externally obtained historical weights, retain their matching configuration, normalization and original provenance; do not pass them off as a fresh run from this reorganized source.
