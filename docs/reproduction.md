# Reproduce the Burnett River study

Run commands from the repository root after installing the package and supplying the external data. The commands below run inside an allocated one-GPU environment. Use [the Gadi templates](../gadi/README.md) to obtain an allocation; do not run training on a login node.

All paths refer to fresh outputs created by this release. Do not resume the original research-directory runs from this reorganized checkout. Keep source files and the environment fixed throughout each run. Segment budgets are checked at epoch boundaries. A `segment_complete` status requires continuation; it is not a completed experiment.

## 1. AOI width comparison

For each width in 64, 128 and 256:

```bash
python scripts/run_width_experiment.py --width 64 --preflight --segment-hours 6
python scripts/run_width_experiment.py --width 64 --segment-hours 6
```

Repeat with 128 and 256. If training ends at a segment limit, repeat the training command with `--resume` until `training_status.json` says `complete`. The width runner then evaluates the validation-selected checkpoint on the 18 held-out events. Preflight has its own output subdirectory.

This stage retains MSE-only training, two coarse and three fine layers. Width 64 is the adopted setting and supplies the checkpoint/normalization/provenance needed for calibration.

## 2. Loss calibration and all 41 training runs

After width 64 has completed training and evaluation:

```bash
python scripts/run_loss_experiment.py --mode calibration --samples 24
python scripts/plan_loss_experiments.py --stage all --names-only
```

Calibration evaluates the original six temperatures and five weights at initialization and the trained width-64 model using training samples. Preserve all 30 combinations. The later T=0.5 and T=1.0 m extension adds ten runs. The full training plan contains 40 composite combinations and one fresh MSE control; do not substitute the earlier width-64 run for that control.

For **every** name printed by the planner, replace `control` below with that name:

```bash
python scripts/run_loss_experiment.py --candidate control --mode preflight --segment-hours 6
python scripts/run_loss_experiment.py --candidate control --mode train --segment-hours 6
```

Continue a segmented training run with the same command plus `--resume`. When it is complete:

```bash
python scripts/run_loss_experiment.py --candidate control --mode evaluate --resume
```

Evaluate **all 41** validation-selected checkpoints. Rank the trials using validation RMSE; held-out results are reporting only. The adopted setting is T=0.3 m and lambda=0.01.

## 3. All 20 message-passing configurations

```bash
python scripts/plan_depth_experiments.py --names-only
```

For every listed name, use the same sequence, replacing `c2_f2`:

```bash
python scripts/run_depth_experiment.py --candidate c2_f2 --mode preflight --segment-hours 6
python scripts/run_depth_experiment.py --candidate c2_f2 --mode train --segment-hours 6
```

Resume segments with `--resume`; after training completes:

```bash
python scripts/run_depth_experiment.py --candidate c2_f2 --mode evaluate --resume
```

The 20 configurations share the 4/6-hop context graph and unchanged target cells. All use width 64, T=0.3 m and lambda=0.01. The 2/2 model was adopted for full-domain computational cost, although 4/4 had the lowest AOI validation RMSE.

## 4. Fresh full-domain training

```bash
python scripts/run_full_domain_experiment.py --mode preflight --segment-seconds 21600
python scripts/run_full_domain_experiment.py --mode train --segment-seconds 21600
```

Preflight measures 20 training and five validation timesteps, reports a throughput estimate and checks the full graph. Training requires a successful preflight and starts with fresh weights and training-only normalization. Continue with:

```bash
python scripts/run_full_domain_experiment.py --mode train --resume --segment-seconds 21600
```

Repeat only after the preceding job finishes, until training is complete. It stops at epoch 100 or earlier if patience 30 is reached. It never extends beyond 100. The original experiment was manually stopped at 100; encoding that cap makes the same stopping boundary explicit for new reproductions. Selection still uses the minimum validation RMSE, and a new run may select a different epoch from the original epoch 95.

```bash
python scripts/run_full_domain_experiment.py --mode evaluate --resume
```

This exports all 18 full-domain events' WSE/depth predictions and metrics. Preserve `checkpoints/`, `best.pt`, `latest.pt`, normalization, histories, provenance, graph cache and predictions. Outputs are under `outputs/v2/full_domain/c2_f2/fold_1/seed_42/`.

## 5. Common-mask LF/GNN-SR/LSG comparison

After full-domain inference, on a sufficiently provisioned CPU allocation:

```bash
python scripts/compare_full_domain_lsg.py --verify-deployment
```

The comparison verifies source/prediction hashes, maps the supplied 769,709-cell mask, scores GNN-SR, reads supplied LSG benchmark metrics, and exports/scores the parent-projected LF baseline. It saves event metrics, peak maps, mapping geometry and LF timestep predictions under the full-domain run's `lsg_wet_mask_comparison/`. Retain the original training graph cache for exact parent mapping.

## 6. Saved-LSG inference for the remaining report tables

This supporting step evaluates externally supplied LSG weights; it does not train another model. In the separate LSG environment:

```bash
python scripts/verify_lsg_full_domain_inputs.py
python scripts/run_lsg_full_domain.py
```

Use `--resume` only for a previously started LSG evaluation. This provides LSG predictions and counts needed for POD/RFA and peak-area timing. See [report outputs](report_outputs.md) for subsequent local plotting and tables.
