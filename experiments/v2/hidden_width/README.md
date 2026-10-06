# Hidden-width experiment

Train widths 64, 128 and 256 in the same 1 km AOI, Fold 1, seed 42. Preserve the
original MSE objective, LSG-compatible splits, validation depth RMSE checkpoint
selection, scheduler and early-stopping patience 30. Follow the
[root README](../../../README.md) for data and environment setup.

From the project root on Gadi, replace PROJECT_CODE with your allocation:

```bash
mkdir -p logs/gadi
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/width64_preflight.log" -v GNNSR_WIDTH=64 experiments/v2/hidden_width/gadi/preflight.pbs
```

After preflight succeeds, submit training:

```bash
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/width64_train.log" -v GNNSR_WIDTH=64 experiments/v2/hidden_width/gadi/train.pbs
```

Repeat for widths 128 and 256, changing width and log name. Resume segmented runs
with GNNSR_WIDTH=64,GNNSR_RESUME=1 and a new log name. Check training_status.json
first; never start concurrent writers. The runner evaluates best.pt after training
completes. Keep each complete outputs/v2/hidden_width/width_N/fold_1/seed_42/ folder.
Calibration needs the width-64 best.pt, normalization JSON and width_evaluation.json.
Produce comparison plots with the full local project.
