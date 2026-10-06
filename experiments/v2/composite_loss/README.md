# Width-64 composite-loss experiment

Use width 64, the existing 1 km AOI, Fold 1 and seed 42. Preserve LSG-compatible
splits, validation depth RMSE checkpoint selection, scheduler and early-stopping
patience 30. A differentiable timestep extent term augments standardized residual
MSE. Physical dry filtering and test metrics remain unchanged. See the
[root README](../../../README.md) for data and environment setup.

The complete initial search is six temperatures by five weights: 30 composite
configurations plus one fresh MSE control, listed in candidates.json. Calibration
probes all 30 pairs in one job, on training data only. It does not train or prune
candidates. All 31 runs retain the full partitions and existing stopping rules.

Calibration requires the completed outputs/v2/hidden_width/width_64/fold_1/seed_42/
run, including best.pt, normalization JSON and width_evaluation.json. Reproduce it
with the width runner or obtain the complete verified run separately.

From the project root on Gadi, replace PROJECT_CODE with your allocation:

```bash
mkdir -p logs/gadi
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/loss_calibration.log" experiments/v2/composite_loss/gadi/calibration.pbs
```

Review calibration gradient norms, alignment and saturation locally before training.
Resolve numerical failures without pruning the agreed grid. For each of the 31
names in candidates.json, replace NAME below with its exact name (for example,
control or t0p03_l0p01) and submit a preflight:

```bash
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/loss_NAME_preflight.log" -v GNNSR_CANDIDATE=NAME experiments/v2/composite_loss/gadi/preflight.pbs
```

After that candidate's preflight succeeds, submit its full training job:

```bash
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/loss_NAME_train.log" -v GNNSR_CANDIDATE=NAME experiments/v2/composite_loss/gadi/train.pbs
```

Repeat for every candidate, including the control. Do not resubmit queued/running
candidates. All runs need not execute concurrently. Resume segments with
GNNSR_CANDIDATE=NAME,GNNSR_RESUME=1
and a new log name after checking training_status.json. Never edit a running trial's source.

Rank trials by minimum pooled validation depth RMSE recorded in history.csv.
Training does not automatically evaluate test events. Evaluate the control and
validation-selected finalists with (replace NAME):

```bash
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/loss_NAME_evaluate.log" -v GNNSR_CANDIDATE=NAME,GNNSR_RESUME=1 experiments/v2/composite_loss/gadi/evaluate.pbs
```

Keep complete run folders and logs. Reporting and plots remain in the full local
project. Never select loss settings using test performance.
