# Gadi execution

The PBS files are execution templates for an account with the required queues and storage. Replace `PROJECT_CODE` with the allocation. Prepare the GPU environment in an internet-enabled setup allocation, installing this package as described in [environment.md](../docs/environment.md). Never upload a local virtual environment.

From the repository root:

```bash
mkdir -p logs/gadi
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/width64_preflight.log" -v GNNSR_WIDTH=64,GNNSR_MODE=preflight experiments/v2/hidden_width/gadi/run.pbs
```

The shared launcher activates `.venv-gadi` by default. Add `GNNSR_VENV=/absolute/path/to/env` to `-v` to override it. If a module is required, also pass `GNNSR_PYTORCH_MODULE=module/version`; no module name or allocation is imposed by the launcher.

| Experiment | PBS path below `experiments/v2/` | Variables |
|---|---|---|
| Width | `hidden_width/gadi/run.pbs` | `GNNSR_WIDTH=64`, `GNNSR_MODE=preflight` or `train` |
| Loss | `composite_loss/gadi/run.pbs` | `GNNSR_CANDIDATE=control`, `GNNSR_MODE=calibration`, `preflight`, `train` or `evaluate` |
| Depth | `message_passing/gadi/run.pbs` | `GNNSR_CANDIDATE=c2_f2`, `GNNSR_MODE=preflight`, `train` or `evaluate` |
| Full domain | `full_domain/gadi/run.pbs` | `GNNSR_MODE=preflight`, `train` or `evaluate` |
| LF/common-mask comparison | `full_domain/gadi/compare.pbs` | Environment overrides as above |
| Saved LSG inference | `lsg_full_domain/gadi/evaluate.pbs` | `GNNSR_LSG_VENV` if not `.venv-lsg`; `LSG_RESUME=1` for continuation |

Set `GNNSR_RESUME=1` for a continuation or evaluation of an existing GNN run. Use a distinct log filename for each submission. Calibration runs once after the width-64 MSE reference has completed; it does not need `GNNSR_RESUME`. Full training starts only after successful preflight.

Example full-domain sequence (submit each only after the preceding step succeeds):

```bash
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/full_preflight.log" -v GNNSR_MODE=preflight experiments/v2/full_domain/gadi/run.pbs
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/full_train_001.log" -v GNNSR_MODE=train experiments/v2/full_domain/gadi/run.pbs
qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE -o "$(pwd)/logs/gadi/full_train_002.log" -v GNNSR_MODE=train,GNNSR_RESUME=1 experiments/v2/full_domain/gadi/run.pbs
```

Inspect `training_status.json`, `pipeline_status.json` and the PBS log. Continue segmented training until complete, then submit evaluation with `GNNSR_MODE=evaluate,GNNSR_RESUME=1`. Run locks prevent duplicate writers; remove a stale lock only after checking that no owning job remains. The public templates do not automatically submit continuations or evaluations.

Download complete run folders for local plotting. Keep full-domain predictions and graph cache until comparisons, GIS exports and timing analyses are complete. Do not overwrite historical runs or mix artifacts from this release with an earlier source deployment.
