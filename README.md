# GNN-SR: Burnett River flood super-resolution

A two-scale graph neural network for fine-resolution flood prediction from coarse
hydraulic model states. Includes training, evaluation and the width/composite-loss
experiments. Data and trained models are supplied separately.

| Path | Purpose |
| --- | --- |
| src/gnnsr/ | Model, data, graph, loss, training and evaluation code |
| experiments/v2/ | Experiment instructions, configurations and Gadi jobs |
| scripts/ | Width and composite-loss experiment runners |
| examples/ | The 1 km development AOI polygon |

## Installation

Use Python 3.11 or later. In an activated environment, from this repository root:

```bash
python -m pip install -e .
```

On Gadi, use the existing compatible CUDA environment. Alternatively, from an
internet-enabled copyq session, create an environment using the module used by the jobs:

```bash
module load pytorch/2.12.0
python3 -m venv --system-site-packages .venv-gadi
source .venv-gadi/bin/activate
python3 -m pip install -e .
```

Check module availability at your site. Jobs accept GNNSR_PYTORCH_MODULE and
GNNSR_VENV overrides and require exactly one visible CUDA GPU.

## External inputs

Supply the original dataset separately at reference_files/BurnettRV/:

```text
BurnettRV_event_summary.csv
Geometry_data/HECRAS_Geometry_data.npz
Geometry_data/Tuflow_Geometry_data.npz
HD_model_data/High-fidelity/       # Original uncompressed TUFLOW NPZ events
HD_model_data/Low-fidelity/        # Original HEC-RAS HDF events
Train_test_split_data/            # Original Train_test_split_ValidateOnGrp_N.npz
```

Preserve event order and saved splits. Config paths are relative to each TOML.

## Training and prediction

Follow the [width experiment](experiments/v2/hidden_width/README.md) and
[composite-loss experiment](experiments/v2/composite_loss/README.md). These runners
record the provenance needed for calibration. Both experiments use the supplied AOI.
Generate held-out predictions and metrics from a matching trained checkpoint:

```bash
gnnsr evaluate --config experiments/v2/hidden_width/configs/width_64.toml --checkpoint outputs/v2/hidden_width/width_64/fold_1/seed_42/best.pt
```

Keep best.pt and its normalization JSON together; supply the matching configuration
and reference data. Transfer complete run folders separately. The CLI performs
dataset-based evaluation and prediction export; arbitrary new-input inference has
no separate CLI command. Use gnnsr --help for preparation and direct-training commands.

Results belong in outputs/<version>/<experiment>/<variant>/fold_N/seed_N/;
caches in cache/ and PBS logs in logs/gadi/. Tests, plots, design reports, old baseline
workflows and release builders remain in the full local project and are omitted here.
This sharing copy omits the optional LSG and raw HEC-RAS comparison commands.
Use the original full project to resume historical runs: experiment provenance
checks include source hashes, which differ for this reduced command interface.
