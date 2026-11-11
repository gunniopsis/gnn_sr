# GNN-SR: Burnett River flood super-resolution

Research code accompanying **Graph Neural Network Super-Resolution for Computationally Efficient Flood Modelling**, Roland Thomas Clemson, University of Sydney, November 2026.

GNN-SR predicts a fine-grid water-surface-elevation correction from the corresponding coarse HEC-RAS state. Two graphs represent the coarse mesh and the 20 m TUFLOW grid. Coarse message passing, directed coarse-to-fine fusion and fine message passing produce one independent prediction per timestep.

This repository reproduces the Burnett River experiments in the final report:

| Stage | Experiment | Runs |
|---|---|---:|
| 1 | AOI latent/MLP widths 64, 128, 256 | 3 |
| 2 | AOI composite loss: 8 temperatures x 5 weights, plus fresh MSE control | 41 |
| 3 | AOI coarse layers {1,2,3,4} x fine layers {1,2,3,4,6} | 20 |
| 4 | Full-domain training: width 64, 2/2 layers, T=0.3 m, lambda=0.01 | 1 |

Every experiment uses fold 1, seed 42 and the supplied LSG-compatible partitions. Checkpoints are selected by pooled validation depth RMSE. Held-out results are reporting only. The full-domain run trains for at most 100 epochs; the original study selected epoch 95. The adopted 2/2 architecture balances performance and computational cost; 4/4 had the lowest AOI validation RMSE.

## Start here

1. [Install the environment](docs/environment.md).
2. [Obtain and arrange the external inputs](docs/data.md).
3. [Follow the reproduction sequence](docs/reproduction.md), or the [Gadi guide](gadi/README.md).
4. [Generate the report comparisons and figures](docs/report_outputs.md).

```bash
python scripts/verify_repository.py
python -m pip install -e '.[dev,plot]'
python -m pytest
gnnsr --help
```

Install a suitable PyTorch CUDA build before the editable installation on GPU systems. The scientific tests use synthetic inputs and run on CPU. The full experiment runners require one visible CUDA GPU and the external Burnett data. No experiments are submitted by installation or tests.

## Repository layout

```text
src/gnnsr/              Model, graph/data handling, losses, training and metrics
experiments/v2/         Protocols, candidate grids, TOML configs and PBS templates
  hidden_width/
  composite_loss/
  message_passing/
  full_domain/
  lsg_full_domain/      Supporting saved-LSG inference for report benchmarks
scripts/               Experiment runners, verification, analysis and GIS exports
examples/              Development AOI and full-domain polygons
tests/                 CPU scientific and reproduction-protocol checks
docs/                  Data, environment, methodology and reproduction guides
gadi/                  Shared execution instructions and environment requirements
SOURCE_MANIFEST.json   Checksums and original-to-public source mapping
```

Data belong in `reference_files/BurnettRV/`; generated runs in `outputs/v2/<experiment>/<variant>/fold_1/seed_42/`; rebuildable graphs in `cache/`; figures and tables in `reports/`. These directories are excluded from Git.

## Reproducibility boundaries

This is a fresh source release for new runs. Use a separate checkout and output tree; historical run provenance intentionally rejects modified source. Datasets, saved LSG models, GNN checkpoints and generated results are not bundled. See [data access](docs/data.md). No public download link for the exact supplied dataset or trained GNN weights is asserted here.

The calculations used for the reported numerical results are preserved. Read the [method and evaluation protocol](docs/methods.md). Hardware and library changes can change training trajectories; a fresh run must select its own validation-best epoch.

## Citation and attribution

See [CITATION.cff](CITATION.cff) and [NOTICE.md](NOTICE.md). The supplied hydraulic data and saved LSG models remain external research inputs; their provenance and terms are separate from this code.
