# Environment

Use Python 3.11 or newer (`tomllib` is used by experiment planning). Python 3.12 is the CPU test target. Install the GPU build of PyTorch appropriate for the compute host before installing the project; the package deliberately does not force a particular CUDA wheel.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,plot]'
python -m pytest
```

On Windows activate with `.venv\Scripts\Activate.ps1`. The `plot` extra includes matplotlib and rasterio for local figures and GIS exports. The CPU tests do not require the real dataset or GPflow. Tests do not run full-domain training.

The original Gadi jobs used the `pytorch/2.12.0` module. That is historical environment information, not a claim that it is available on every host. Check available modules; set `GNNSR_PYTORCH_MODULE` if needed. Preserve the environment throughout one run because source/environment provenance is checked on resume. Each experiment records Python, PyTorch, CUDA and GPU information. Save `python -m pip freeze` with the generated run record for a new reproduction.

## Optional saved-LSG inference

Basic GNN training and the five-metric comparison against the supplied LSG summary do not require TensorFlow. Recomputing LSG predictions, POD/RFA and peak-area timing does. Use a separate environment to avoid changing the GNN environment:

```bash
python -m venv .venv-lsg
source .venv-lsg/bin/activate
python -m pip install -r gadi/lsg-requirements.txt
python -m pip install -e .
```

`gadi/lsg-requirements.txt` preserves the tested historical LSG dependency pins. It is not a full platform-independent lock file. PyTorch still needs to be provided for the target platform. For compatibility work outside that historical environment, the `lsg` extra declares broad package requirements, but any changed environment needs its own validation.

## Resource expectations

AOI experiments use 2,314 target cells; the full-domain graph has 780,785 fine nodes. Full-domain training was performed on one V100 GPU with 32 GB allocated host memory. The source PBS templates use 12 CPUs and one GPU. Preflight measures throughput and memory before training. Preserve every full-domain epoch checkpoint and allow substantial disk space for 18 events of WSE/depth predictions. The common-mask comparison requests 64 GB host RAM and LSG inference 48 GB.

The six-hour segment budget in the eight-hour PBS templates leaves time for an epoch to finish and checkpoints to be written. The budget is checked at epoch boundaries; assess the preflight timing before using these limits on different hardware. Training continuation is explicit in this release, with optimizer, scheduler and RNG state restored.
