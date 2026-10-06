#!/bin/bash
# Shared body sourced by the hidden-width GPU PBS jobs.
set -euo pipefail

PROJECT_ROOT="${PBS_O_WORKDIR:?Submit from the GNNSR project root}"
WIDTH="${GNNSR_WIDTH:?Set GNNSR_WIDTH to 64, 128 or 256 with qsub -v}"
case "${WIDTH}" in
    64|128|256) ;;
    *) echo "Unsupported GNNSR_WIDTH=${WIDTH}" >&2; exit 2 ;;
esac
case "${GNNSR_RESUME:-0}" in
    0|1) ;;
    *) echo "GNNSR_RESUME must be 0 or 1" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
module load "${GNNSR_PYTORCH_MODULE:-pytorch/2.12.0}"
source "${GNNSR_VENV:-${PROJECT_ROOT}/.venv-gadi}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${PBS_NCPUS:-12}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

ARGS=(--width "${WIDTH}")
if [[ "${GNNSR_RESUME:-0}" == "1" ]]; then
    ARGS+=(--resume)
fi
if [[ "${WIDTH_JOB_MODE}" == "preflight" ]]; then
    ARGS+=(--preflight --segment-hours 0.75)
else
    ARGS+=(--segment-hours 43)
fi
echo "Width=${WIDTH}; mode=${WIDTH_JOB_MODE}; PBS_JOBID=${PBS_JOBID:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/run_width_experiment.py "${ARGS[@]}"
