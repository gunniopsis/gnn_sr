#!/bin/bash
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the GNNSR project root}"
module load "${GNNSR_PYTORCH_MODULE:-pytorch/2.12.0}"
source "${GNNSR_VENV:-${PBS_O_WORKDIR}/.venv-gadi}/bin/activate"
export PYTHONPATH="${PBS_O_WORKDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${PBS_NCPUS:-12}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ARGS=(--mode "${LOSS_JOB_MODE}")
if [[ "${LOSS_JOB_MODE}" != "calibration" ]]; then
    ARGS+=(--candidate "${GNNSR_CANDIDATE:?Specify GNNSR_CANDIDATE}")
fi
case "${GNNSR_RESUME:-0}" in
    0) ;;
    1) ARGS+=(--resume) ;;
    *) echo "GNNSR_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${LOSS_JOB_MODE}" == "preflight" ]]; then
    ARGS+=(--segment-hours 0.75)
else
    ARGS+=(--segment-hours 43)
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/run_loss_experiment.py "${ARGS[@]}"
