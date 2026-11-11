#!/bin/bash
# Shared launcher. Submission and continuation are explicit, never automatic.
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
if [[ -n "${GNNSR_PYTORCH_MODULE:-}" ]]; then module load "$GNNSR_PYTORCH_MODULE"; fi
source "${GNNSR_VENV:-$PWD/.venv-gadi}/bin/activate"
export OMP_NUM_THREADS="${PBS_NCPUS:-12}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
python scripts/verify_repository.py
mode="${GNNSR_MODE:-preflight}"
resume=()
if [[ "${GNNSR_RESUME:-0}" == 1 ]]; then resume+=(--resume); fi
case "${GNNSR_STAGE:?}" in
  width)
    args=(--width "${GNNSR_WIDTH:?Set GNNSR_WIDTH to 64, 128 or 256}" --segment-hours 6)
    if [[ "$mode" == preflight ]]; then args+=(--preflight)
    elif [[ "$mode" != train ]]; then echo 'Width mode must be preflight or train'; exit 2; fi
    python scripts/run_width_experiment.py "${args[@]}" "${resume[@]}"
    ;;
  loss)
    python scripts/run_loss_experiment.py --candidate "${GNNSR_CANDIDATE:-control}" --mode "$mode" --segment-hours 6 "${resume[@]}"
    ;;
  depth)
    python scripts/run_depth_experiment.py --candidate "${GNNSR_CANDIDATE:?}" --mode "$mode" --segment-hours 6 "${resume[@]}"
    ;;
  full)
    python scripts/run_full_domain_experiment.py --mode "$mode" --segment-seconds 21600 "${resume[@]}"
    ;;
  *) echo 'Unknown stage'; exit 2 ;;
esac

