"""Preview initial or temperature-extension Gadi jobs; never submit or access a GPU."""

import argparse
import json
import shlex
from itertools import product
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("experiments/v2/composite_loss")
TEMPERATURES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
EXTENSION_TEMPERATURES = (0.5, 1.0)
WEIGHTS = (0.001, 0.01, 0.1, 1.0, 10.0)
STAGES = ("initial", "temperature_extension", "all")


def load_plan(root=ROOT, stage="initial"):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    candidates = json.loads((root / EXPERIMENT / "candidates.json").read_text())
    pairs = [(c["temperature_m"], c["extent_weight"]) for c in candidates]
    names = [c["name"] for c in candidates]
    expected = set(product(TEMPERATURES + EXTENSION_TEMPERATURES, WEIGHTS)) | {(0.03, 0.0)}
    if len(pairs) != 41 or set(pairs) != expected or len(set(names)) != 41:
        raise ValueError("Expected 31 initial runs plus 10 temperature-extension pairs")
    for candidate in candidates:
        name = candidate["name"]
        expected_name = (
            "control"
            if candidate["extent_weight"] == 0
            else f"t{candidate['temperature_m']:g}_l{candidate['extent_weight']:g}".replace(
                ".", "p"
            )
        )
        if name != expected_name:
            raise ValueError(f"Unexpected candidate name: {name}")
        config = tomllib.loads((root / EXPERIMENT / "configs" / f"{name}.toml").read_text())
        if config["model"]["hidden_dim"] != 64 or config["training"]["seed"] != 42:
            raise ValueError(f"Expected width 64, seed 42: {name}")
        if any(config["loss"][key] != candidate[key] for key in ("temperature_m", "extent_weight")):
            raise ValueError(f"Configuration differs from plan: {name}")
    if stage == "all":
        return candidates
    return [
        c for c in candidates
        if (c["temperature_m"] in EXTENSION_TEMPERATURES) == (stage == "temperature_extension")
    ]


def commands(mode="train", root=ROOT, stage="initial"):
    if mode not in ("train", "preflight"):
        raise ValueError("Only preflight and initial training commands are planned")
    for candidate in load_plan(root, stage):
        name = candidate["name"]
        yield (
            "qsub -P PROJECT_CODE -l storage=scratch/PROJECT_CODE "
            f'-o "$(pwd)/logs/gadi/$(date +%Y%m%dT%H%M%S)_loss_{mode}_{name}.log" '
            f"-v GNNSR_CANDIDATE={shlex.quote(name)},GNNSR_MODE={mode} "
            f"{EXPERIMENT.as_posix()}/gadi/run.pbs"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "preflight"), default="train")
    parser.add_argument("--stage", choices=STAGES, default="initial")
    parser.add_argument("--names-only", action="store_true")
    args = parser.parse_args()
    if args.names_only:
        print("\n".join(c["name"] for c in load_plan(stage=args.stage)))
    else:
        print(f"# Preview only: {len(load_plan(stage=args.stage))} {args.stage} runs, Fold 1, seed 42.")
        print("# Execute from the Gadi project root after numerical checks; never resubmit active jobs.")
        print("\n".join(commands(args.mode, stage=args.stage)))
