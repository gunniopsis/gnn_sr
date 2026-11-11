"""Local-only validation ranking and loss/CSI plots for completed AOI trials."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from plan_loss_experiments import load_plan
from plot_training_dashboard import load_history, plot_dashboard
from report_loss_evaluation import report_heldout
from run_loss_experiment import ROOT
from run_width_experiment import comparison_settings, sha256, write_json


def load_trial(directory):
    config = json.loads((directory / "config.json").read_text())
    status = json.loads((directory / "training_status.json").read_text())
    provenance = json.loads((directory / "loss_experiment.json").read_text())
    if status["state"] != "complete":
        raise ValueError(f"Incomplete trial: {directory}")
    if config["model"]["hidden_dim"] != 64:
        raise ValueError("Expected hidden width 64")
    if any(
        config["training"].get(f"debug_max_{split}_samples") is not None
        for split in ("train", "validation", "test")
    ):
        raise ValueError("Preflight/debug run cannot enter the comparison")
    if provenance["loss"] != config["loss"] or provenance["settings"] != comparison_settings(
        config
    ):
        raise ValueError("Saved configuration differs from provenance")
    with (directory / "history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(r["epoch"]) for r in rows] != list(range(1, status["epoch"] + 1)):
        raise ValueError("Incomplete or discontinuous history")
    if not all(np.isfinite(float(r["validation_depth_rmse_m"])) for r in rows):
        raise ValueError("Nonfinite validation RMSE")
    best = min(rows, key=lambda r: float(r["validation_depth_rmse_m"]))
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    if (
        checkpoint["epoch"] != int(best["epoch"])
        or not np.isclose(
            checkpoint["validation_depth_rmse_m"], float(best["validation_depth_rmse_m"])
        )
        or checkpoint["config"]["loss"] != config["loss"]
        or comparison_settings(checkpoint["config"]) != provenance["settings"]
    ):
        raise ValueError("Checkpoint does not match validation selection/configuration")
    norm_name = checkpoint["normalization_path"].replace("\\", "/").split("/")[-1]
    name = directory.parents[1].name
    return {
        "name": name,
        "directory": directory,
        "rows": rows,
        "best": best,
        "provenance": provenance,
        "normalization_sha256": sha256(directory / norm_name),
        "checkpoint_sha256": sha256(directory / "best.pt"),
        "config": config,
    }


def report(root, stage="initial"):
    plan = load_plan(stage=stage)
    planned = {item["name"]: item for item in plan}
    directories = sorted(root.glob("*/fold_1/seed_42"))
    trials = []
    skipped = []
    for directory in directories:
        if directory.parents[1].name not in planned:
            continue
        status_path = directory / "training_status.json"
        if not status_path.exists() or json.loads(status_path.read_text())["state"] != "complete":
            skipped.append(str(directory))
            continue
        trials.append(load_trial(directory))
    if not trials:
        raise ValueError("No completed trials; download full run directories first")
    reference = trials[0]
    for trial in trials:
        expected = planned.get(trial["name"])
        if expected is None or any(
            trial["config"]["loss"][key] != expected[key]
            for key in ("temperature_m", "extent_weight")
        ):
            raise ValueError(f"Trial differs from the approved grid: {trial['name']}")
        for key in (
            "settings",
            "splits",
            "split_file_sha256",
            "aoi_sha256",
            "source_sha256",
            "runner_sha256",
            "provenance_helper_sha256",
            "python",
            "torch",
            "cuda",
            "gpu",
        ):
            if trial["provenance"][key] != reference["provenance"][key]:
                raise ValueError(f"Incomparable trial {trial['name']}: {key}")
        if trial["normalization_sha256"] != reference["normalization_sha256"]:
            raise ValueError("Different normalization")
    trials.sort(key=lambda trial: float(trial["best"]["validation_depth_rmse_m"]))
    output = root / ("comparison" if stage == "initial" else "comparison_extended")
    output.mkdir(parents=True, exist_ok=True)
    completed = {trial["name"] for trial in trials}
    coverage = [
        {
            **item,
            "local_status": (
                "complete_verified"
                if item["name"] in completed
                else "incomplete_or_unverified"
                if (root / item["name"] / "fold_1/seed_42").exists()
                else "not_present_locally"
            ),
        }
        for item in plan
    ]
    with (output / "initial_grid_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    rankings = [
        {
            "candidate": t["name"],
            **t["config"]["loss"],
            "best_epoch": t["best"]["epoch"],
            "epochs": len(t["rows"]),
            "validation_rmse_m": t["best"]["validation_depth_rmse_m"],
            "validation_mean_timestep_csi": t["best"]["validation_mean_timestep_csi"],
            "validation_false_alarms": t["best"]["validation_false_alarms"],
            "validation_misses": t["best"]["validation_misses"],
        }
        for t in trials
    ]
    with (output / "validation_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rankings[0]))
        writer.writeheader()
        writer.writerows(rankings)
    selected = trials[:3]
    control = next((t for t in trials if t["name"] == "control"), None)
    if control is not None and control not in selected:
        selected.append(control)
    colors = {t["name"]: plt.get_cmap("tab10")(i) for i, t in enumerate(selected)}
    plot_dashboard(
        tuple(load_history(t["directory"] / "history.csv", t["name"]) for t in selected),
        output / "training.png",
        output / "training.pdf",
        colors=colors,
        title="Composite loss: width 64, validation-ranked trials and control",
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    components = ("residual_mse", "extent_loss", "weighted_extent_loss", "total_loss")
    for row, phase in enumerate(("train", "validation")):
        for col, component in enumerate(components):
            axis = axes[row, col]
            for trial in selected:
                axis.plot(
                    [int(r["epoch"]) for r in trial["rows"]],
                    [float(r[f"{phase}_{component}"]) for r in trial["rows"]],
                    label=trial["name"],
                    color=colors[trial["name"]],
                    linewidth=1,
                )
            axis.set(title=f"{phase}: {component.replace('_', ' ')}", xlabel="Epoch")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Loss components (different T/lambda define different objectives)")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"loss_components.{suffix}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for trial in selected:
        for phase, style in (("train", "--"), ("validation", "-")):
            for axis, key in zip(axes, ("mean_timestep_depth_rmse_m", "mean_timestep_csi")):
                axis.plot(
                    [int(r["epoch"]) for r in trial["rows"]],
                    [float(r[f"{phase}_{key}"] or "nan") for r in trial["rows"]],
                    style,
                    color=colors[trial["name"]],
                    label=f"{trial['name']} {phase}",
                )
    axes[0].set(title="Mean timestep depth RMSE", ylabel="m", xlabel="Epoch")
    axes[1].set(title="Mean defined timestep CSI", ylim=(0, 1.02), xlabel="Epoch")
    axes[0].legend(fontsize=6)
    fig.suptitle("Training pass (dashed); fixed-model validation (solid); not event maxima")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"timestep_metrics.{suffix}", dpi=200)
    plt.close(fig)
    write_json(
        output / "verification.json",
        {
            "verified": True,
            "ranking": "validation RMSE only",
            "stage": stage,
            "initial_grid_expected": len(plan),
            "initial_grid_complete": len(completed) == len(plan),
            "initial_grid_missing_locally_or_incomplete": sorted(set(planned) - completed),
            "skipped_incomplete": skipped,
            "rankings": rankings,
            "checkpoints": {t["name"]: t["checkpoint_sha256"] for t in trials},
        },
    )
    report_heldout(trials, plan, output)
    print(f"Validated {len(trials)} trials; wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/v2/composite_loss")
    parser.add_argument("--stage", choices=("initial", "all"), default="initial")
    args = parser.parse_args()
    report(args.root, args.stage)
