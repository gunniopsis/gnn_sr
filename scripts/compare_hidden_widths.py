"""Verify and plot completed 64/128/256 MSE-only AOI experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from plot_fold1_boxplots import PANELS, box_statistics, create_figure
from plot_training_dashboard import load_history, plot_dashboard
from run_width_experiment import ROOT, WIDTHS, comparison_settings, sha256, write_json

COLORS = {"Width 64": "#252A83", "Width 128": "#1B8A8F", "Width 256": "#B56727"}
DEFAULT_ROOT = ROOT / "outputs/v2/hidden_width"


def metric_rows(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    rows = {}
    for row in records:
        name = row["event_name"]
        if name in rows:
            raise ValueError(f"Duplicate event {name}: {path}")
        rows[name] = {
            "event_name": name,
            **{panel.key: float(row[panel.key]) for panel in PANELS},
            "n_timesteps": int(row["n_timesteps"]),
            "n_target_cells": int(row["n_target_cells"]),
        }
    if len(rows) != 18:
        raise ValueError(f"Expected all 18 Fold 1 events, found {len(rows)}: {path}")
    return rows


def load_run(run_dir: Path, width: int) -> dict:
    def read(name):
        return json.loads((run_dir / name).read_text(encoding="utf-8"))

    provenance = read("width_experiment.json")
    config = read("config.json")
    status = read("training_status.json")
    evaluation = read("width_evaluation.json")
    if status["state"] != "complete":
        raise ValueError(f"Training is not complete: {run_dir}")
    if config["model"]["hidden_dim"] != width or provenance["hidden_dim"] != width:
        raise ValueError(f"Wrong hidden width: {run_dir}")
    if comparison_settings(config) != provenance["settings"]:
        raise ValueError(f"Configuration differs from provenance: {run_dir}")
    if config["training"]["fold"] != 1 or config["training"]["seed"] != 42:
        raise ValueError("This comparison requires fold 1 and seed 42")
    for name in ("train", "validation", "test"):
        if config["training"].get(f"debug_max_{name}_samples") is not None:
            raise ValueError("Debug/preflight sample limits are not valid comparison runs")
    checkpoint_path = run_dir / "best.pt"
    if sha256(checkpoint_path) != evaluation["checkpoint_sha256"]:
        raise ValueError(f"Evaluated checkpoint has changed: {run_dir}")
    if sha256(run_dir / "test_event_metrics.csv") != evaluation["metrics_sha256"]:
        raise ValueError(f"Evaluated metrics have changed: {run_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["config"]["model"] != config["model"]:
        raise ValueError(f"Checkpoint model differs from configuration: {run_dir}")
    if comparison_settings(checkpoint["config"]) != provenance["settings"]:
        raise ValueError(f"Checkpoint training settings differ: {run_dir}")
    normalization_name = checkpoint["normalization_path"].replace("\\", "/").split("/")[-1]
    if sha256(run_dir / normalization_name) != evaluation["normalization_sha256"]:
        raise ValueError(f"Normalization has changed: {run_dir}")
    label = f"Width {width}"
    history = load_history(run_dir / "history.csv", label)
    arrays = (
        history.training_mse,
        history.validation_rmse,
        history.learning_rate,
        history.gradient_norm,
        history.clipped_fraction,
        history.epoch_seconds,
    )
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"Nonfinite training history: {run_dir}")
    if not np.array_equal(history.epoch, np.arange(1, len(history.epoch) + 1)):
        raise ValueError(f"History must contain consecutive epochs: {run_dir}")
    if np.any(history.epoch_seconds <= 0):
        raise ValueError(f"Nonpositive epoch duration: {run_dir}")
    best = history.best_index
    if checkpoint["epoch"] != int(history.epoch[best]):
        raise ValueError(f"Checkpoint is not the RMSE-selected history epoch: {run_dir}")
    if evaluation["checkpoint_epoch"] != checkpoint["epoch"]:
        raise ValueError(f"Evaluation checkpoint epoch differs: {run_dir}")
    if not np.isclose(checkpoint["validation_depth_rmse_m"], history.validation_rmse[best]):
        raise ValueError(f"Checkpoint RMSE differs from history: {run_dir}")
    if status["epoch"] != int(history.epoch[-1]):
        raise ValueError(f"History does not end at the completed epoch: {run_dir}")
    return {
        "provenance": provenance,
        "evaluation": evaluation,
        "history": history,
        "metrics": metric_rows(run_dir / "test_event_metrics.csv"),
    }


def validate_comparison(runs: dict[int, dict]) -> list[str]:
    reference = runs[64]
    events = sorted(reference["metrics"])
    shared_keys = (
        "settings",
        "aoi_sha256",
        "split_file_sha256",
        "splits",
        "source_sha256",
        "runner_sha256",
        "python",
        "torch",
        "cuda",
        "gpu",
    )
    for width, run in runs.items():
        for key in shared_keys:
            if run["provenance"][key] != reference["provenance"][key]:
                raise ValueError(f"Width {width} differs in {key}")
        if (
            run["evaluation"]["normalization_sha256"]
            != reference["evaluation"]["normalization_sha256"]
        ):
            raise ValueError(f"Width {width} used different normalization")
        if sorted(run["metrics"]) != events:
            raise ValueError(f"Width {width} has different test events")
        for event in events:
            for key in ("n_timesteps", "n_target_cells"):
                if run["metrics"][event][key] != reference["metrics"][event][key]:
                    raise ValueError(f"Width {width}: {event} differs in {key}")
    return events


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(root: Path, output: Path) -> None:
    runs = {width: load_run(root / f"width_{width}/fold_1/seed_42", width) for width in WIDTHS}
    events = validate_comparison(runs)
    output.mkdir(parents=True, exist_ok=True)
    model_order = tuple(COLORS)
    rows = {f"Width {width}": run["metrics"] for width, run in runs.items()}
    create_figure(
        rows,
        events,
        model_order,
        output / "hidden_width_boxplots.png",
        output / "hidden_width_boxplots.pdf",
        colors=COLORS,
        legend_labels={label: label for label in model_order},
        title="Hidden width comparison: 1 km AOI, 18 held-out events, MSE-only",
    )
    plot_dashboard(
        tuple(run["history"] for run in runs.values()),
        output / "hidden_width_training.png",
        output / "hidden_width_training.pdf",
        colors=COLORS,
        title="Hidden width comparison: Fold 1, seed 42, MSE-only",
    )
    statistics = []
    for panel in PANELS:
        for width, run in runs.items():
            values = np.array([run["metrics"][event][panel.key] for event in events])
            finite = values[np.isfinite(values)]
            stats = (
                box_statistics(finite)
                if len(finite)
                else {
                    "n": 0,
                    **{
                        key: float("nan")
                        for key in (
                            "minimum",
                            "q1",
                            "median",
                            "q3",
                            "maximum",
                            "lower_whisker",
                            "upper_whisker",
                            "mean",
                        )
                    },
                }
            )
            statistics.append(
                {
                    "metric": panel.key,
                    "hidden_dim": width,
                    "undefined_events": int(len(values) - len(finite)),
                    **stats,
                }
            )
    summary = []
    for width, run in runs.items():
        history = run["history"]
        best = history.best_index
        summary.append(
            {
                "hidden_dim": width,
                "trainable_parameters": run["provenance"]["trainable_parameters"],
                "epochs": len(history.epoch),
                "best_epoch": int(history.epoch[best]),
                "best_validation_depth_rmse_m": float(history.validation_rmse[best]),
                "hours_to_best": float(history.elapsed_hours[best]),
                "total_epoch_hours": float(history.elapsed_hours[-1]),
            }
        )
    write_csv(output / "boxplot_statistics.csv", statistics)
    write_csv(output / "training_summary.csv", summary)
    write_csv(
        output / "event_metrics.csv",
        [
            {"hidden_dim": width, **run["metrics"][event]}
            for width, run in runs.items()
            for event in events
        ],
    )
    write_json(
        output / "verification.json",
        {
            "verified": True,
            "events": events,
            "source_root": str(root.resolve()),
            "runs": summary,
            "checkpoint_selection": "Minimum pooled validation physical-depth RMSE",
            "checks": [
                "identical splits, AOI and normalization",
                "matched source and environment",
                "matched model settings except width",
                "complete training histories",
                "RMSE-selected checkpoints",
                "checkpoint and metrics hashes",
                "matching event names, timestep counts and target-cell counts",
            ],
        },
    )
    report = [
        "# Hidden width comparison",
        "",
        "Widths 64, 128 and 256; original MSE loss; original LSG-compatible splits.",
        "Each test result uses that width's minimum-validation-RMSE checkpoint.",
        "",
        "![Five event metric boxplots](hidden_width_boxplots.png)",
        "",
        "![Training dashboard](hidden_width_training.png)",
        "",
        "See training_summary.csv and boxplot_statistics.csv for numerical results.",
        "Nonfinite event metrics are omitted per box and counted in the statistics CSV.",
        "The comparison does not automatically select a width or start loss development.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Verified comparison saved to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compare(args.root, args.output or args.root / "comparison")


if __name__ == "__main__":
    main()
