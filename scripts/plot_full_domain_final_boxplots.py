"""Plot the completed common-mask Burnett comparison without changing evaluation data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/v2/full_domain/c2_f2/fold_1/seed_42/lsg_wet_mask_comparison"
DESTINATION = ROOT / "reports/figures/full_domain_comparison"
MODELS = (("LF", "LF HEC-RAS", "#B56727"), ("GNN-SR", "GNN-SR", "#252A83"),
          ("LSG", "LSG", "#5C7331"))
PANELS = (
    ("maximum_extent_csi", "Maximum flood extent", "Critical Success Index (CSI)", 1),
    ("average_peak_difference_m", "Peak water depth", "AvgPeakDiff [m]", 0),
    ("maximum_depth_r2", "Peak water depth", "Coefficient of determination ($R^2$)", 1),
    ("rmse_m", "Water depth hydrographs", "AvgRMSE [m]", 0),
    ("fidelity_index", "Water depth hydrographs", "Fidelity Index (FI)", 1),
)


def plot(source=SOURCE, destination=DESTINATION):
    metrics_path = source / "event_metrics.csv"
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    completion = json.loads((source / "completion.json").read_text())
    if completion["state"] != "complete" or completion["artifacts"]["event_metrics.csv"] != digest:
        raise ValueError("Comparison is incomplete or metric checksum differs")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 54 or {row["model"] for row in rows} != {m[0] for m in MODELS}:
        raise ValueError("Expected 18 events for each of the three models")
    with (ROOT / "reference_files/BurnettRV/BurnettRV_event_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        events = [row["tuflow_evt_name_old"] for row in csv.DictReader(handle) if int(row["Group"]) == 1]
    grouped = {}
    for model, _, _ in MODELS:
        selected = [row for row in rows if row["model"] == model]
        if len(selected) != 18 or {r["event_name"] for r in selected} != set(events):
            raise ValueError(f"Event coverage differs for {model}")
        if any(int(r["group"]) != 1 or int(r["n_target_cells"]) != 769709 or int(r["n_timesteps"]) != 383 for r in selected):
            raise ValueError("Spatial or temporal scoring support differs")
        grouped[model] = {row["event_name"]: row for row in selected}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "axes.linewidth": .8})
    fig = plt.figure(figsize=(9, 10.2))
    grid = fig.add_gridspec(3, 2, left=.105, right=.98, top=.96, bottom=.06,
                           wspace=.45, hspace=.48)
    positions = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]
    axes = [fig.add_subplot(grid[row, column]) for row, column in positions]
    statistics = []
    axis_limits = {}
    for letter, ax, (key, title, ylabel, ideal) in zip("abcde", axes, PANELS, strict=True):
        values = [np.array([float(grouped[model][event][key]) for event in events]) for model, _, _ in MODELS]
        if any(not np.isfinite(a).all() for a in values):
            raise ValueError(f"Non-finite values in {key}; no events may be silently dropped")
        artists = ax.boxplot(values, widths=.5, patch_artist=True, whis=1.5,
                             medianprops={"color": "black", "linewidth": 1.4},
                             boxprops={"edgecolor": "black", "linewidth": .9},
                             whiskerprops={"color": "black", "linewidth": .9},
                             capprops={"color": "black", "linewidth": .9},
                             flierprops={"marker": "o", "markerfacecolor": "none",
                                         "markeredgecolor": "black", "markersize": 4})
        for box, (_, _, colour) in zip(artists["boxes"], MODELS, strict=True):
            box.set_facecolor(colour)
            box.set_alpha(.85)
        ax.set_xticks([1, 2, 3], [m[1] for m in MODELS])
        ax.set_title(f"({letter}) {title}", loc="left", pad=10)
        ax.set_ylabel(ylabel)
        all_values = np.concatenate(values)
        low, high = float(all_values.min()), float(all_values.max())
        padding = (high - low) * .08 if high > low else max(abs(low), .01) * .08
        limits = (low - padding, high + padding)
        # Include the perfect-score reference with breathing room in panels d/e.
        if letter in ("d", "e"):
            limits = (min(limits[0], ideal - padding), max(limits[1], ideal + padding))
        ax.set_ylim(*limits)
        axis_limits[key] = list(limits)
        # Panels a-c retain their existing data-fitted limits.
        if limits[0] <= ideal <= limits[1]:
            ax.axhline(ideal, linestyle="--", linewidth=.85, color="#666666", zorder=0)
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for (model, _, _), data in zip(MODELS, values, strict=True):
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            statistics.append({"model": model, "metric": key, "n": len(data),
                               "q1": float(q1), "median": float(median), "q3": float(q3),
                               "mean": float(data.mean()), "min": float(data.min()), "max": float(data.max())})
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / "burnett_full_domain_lf_gnnsr_lsg_boxplots"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)
    metadata = {"source": str(metrics_path), "source_sha256": digest, "events": events,
                "checkpoint_epoch": completion["checkpoint_epoch"], "n_scoring_cells": 769709,
                "statistics": statistics,
                "layout": {"rows": 3, "columns": 2, "panel_positions_zero_based": positions},
                "y_axis_limits": axis_limits,
                "metric_convention": "Report labels; unchanged adopted evaluation calculations, including pooled event RMSE and original FI."}
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    plot(args.source, args.destination)
