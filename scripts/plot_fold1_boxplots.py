"""Create two Fold 1 boxplot figures: LSG/GNN, and HEC-RAS/LSG/GNN."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GNN = ROOT / "outputs" / "recommended_aoi" / "fold_1" / "seed_42" / "test_event_metrics.csv"
DEFAULT_LSG = ROOT / "outputs" / "recommended_aoi" / "lsg" / "fold_1" / "event_metrics.csv"
DEFAULT_HEC = ROOT / "outputs" / "recommended_aoi" / "hec_ras" / "fold_1" / "event_metrics.csv"
DEFAULT_PNG = ROOT / "outputs" / "recommended_aoi" / "comparison" / "fold1_gnn_lsg_boxplots.png"
DEFAULT_PDF = ROOT / "reports" / "pdf" / "fold1_gnn_lsg_boxplots.pdf"
DEFAULT_BASELINE_PNG = (
    ROOT
    / "outputs"
    / "recommended_aoi"
    / "comparison"
    / "fold1_hec_ras_lsg_gnn_boxplots.png"
)
DEFAULT_BASELINE_PDF = ROOT / "reports" / "pdf" / "fold1_hec_ras_lsg_gnn_boxplots.pdf"
DEFAULT_STATS = ROOT / "outputs" / "recommended_aoi" / "comparison" / "fold1_boxplot_statistics.csv"


@dataclass(frozen=True)
class Panel:
    key: str
    title: str
    ylabel: str
    ideal: float | None
    fixed_limits: tuple[float, float] | None = None


PANELS = (
    Panel("maximum_extent_csi", "Maximum flood extent", "Critical Success Index [-]", 1.0, (0.65, 1.01)),
    Panel("average_peak_difference_m", "Peak water depth difference", "Average peak difference [m]", 0.0),
    Panel("maximum_depth_r2", "Peak water depth", "Max water depth, R² [-]", 1.0),
    Panel("rmse_m", "Water depth hydrograph", "RMSE [m]", 0.0),
    Panel("fidelity_index", "Water depth hydrograph", "Fidelity Index [-]", 1.0, (0.76, 1.01)),
)

ALL_MODELS = ("HEC-RAS", "LSG", "GNN")
LSG_GNN_MODELS = ("LSG", "GNN")
COLORS = {
    "HEC-RAS": "#B56727",
    "LSG": "#5C7331",
    "GNN": "#252A83",
    "Laptop GNN": "#252A83",
    "Gadi GPU GNN": "#1B8A8F",
}
LEGEND_LABELS = {
    "HEC-RAS": "HEC-RAS baseline",
    "LSG": "LSG model",
    "GNN": "GNN model",
    "Laptop GNN": "Laptop-trained GNN",
    "Gadi GPU GNN": "Gadi GPU-trained GNN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn", type=Path, default=DEFAULT_GNN)
    parser.add_argument("--lsg", type=Path, default=DEFAULT_LSG)
    parser.add_argument("--hec", type=Path, default=DEFAULT_HEC)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--baseline-png", type=Path, default=DEFAULT_BASELINE_PNG)
    parser.add_argument("--baseline-pdf", type=Path, default=DEFAULT_BASELINE_PDF)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATS)
    return parser.parse_args()


def load_rows(path: Path, model: str) -> dict[str, dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: dict[str, dict[str, float | str]] = {}
    for row in rows:
        values: dict[str, float | str] = {"event_name": row["event_name"], "model": model}
        for panel in PANELS:
            values[panel.key] = float(row[panel.key])
        parsed[row["event_name"]] = values
    return parsed


def padded_limits(values: np.ndarray, ideal: float | None) -> tuple[float, float]:
    candidates = values
    if ideal is not None:
        candidates = np.append(candidates, ideal)
    low = float(np.min(candidates))
    high = float(np.max(candidates))
    span = max(high - low, max(abs(low), abs(high), 1.0) * 0.04)
    return low - 0.10 * span, high + 0.10 * span


def box_statistics(values: np.ndarray) -> dict[str, float | int]:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    within = values[(values >= lower_fence) & (values <= upper_fence)]
    return {
        "n": len(values),
        "minimum": float(np.min(values)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "maximum": float(np.max(values)),
        "lower_whisker": float(np.min(within)),
        "upper_whisker": float(np.max(within)),
        "mean": float(np.mean(values)),
    }


def create_figure(
    rows: dict[str, dict[str, dict[str, float | str]]],
    events: list[str],
    model_order: tuple[str, ...],
    png_path: Path,
    pdf_path: Path,
    *,
    colors: dict[str, str] | None = None,
    legend_labels: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    colors = COLORS if colors is None else colors
    legend_labels = LEGEND_LABELS if legend_labels is None else legend_labels
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(11.7, 7.6), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        6,
        left=0.085,
        right=0.985,
        top=0.925,
        bottom=0.135,
        wspace=1.15,
        hspace=0.37,
    )
    axes = (
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    )
    for label, panel, axis in zip("abcde", PANELS, axes, strict=True):
        data = [
            np.asarray([rows[model][event][panel.key] for event in events], dtype=float)
            for model in model_order
        ]
        data = [values[np.isfinite(values)] for values in data]
        artists = axis.boxplot(
            data,
            positions=range(1, len(model_order) + 1),
            widths=0.54,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            boxprops={"edgecolor": "black", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "none",
                "markeredgecolor": "black",
                "markersize": 3.5,
                "linestyle": "none",
            },
        )
        for box, model in zip(artists["boxes"], model_order, strict=True):
            box.set_facecolor(colors[model])
        axis.set_xticks(range(1, len(model_order) + 1), model_order)
        axis.set_title(f"({label}) {panel.title}", loc="left", fontweight="normal")
        axis.set_ylabel(panel.ylabel)
        axis.grid(axis="y", color="#D2D2D2", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        combined = np.concatenate(data)
        if not len(combined):
            combined = np.array([0.0, 1.0])
            axis.text(0.5, 0.5, "No finite values", transform=axis.transAxes, ha="center")
        data_limits = padded_limits(combined, panel.ideal)
        if panel.fixed_limits is None:
            limits = data_limits
        else:
            limits = (
                min(panel.fixed_limits[0], data_limits[0]),
                max(panel.fixed_limits[1], data_limits[1]),
            )
        axis.set_ylim(*limits)
        if panel.ideal is not None:
            axis.axhline(panel.ideal, color="#555555", linewidth=0.8, linestyle="--", zorder=0)

    legend = [
        Patch(facecolor=colors[model], edgecolor="black", label=legend_labels[model])
        for model in model_order
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=len(model_order),
        frameon=False,
        bbox_to_anchor=(0.5, 0.054),
    )
    fig.suptitle(title or "Fold 1 AOI comparison across 18 held-out Burnett River events",
                 fontsize=13, y=0.985)
    fig.text(
        0.5,
        0.018,
        "Boxes show Q1–Q3 and the median; whiskers extend to 1.5×IQR and circles denote outliers.",
        ha="center",
        fontsize=8,
    )

    for path in (png_path, pdf_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = {
        "HEC-RAS": load_rows(args.hec, "HEC-RAS"),
        "LSG": load_rows(args.lsg, "LSG"),
        "GNN": load_rows(args.gnn, "GNN"),
    }
    event_sets = {model: set(model_rows) for model, model_rows in rows.items()}
    if len({frozenset(events) for events in event_sets.values()}) != 1:
        raise ValueError("HEC-RAS, LSG, and GNN event names do not match")
    events = sorted(event_sets["GNN"])
    if len(events) != 18:
        raise ValueError(f"Expected 18 Fold 1 events, found {len(events)}")

    create_figure(rows, events, LSG_GNN_MODELS, args.png, args.pdf)
    create_figure(rows, events, ALL_MODELS, args.baseline_png, args.baseline_pdf)

    statistics_rows: list[dict[str, float | int | str]] = []
    for panel in PANELS:
        for model in ALL_MODELS:
            values = np.asarray(
                [rows[model][event][panel.key] for event in events], dtype=float
            )
            statistics_rows.append(
                {"metric": panel.key, "model": model, **box_statistics(values)}
            )

    fields = ["metric", "model", "n", "minimum", "q1", "median", "q3", "maximum", "lower_whisker", "upper_whisker", "mean"]
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    with args.statistics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(statistics_rows)
    print(args.png.resolve())
    print(args.pdf.resolve())
    print(args.baseline_png.resolve())
    print(args.baseline_pdf.resolve())
    print(args.statistics.resolve())


if __name__ == "__main__":
    main()
