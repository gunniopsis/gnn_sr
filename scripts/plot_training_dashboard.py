"""Plot a six-panel laptop CPU versus Gadi GPU training dashboard."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAPTOP = (
    ROOT / "outputs" / "recommended_aoi" / "fold_1" / "seed_42" / "history.csv"
)
DEFAULT_GADI = (
    ROOT / "outputs" / "gadi_1x1km_gpu" / "fold_1" / "seed_42" / "history.csv"
)
DEFAULT_PNG = (
    ROOT
    / "outputs"
    / "gadi_1x1km_gpu"
    / "comparison"
    / "fold1_training_dashboard.png"
)
DEFAULT_PDF = DEFAULT_PNG.with_suffix(".pdf")

COLORS = {"Laptop CPU": "#252A83", "Gadi GPU": "#1B8A8F"}


@dataclass(frozen=True)
class RunHistory:
    label: str
    epoch: np.ndarray
    training_mse: np.ndarray
    validation_rmse: np.ndarray
    learning_rate: np.ndarray
    gradient_norm: np.ndarray
    clipped_fraction: np.ndarray
    epoch_seconds: np.ndarray

    @property
    def elapsed_hours(self) -> np.ndarray:
        return np.cumsum(self.epoch_seconds) / 3600.0

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.validation_rmse))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laptop", type=Path, default=DEFAULT_LAPTOP)
    parser.add_argument("--gadi", type=Path, default=DEFAULT_GADI)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    return parser.parse_args()


def load_history(path: Path, label: str) -> RunHistory:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training history is empty: {path}")

    def values(field: str) -> np.ndarray:
        return np.asarray([float(row[field]) for row in rows], dtype=float)

    return RunHistory(
        label=label,
        epoch=values("epoch"),
        training_mse=values("training_mse"),
        validation_rmse=values("validation_depth_rmse_m"),
        learning_rate=values("learning_rate"),
        gradient_norm=values("mean_gradient_norm_before_clipping"),
        clipped_fraction=values("clipped_step_fraction"),
        epoch_seconds=values("epoch_seconds"),
    )


def rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def style_axis(axis: plt.Axes, label: str, title: str, ylabel: str) -> None:
    axis.set_title(f"({label}) {title}", loc="left", fontweight="normal")
    axis.set_ylabel(ylabel)
    axis.grid(color="#D2D2D2", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_dashboard(
    runs: tuple[RunHistory, ...], png_path: Path, pdf_path: Path, *,
    colors: dict[str, str] | None = None, title: str | None = None,
) -> None:
    colors = COLORS if colors is None else colors
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(11.7, 10.0))
    ax_train, ax_val, ax_time, ax_lr, ax_grad, ax_clip = axes.flat

    for run_index, run in enumerate(runs):
        color = colors[run.label]
        ax_train.plot(run.epoch, run.training_mse, color=color, linewidth=1.45, label=run.label)

        smoothed = rolling_mean(run.validation_rmse)
        ax_val.plot(run.epoch, run.validation_rmse, color=color, linewidth=0.7, alpha=0.28)
        ax_val.plot(run.epoch, smoothed, color=color, linewidth=1.6, label=run.label)
        best = run.best_index
        ax_val.scatter(
            run.epoch[best],
            run.validation_rmse[best],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=42,
            zorder=4,
        )
        ax_val.text(
            0.97,
            0.95 - 0.09 * run_index,
            f"{run.label}: best epoch {int(run.epoch[best])}, "
            f"{run.validation_rmse[best]:.4f} m",
            transform=ax_val.transAxes,
            ha="right",
            va="top",
            color=color,
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

        ax_time.plot(
            run.elapsed_hours,
            run.validation_rmse,
            color=color,
            linewidth=0.75,
            alpha=0.30,
        )
        ax_time.plot(
            run.elapsed_hours,
            rolling_mean(run.validation_rmse),
            color=color,
            linewidth=1.6,
            label=run.label,
        )
        ax_time.scatter(
            run.elapsed_hours[best],
            run.validation_rmse[best],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=42,
            zorder=4,
        )

        ax_lr.step(run.epoch, run.learning_rate, where="post", color=color, linewidth=1.5)
        ax_grad.plot(run.epoch, run.gradient_norm, color=color, linewidth=1.35)
        ax_clip.plot(run.epoch, run.clipped_fraction, color=color, linewidth=1.35)

    style_axis(ax_train, "a", "Training loss", "Training MSE [normalised]")
    style_axis(ax_val, "b", "Validation performance", "Validation depth RMSE [m]")
    style_axis(ax_time, "c", "Convergence efficiency", "Validation depth RMSE [m]")
    style_axis(ax_lr, "d", "Learning-rate schedule", "Learning rate")
    style_axis(ax_grad, "e", "Gradient magnitude", "Mean gradient norm before clipping")
    style_axis(ax_clip, "f", "Gradient-clipping activity", "Clipped optimizer steps [fraction]")

    ax_train.set_xlabel("Epoch")
    ax_train.set_yscale("log")
    ax_val.set_xlabel("Epoch")
    ax_time.set_xlabel("Cumulative epoch time [hours]")
    ax_lr.set_xlabel("Epoch")
    ax_grad.set_xlabel("Epoch")
    ax_clip.set_xlabel("Epoch")
    ax_lr.set_yscale("log")
    ax_clip.set_ylim(bottom=0)

    handles, labels = ax_train.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(runs),
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
    )
    fig.suptitle(title or "Fold 1, seed 42: laptop CPU and Gadi GPU training",
                 fontsize=14, y=0.985)
    fig.text(
        0.5,
        0.055,
        "Validation panels show raw epoch values (faint), five-epoch rolling means (solid), and the best checkpoint (circle).",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.105, wspace=0.25, hspace=0.36)

    for path in (png_path, pdf_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = (
        load_history(args.laptop, "Laptop CPU"),
        load_history(args.gadi, "Gadi GPU"),
    )
    plot_dashboard(runs, args.png, args.pdf)
    print(f"Saved {args.png}")
    print(f"Saved {args.pdf}")


if __name__ == "__main__":
    main()
