"""Recreate the report training figure and export its epoch table."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_data(run, last_epoch):
    with (run / "history.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    rows = [r for r in rows if r["epoch"] <= last_epoch]
    assert [int(r["epoch"]) for r in rows] == list(range(1, last_epoch + 1))
    config = json.loads((run / "config.json").read_text())
    for row in rows:
        assert all(math.isfinite(v) for v in row.values())
        assert row["loss_extent_weight"] == config["loss"]["extent_weight"]
        assert row["loss_temperature_m"] == config["loss"]["temperature_m"]
        assert math.isclose(row["validation_depth_rmse_m"], row["validation_pooled_depth_rmse_m"])
        for phase in ("train", "validation"):
            assert math.isclose(row[f"{phase}_total_loss"], row[f"{phase}_residual_mse"] + row[f"{phase}_weighted_extent_loss"], rel_tol=1e-6)
            assert math.isclose(row[f"{phase}_weighted_extent_loss"], config["loss"]["extent_weight"] * row[f"{phase}_extent_loss"], rel_tol=1e-6)
    best = min(rows, key=lambda r: r["validation_depth_rmse_m"])
    return rows, config, best


def make_plot(rows, best, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(7.15, 7.15))
    x = np.array([r["epoch"] for r in rows])
    vals = lambda key: np.array([r[key] for r in rows])
    blue, red, dark, orange = "#176DA3", "#AB3F53", "#253547", "#BF6D1C"
    for j, phase in enumerate(("train", "validation")):
        name = "Training" if phase == "train" else "Validation"
        ax = axes[0, j]
        ax.plot(x, vals(f"{phase}_total_loss"), color=dark, lw=1.25, label=r"Total ($\mathcal{L}$)")
        ax.plot(x, vals(f"{phase}_residual_mse"), color=blue, lw=1.1, label=r"Residual MSE ($\mathcal{L}_{\mathrm{MSE}}$)")
        ax.set_title(f"({'ab'[j]}) {name} loss", loc="left")
        ax.set_yscale("log")
        ax.set_ylim(.004, .025)
        ax.yaxis.set_major_locator(FixedLocator([.005, .01, .02]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("Loss")
        ax.legend(frameon=False, fontsize=7, handlelength=1.6, handletextpad=.5,
                  labelspacing=.3, loc="upper right")
        ax = axes[1, j]
        y = vals(f"{phase}_weighted_extent_loss")
        ax.plot(x, y, color=orange, lw=1.1)
        ax.set_title(f"({'cd'[j]}) {name} weighted extent loss", loc="left")
        ax.set_yscale("log")
        low, high = y.min(), y.max()
        ax.set_ylim(low * .999, high * 1.001)
        ax.yaxis.set_major_locator(FixedLocator(np.geomspace(low, high, 4)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.5f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(r"Loss ($\lambda\mathcal{L}_{\mathrm{extent}}$)")
    for phase, label, color in (("train", "Training", blue), ("validation", "Validation", red)):
        symbol = "train" if phase == "train" else "val"
        axes[2, 0].plot(x, vals(f"{phase}_pooled_depth_rmse_m"), color=color, lw=1.1,
                        label=label + r" ($\mathrm{RMSE}_{\mathrm{" + symbol + r"}}$)")
        axes[2, 1].plot(x, vals(f"{phase}_mean_timestep_csi"), color=color, lw=1.1, label=label)
    axes[2, 0].set_title("(e) Pooled depth RMSE", loc="left")
    axes[2, 1].set_title("(f) Mean timestep CSI", loc="left")
    for ax, ticks, limits, label in ((axes[2, 0], [.2, .25, .3, .4, .5], (.20, .51), "RMSE (m)"),
                                   (axes[2, 1], [.5, .6, .7, .8, .9], (.45, .93), "CSI")):
        ax.set_yscale("log")
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=6.5, handlelength=1.6, handletextpad=.5,
                  labelspacing=.3, loc="upper right" if ax == axes[2, 0] else "lower right")
    axes[3, 0].step(x, vals("learning_rate"), where="post", color=dark, lw=1.2)
    axes[3, 0].set_title("(g) Learning rate after scheduling", loc="left")
    axes[3, 0].set_ylabel("Learning rate")
    axes[3, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[3, 0].set_ylim(0, .0011)
    for field, label, color in (("training_seconds", "Training", blue), ("validation_seconds", "Validation", red)):
        axes[3, 1].plot(x, vals(field) / 60, color=color, lw=1.1, label=label)
    axes[3, 1].set_title("(h) Time per epoch", loc="left")
    axes[3, 1].set_ylabel("Time (min)")
    axes[3, 1].set_ylim(0, 85)
    axes[3, 1].legend(frameon=False, fontsize=7, handlelength=1.6, handletextpad=.5,
                      labelspacing=.3, loc="center right")
    for ax in axes.flat:
        ax.set_xlim(1, x[-1] + 1)
        ax.set_xticks([1, 25, 50, 75, 100])
        ax.set_xlabel(r"Epoch ($e$)", labelpad=1)
        ax.axvline(best["epoch"], color="#777777", ls=":", lw=.8, zorder=0)
        ax.grid(alpha=.18, lw=.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.115, right=.985, top=.965, bottom=.055, wspace=.36, hspace=.65)
    fig.savefig(out / "appendix_c_training_metrics.png", dpi=350, facecolor="white")
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/v2/full_domain/c2_f2/fold_1/seed_42")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/training")
    args = parser.parse_args()
    with (args.run / "history.csv").open(newline="") as handle:
        last_epoch = max(int(r["epoch"]) for r in csv.DictReader(handle))
    rows, config, best = read_data(args.run, last_epoch)
    args.output.mkdir(parents=True, exist_ok=True)
    make_plot(rows, best, args.output)
    with (args.output / "epoch_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected epoch {int(best['epoch'])}; exported {len(rows)} epochs to {args.output}")
