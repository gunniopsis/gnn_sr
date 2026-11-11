"""Recreate the report AOI figures from verified run summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures/aoi_tuning"
SOURCES = {}
METRICS = [
    ("validation_rmse_m", "Validation depth\nRMSE (m)", "lower", 4),
    ("maximum_extent_csi_mean", "Flood extent\nCSI", "higher", 4),
    ("rmse_m_mean", "Depth hydrographs\nAvgRMSE (m)", "lower", 4),
    ("average_peak_difference_m_mean", "Peak depth\nAvgPeakDiff (m)", "zero", 4),
    ("maximum_depth_r2_mean", "Peak depth\nR²", "higher", 5),
    ("fidelity_index_mean", "Depth hydrographs\nFI", "higher", 4),
]


def tracked(path):
    path = ROOT / path
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_csv(path):
    with tracked(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(tracked(path).read_text(encoding="utf-8-sig"))


def collect():
    widths = read_csv("outputs/v2/hidden_width/comparison/training_summary.csv")
    loss = read_csv("outputs/v2/composite_loss/comparison_extended/test_metric_summary.csv")
    layers = read_csv("outputs/v2/message_passing/comparison/metric_summary.csv")
    assert len(widths) == 3 and len(loss) == 41 and len(layers) == 20
    event_names, common_splits = None, None
    for family, rows, prefix in (("hidden_width", widths, "width"),
                                 ("composite_loss", loss, "loss"),
                                 ("message_passing", layers, "depth")):
        for row in rows:
            name = f"width_{row['hidden_dim']}" if family == "hidden_width" else row["candidate"]
            run = Path("outputs/v2") / family / name / "fold_1/seed_42"
            history = read_csv(run / "history.csv")
            status = read_json(run / "training_status.json")
            provenance = read_json(run / f"{prefix}_experiment.json")
            assert status["state"] == "complete"
            assert [int(r["epoch"]) for r in history] == list(range(1, len(history) + 1))
            best = min(history, key=lambda r: float(r["validation_depth_rmse_m"]))
            metric = float(best["validation_depth_rmse_m"])
            summary_key = "best_validation_depth_rmse_m" if family == "hidden_width" else "validation_rmse_m"
            assert math.isclose(float(row[summary_key]), metric, rel_tol=1e-12)
            epoch_key = "checkpoint_epoch" if family == "composite_loss" else "best_epoch"
            assert int(row[epoch_key]) == int(best["epoch"])
            assert math.isclose(status["best_validation_depth_rmse_m"], metric, rel_tol=1e-12)
            assert provenance["settings"]["training"]["fold"] == 1
            assert provenance["settings"]["training"]["seed"] == 42
            if common_splits is None:
                common_splits = provenance["splits"]
            assert provenance["splits"] == common_splits
            events = read_csv(run / "test_event_metrics.csv")
            assert len(events) == 18
            names = sorted(r["event_name"] for r in events)
            if event_names is None:
                event_names = names
            assert names == event_names
            assert all(int(r["n_target_cells"]) == 2314 and int(r["n_timesteps"]) == 383 for r in events)
            row["events"] = events
            row["validation_rmse_m"] = metric
            row["parameters"] = provenance["trainable_parameters"]
            for key, *_ in METRICS[1:]:
                actual = mean(float(r[key.removesuffix("_mean")]) for r in events)
                assert math.isfinite(actual)
                if family != "hidden_width":
                    assert math.isclose(float(row[key]), actual, rel_tol=1e-12)
                row[key] = actual
    grid_loss = [r for r in loss if r["candidate"] != "control"]
    assert {(float(r["temperature_m"]), float(r["extent_weight"])) for r in grid_loss} == {
        (t, w) for t in [.001, .003, .01, .03, .1, .3, .5, 1] for w in [.001, .01, .1, 1, 10]}
    assert {(int(r["coarse_layers"]), int(r["fine_layers"])) for r in layers} == {
        (c, f) for c in [1, 2, 3, 4] for f in [1, 2, 3, 4, 6]}
    aoi = read_json("examples/recommended_development_aoi.geojson")
    assert aoi["features"][0]["properties"]["side_m"] == 1000
    return widths, loss, layers


def figures(widths, loss, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42})

    def save(fig, name):
        fig.savefig(OUT / f"{name}.png", dpi=350, facecolor="white")
        fig.savefig(OUT / f"{name}.pdf", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.1))
    for n, (ax, (key, title, direction, precision)) in enumerate(zip(axes.flat, METRICS)):
        ax.set_title(f"({'abcdef'[n]}) {title}", loc="left")
        if n == 0:
            ys = [r[key] for r in widths]
            ax.plot(range(1, 4), ys, "o-", color="#597B99", lw=1, markersize=4)
            for i, v in enumerate(ys):
                ax.annotate(f"{v:.4f}", (i + 1, v), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=7)
            ax.set_ylim(.235, .25)
        else:
            values = [[float(e[key.removesuffix("_mean")]) for e in r["events"]] for r in widths]
            bp = ax.boxplot(values, widths=.46, patch_artist=True,
                            flierprops={"marker": ".", "markersize": 3, "markeredgecolor": "#666666"},
                            medianprops={"color": "#253547", "linewidth": 1})
            for patch in bp["boxes"]:
                patch.set_facecolor("#D9E7F2")
                patch.set_edgecolor("#597B99")
            ax.margins(y=.2)
        ax.axvspan(.65, 1.35, facecolor="none", edgecolor="#D17B12", linewidth=1.5, zorder=4)
        ax.set_xlim(.5, 3.5)
        ax.set_xticks([1, 2, 3], ["64", "128", "256"])
        ax.set_xlabel("Latent node dimension")
        ax.grid(axis="y", alpha=.2, lw=.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.subplots_adjust(left=.08, right=.99, top=.90, bottom=.10, wspace=.45, hspace=.78)
    save(fig, "B2_model_size")

    def heatmaps(rows, family):
        is_loss = family == "loss"
        xs = [.001, .01, .1, 1, 10] if is_loss else [1, 2, 3, 4, 6]
        ys = [.001, .003, .01, .03, .1, .3, .5, 1] if is_loss else [1, 2, 3, 4]
        xkey, ykey = ("extent_weight", "temperature_m") if is_loss else ("fine_layers", "coarse_layers")
        cells = {(float(r[ykey]), float(r[xkey])): r for r in rows if r.get("candidate") != "control"}
        adopted = (.3, .01) if is_loss else (2, 2)
        fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.65 if is_loss else 6.2))
        for n, (ax, (key, title, direction, precision)) in enumerate(zip(axes.flat, METRICS)):
            raw = np.array([[float(cells[y, x][key]) for x in xs] for y in ys])
            colored = np.abs(raw) if direction == "zero" else raw
            norm = Normalize(float(colored.min()), float(colored.max()))
            cmap = plt.get_cmap("Blues" if direction == "higher" else "Blues_r")
            mesh = ax.imshow(colored, norm=norm, cmap=cmap, aspect="auto", interpolation="nearest",
                             origin="upper" if is_loss else "lower")
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    luminance = np.dot(cmap(norm(colored[i, j]))[:3], [.2126, .7152, .0722])
                    ax.text(j, i, f"{raw[i, j]:.{precision}f}", ha="center", va="center", fontsize=7.1,
                            color="white" if luminance < .5 else "#111111")
            i, j = ys.index(adopted[0]), xs.index(adopted[1])
            for linewidth, color in ((3.4, "white"), (1.7, "#D17B12")):
                ax.add_patch(Rectangle((j - .48, i - .48), .96, .96, fill=False,
                                       linewidth=linewidth, edgecolor=color))
            ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
            ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
            ax.set_xlabel("Extent-loss weight (λ)" if is_loss else "Fine message-passing layers", labelpad=2)
            ax.set_ylabel("Soft-wetness temperature T (m)" if is_loss else "Coarse message-passing layers", labelpad=3)
            ax.set_title(f"({'abcdef'[n]}) {title}", loc="left", pad=7)
            bar = fig.colorbar(mesh, ax=ax, fraction=.035, pad=.025)
            bar.ax.tick_params(labelsize=6.5, length=2)
            bar.locator = MaxNLocator(4)
            bar.formatter = FormatStrFormatter("%.4f" if key == "maximum_depth_r2_mean" else "%.3f")
            bar.update_ticks()
        fig.subplots_adjust(left=.105, right=.95, top=.935, bottom=.06, wspace=.40, hspace=.56)
        save(fig, "B3_loss_function_grid" if is_loss else "B4_message_passing_grid")

    heatmaps(loss, "loss")
    heatmaps(layers, "layers")



if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    widths, loss, layers = collect()
    figures(widths, loss, layers)
    (OUT / "provenance.json").write_text(json.dumps({
        "source_sha256": SOURCES, "verified_runs": 64,
        "adopted": {"width": 64, "temperature_m": .3, "extent_weight": .01,
                    "coarse_layers": 2, "fine_layers": 2},
        "validation_best_loss": min(loss, key=lambda r: r["validation_rmse_m"])["candidate"],
        "validation_best_depth": min(layers, key=lambda r: r["validation_rmse_m"])["candidate"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Generated report AOI figures in {OUT}")
