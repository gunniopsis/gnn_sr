"""Plot the verified temperature/weight grid locally, without changing selection."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from plan_loss_experiments import ROOT, WEIGHTS, load_plan

# key, title, preferred direction, annotation precision, gain multiplier/unit
PANELS = (
    ("validation_rmse_m", "Validation water depth\nRoot Mean Square Error (RMSE)", "lower", 4, 100, "cm"),
    ("maximum_extent_csi_mean", "Flood extent\nCritical Success Index (CSI)", "higher", 4, 100, "percentage points"),
    ("rmse_m_mean", "Water depth hydrographs\nAverage Root Mean Square Error (AvgRMSE)", "lower", 4, 100, "cm"),
    ("average_peak_difference_m_mean", "Peak water depth\nAverage Peak Difference (AvgPeakDiff)", "zero", 4, 100, "cm"),
    ("maximum_depth_r2_mean", "Peak water depth\nCoefficient of determination (R²)", "higher", 5, 100, "percentage points"),
    ("fidelity_index_mean", "Water depth hydrographs\nFidelity Index (FI)", "higher", 4, 100, "percentage points"),
)

# User-approved report terminology; stored metric keys and calculations are unchanged.
METRIC_NOTES = (
    "AvgPeakDiff = reference minus predicted peak depth: positive = underprediction; negative = overprediction."
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(comparison, stage="initial"):
    with (comparison / "test_metric_summary.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    evaluated = {r["candidate"]: r for r in records}
    plan = load_plan(stage=stage)
    names = {r["name"] for r in plan}
    temperatures = sorted({r["temperature_m"] for r in plan if r["extent_weight"] > 0})
    if len(records) != len(evaluated) or not set(evaluated) <= names or "control" not in evaluated:
        raise ValueError("Invalid held-out summary coverage")
    pending = sorted(names - set(evaluated))
    if stage == "initial" and pending:
        raise ValueError("Initial heatmaps require all 31 held-out evaluations")
    verification = json.loads((comparison / "evaluation_verification.json").read_text())
    if verification["complete"] != (not pending) or sorted(verification["missing"]) != pending:
        raise ValueError("Held-out verification disagrees with available results")
    with (comparison / "validation_ranking.csv").open(newline="") as handle:
        ranking_records = list(csv.DictReader(handle))
    ranking = {r["candidate"]: r for r in ranking_records}
    if len(ranking_records) != len(plan) or set(ranking) != names:
        raise ValueError("Complete validation ranking required for the selected stage")
    rows = {}
    for item in plan:
        name = item["name"]
        row = rows[name] = dict(ranking[name])
        if any(float(row[k]) != item[k] for k in ("temperature_m", "extent_weight")):
            raise ValueError(f"Summary coordinates differ from plan: {name}")
        directory = comparison.parent / name / "fold_1/seed_42"
        status = json.loads((directory / "training_status.json").read_text())
        with (directory / "history.csv").open(newline="") as handle:
            history = list(csv.DictReader(handle))
        values = np.array([float(r["validation_depth_rmse_m"]) for r in history])
        if status["state"] != "complete" or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Incomplete or invalid history: {name}")
        if not np.isclose(float(row["validation_rmse_m"]), values.min(), rtol=1e-12):
            raise ValueError(f"Ranking disagrees with validation history: {name}")
        if name not in evaluated:
            row.update({key: float("nan") for key, *_ in PANELS if key != "validation_rmse_m"})
            continue
        source_row = evaluated[name]
        if any(float(source_row[k]) != item[k] for k in ("temperature_m", "extent_weight")):
            raise ValueError(f"Held-out coordinates differ from plan: {name}")
        row.update(source_row)
        if sha256(directory / "test_event_metrics.csv") != verification["metrics_sha256"][name]:
            raise ValueError(f"Source metrics changed: {name}")
        with (directory / "test_event_metrics.csv").open(newline="") as handle:
            events = list(csv.DictReader(handle))
        for key, *_ in PANELS:
            value = float(row[key])
            if key == "validation_rmse_m":
                expected = float(ranking[name][key])
            else:
                metric = key.removesuffix("_mean")
                values = np.array([float(e[metric]) for e in events])
                finite = values[np.isfinite(values)]
                if len(finite) != 18 or int(row[f"{metric}_n"]) != 18:
                    raise ValueError("Expected 18 finite event metrics in every cell")
                expected = np.mean(finite)
            if not np.isfinite(value) or not np.isclose(value, expected, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Summary value disagrees with source: {name}, {key}")
    grid = {
        (float(r["temperature_m"]), float(r["extent_weight"])): r
        for name, r in rows.items()
        if name != "control"
    }
    winner = min(rows.values(), key=lambda r: float(r["validation_rmse_m"]))
    winner_cell = (
        None
        if winner["candidate"] == "control"
        else (
            WEIGHTS.index(float(winner["extent_weight"])),
            temperatures.index(float(winner["temperature_m"])),
        )
    )
    metadata = {
        "selected_by": "Validation RMSE only",
        "selected_candidate": winner["candidate"],
        "rows_temperature_m": temperatures,
        "stage": stage,
        "pending_heldout_evaluations": pending,
        "validation_ranking_sha256": sha256(comparison / "validation_ranking.csv"),
        "columns_extent_weight": WEIGHTS,
        "test_aggregation": "Arithmetic mean of 18 event metrics; equal event weight",
        "summary_sha256": sha256(comparison / "test_metric_summary.csv"),
        "panels": {},
        "metric_definitions_note": METRIC_NOTES,
    }
    for mode in ("values", "improvement"):
        fig, axes = plt.subplots(2, 3, figsize=(17, 11.5 + 0.6 * (len(temperatures) - 6)), layout="constrained")
        for ax, (key, title, preference, precision, scale, units) in zip(axes.flat, PANELS):
            raw = np.array([[float(grid[t, w][key]) for w in WEIGHTS] for t in temperatures])
            baseline = float(rows["control"][key])
            gain = (
                raw - baseline
                if preference == "higher"
                else baseline - raw
                if preference == "lower"
                else abs(baseline) - np.abs(raw)
            ) * scale
            metadata["panels"][key] = {
                "control": baseline,
                "raw": [[float(v) if np.isfinite(v) else None for v in r] for r in raw],
                "improvement": [[float(v) if np.isfinite(v) else None for v in r] for r in gain],
                "improvement_units": units,
                "better": preference,
            }
            if mode == "values":
                colored = np.abs(raw) if preference == "zero" else raw
                norm = Normalize(vmin=float(np.nanmin(colored)), vmax=float(np.nanmax(colored)))
                cmap = plt.get_cmap("Blues" if preference == "higher" else "Blues_r")
                displayed = raw
                direction = (
                    "higher is better"
                    if preference == "higher"
                    else (
                        "lower is better"
                        if preference == "lower"
                        else "mean bias closer to 0 is better"
                    )
                )
                unit = (
                    "m"
                    if key in ("validation_rmse_m", "rmse_m_mean", "average_peak_difference_m_mean")
                    else "dimensionless"
                )
                bar_label = f"{'Absolute mean bias' if preference == 'zero' else 'Value'} ({unit})"
                subtitle = f"Control: {baseline:.{precision}f} | {direction}"
            else:
                colored = displayed = gain
                limit = max(float(np.nanmax(np.abs(gain))), 1e-12)
                norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
                cmap = plt.get_cmap("RdBu")
                precision = 2
                bar_label = f"Improvement ({units})"
                subtitle = "Blue: improvement | red: deterioration"
            cmap = cmap.copy()
            cmap.set_bad("#dedede")
            mesh = ax.imshow(np.ma.masked_invalid(colored), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            for i in range(len(temperatures)):
                for j in range(len(WEIGHTS)):
                    if not np.isfinite(displayed[i, j]):
                        ax.text(j, i, "Pending", ha="center", va="center", fontsize=8, color="#555555")
                        continue
                    rgb = cmap(norm(colored[i, j]))[:3]
                    luminance = sum(a * b for a, b in zip(rgb, (0.2126, 0.7152, 0.0722)))
                    text = (
                        f"{displayed[i, j]:+.{precision}f}"
                        if mode == "improvement"
                        else f"{displayed[i, j]:.{precision}f}"
                    )
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if luminance < 0.5 else "#111111",
                    )
            if winner_cell is not None:
                x, y = winner_cell
                for linewidth, edge in ((4, "white"), (1.8, "black")):
                    ax.add_patch(
                        Rectangle(
                            (x - 0.48, y - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor=edge,
                            linewidth=linewidth,
                        )
                    )
            ax.set_xticks(range(len(WEIGHTS)), [f"{w:g}" for w in WEIGHTS])
            ax.set_yticks(range(len(temperatures)), [f"{t:g}" for t in temperatures])
            ax.set_xlabel("Extent weight λ")
            ax.set_ylabel("Temperature T (m)")
            ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=10)
            bar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.03)
            bar.set_label(bar_label, fontsize=8)
            bar.ax.tick_params(labelsize=8)
            bar.locator = MaxNLocator(5)
            bar.update_ticks()
        main_title = (
            "Temperature–weight grid: measured values"
            if mode == "values"
            else "Temperature–weight grid: improvement over MSE control"
        )
        selection_label = (
            "Validation selected the MSE control (outside the grid)"
            if winner_cell is None
            else f"Outlined cell: validation-selected setting (T={float(winner['temperature_m']):g} m, "
            f"λ={float(winner['extent_weight']):g})"
        )
        fig.suptitle(main_title + "\n" + selection_label, fontsize=16)
        fig.supxlabel(
            "Test cells: means across 18 held-out events. Equal-spaced categorical axes; independent colour scales.\n"
            "AvgPeakDiff colour: |event-mean bias| in values; reduction in |event-mean bias| in improvement.\n"
            + METRIC_NOTES
            + (f"\nGrey cells: held-out evaluation pending for {len(pending)} new runs; validation results are complete." if pending else ""),
            fontsize=9,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(comparison / f"temperature_weight_{mode}.{suffix}", dpi=180)
        plt.close(fig)
    (comparison / "heatmap_data.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Plotted {len(plan)-1} grid cells per panel; {len(pending)} held-out evaluations pending: {comparison}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", type=Path
    )
    parser.add_argument("--stage", choices=("initial", "all"), default="initial")
    args = parser.parse_args()
    comparison = args.comparison or ROOT / "outputs/v2/composite_loss" / (
        "comparison" if args.stage == "initial" else "comparison_extended"
    )
    generate(comparison, args.stage)
