"""Local held-out reporting in fixed grid order; never select models on test data."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from compare_hidden_widths import metric_rows
from matplotlib.backends.backend_pdf import PdfPages
from plot_fold1_boxplots import PANELS, create_figure
from run_width_experiment import sha256, write_json


def verified_metrics(trial):
    directory = trial["directory"]
    marker = directory / "loss_evaluation.json"
    if not marker.exists():
        return None
    evidence = json.loads(marker.read_text())
    path = directory / "test_event_metrics.csv"
    if (
        evidence["checkpoint_sha256"] != trial["checkpoint_sha256"]
        or evidence["normalization_sha256"] != trial["normalization_sha256"]
        or evidence["metrics_sha256"] != sha256(path)
        or evidence["checkpoint_epoch"] != int(trial["best"]["epoch"])
    ):
        raise ValueError(f"Evaluation differs from selected checkpoint/metrics: {trial['name']}")
    rows = metric_rows(path)
    with path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    if any(int(row["group"]) != 1 for row in raw):
        raise ValueError("Expected held-out Fold 1 group")
    summary = json.loads((directory / "test_summary.json").read_text())
    if summary["events"] != 18:
        raise ValueError("Expected 18 summary events")
    for panel in PANELS:
        values = np.array([r[panel.key] for r in rows.values()])
        finite = values[np.isfinite(values)]
        saved = summary["mean_event_metrics"][panel.key]
        if len(finite):
            if saved is None or not np.isclose(np.mean(finite), saved, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Summary disagrees with event metrics: {trial['name']}")
        elif saved is not None:
            raise ValueError("Undefined mean must be null")
    return rows


def report_heldout(trials, plan, output):
    temperatures = sorted({c["temperature_m"] for c in plan if c["extent_weight"] > 0})
    evaluated = {}
    by_name = {t["name"]: t for t in trials}
    for trial in trials:
        rows = verified_metrics(trial)
        if rows is not None:
            evaluated[trial["name"]] = rows
    order = [item["name"] for item in plan if item["name"] in evaluated]
    missing = [item["name"] for item in plan if item["name"] not in evaluated]
    if not order:
        write_json(output / "evaluation_verification.json", {"complete": False, "missing": missing})
        return
    reference = evaluated[order[0]]
    events = sorted(reference)
    for name in order:
        rows = evaluated[name]
        if sorted(rows) != events or any(
            rows[event][key] != reference[event][key]
            for event in events
            for key in ("n_timesteps", "n_target_cells")
        ):
            raise ValueError(f"Held-out event supports differ: {name}")
    summary_rows = []
    for name in order:
        trial = by_name[name]
        row = {
            "candidate": name,
            "temperature_m": trial["config"]["loss"]["temperature_m"],
            "extent_weight": trial["config"]["loss"]["extent_weight"],
            "checkpoint_epoch": int(trial["best"]["epoch"]),
            "validation_rmse_m": float(trial["best"]["validation_depth_rmse_m"]),
        }
        for panel in PANELS:
            values = np.array([evaluated[name][e][panel.key] for e in events])
            finite = values[np.isfinite(values)]
            row[f"{panel.key}_n"] = len(finite)
            row[f"{panel.key}_mean"] = float(finite.mean()) if len(finite) else None
            row[f"{panel.key}_median"] = float(np.median(finite)) if len(finite) else None
        summary_rows.append(row)
    with (output / "test_metric_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    colors = {t: plt.get_cmap("tab10")(i) for i, t in enumerate(temperatures)}
    color = lambda n: (
        "#555555" if n == "control" else colors[by_name[n]["config"]["loss"]["temperature_m"]]
    )
    label = lambda n: (
        "MSE control"
        if n == "control"
        else (
            f"T={by_name[n]['config']['loss']['temperature_m']:g}, "
            f"λ={by_name[n]['config']['loss']['extent_weight']:g}"
        )
    )
    with PdfPages(output / "all_models_boxplots.pdf") as pdf:
        for panel in PANELS:
            fig, ax = plt.subplots(
                figsize=(10, max(5, 0.30 * len(order) + 2)), layout="constrained"
            )
            arrays = [np.array([evaluated[n][e][panel.key] for e in events]) for n in order]
            arrays = [v[np.isfinite(v)] for v in arrays]
            boxes = ax.boxplot(
                arrays,
                vert=False,
                patch_artist=True,
                widths=0.65,
                medianprops={"color": "black"},
                flierprops={"markersize": 3},
            )
            for box, name in zip(boxes["boxes"], order):
                box.set_facecolor(color(name))
                box.set_alpha(0.65)
            ax.set_yticks(range(1, len(order) + 1), [label(n) for n in order], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(panel.ylabel)
            ax.set_title(
                f"{panel.title}: {len(order)} models × 18 held-out events\n"
                "Best validation-RMSE checkpoints; fixed temperature/weight order\n"
                "Dotted line: control median; dashed line: ideal value"
            )
            ax.grid(axis="x", alpha=0.25)
            if panel.ideal is not None:
                ax.axvline(panel.ideal, color="#666666", linestyle="--", linewidth=0.8)
            if "control" in evaluated:
                values = [evaluated["control"][e][panel.key] for e in events]
                ax.axvline(
                    np.nanmedian(values), color="#333333", linestyle=":", label="Control median"
                )
            fig.savefig(output / f"all_models_{panel.key}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    # Compact five-metric figures: control and five weights per temperature.
    for temperature in temperatures:
        names = (["control"] if "control" in evaluated else []) + [
            n
            for n in order
            if n != "control" and by_name[n]["config"]["loss"]["temperature_m"] == temperature
        ]
        if len(names) < 2:
            continue
        labels = {
            n: "MSE" if n == "control" else f"{by_name[n]['config']['loss']['extent_weight']:g}"
            for n in names
        }
        mapped = {labels[n]: evaluated[n] for n in names}
        palette = {
            labels[n]: "#555555" if n == "control" else plt.get_cmap("viridis")(0.15 + 0.16 * i)
            for i, n in enumerate(names)
        }
        stem = f"temperature_{temperature:g}".replace(".", "p")
        create_figure(
            mapped,
            events,
            tuple(mapped),
            output / f"{stem}_boxplots.png",
            output / f"{stem}_boxplots.pdf",
            colors=palette,
            legend_labels={
                label: "MSE control" if label == "MSE" else f"λ={label}" for label in mapped
            },
            title=f"T={temperature:g} m: five held-out metrics (x-axis: loss weight λ)",
        )
    write_json(
        output / "evaluation_verification.json",
        {
            "complete": not missing,
            "evaluated_models": len(order),
            "events_per_model": len(events),
            "missing": missing,
            "selection": "Unchanged validation RMSE; test metrics are reporting only",
            "aggregation": "Equal event means/medians; finite-value counts included",
            "metrics_sha256": {
                n: sha256(by_name[n]["directory"] / "test_event_metrics.csv") for n in order
            },
            "checkpoint_sha256": {n: by_name[n]["checkpoint_sha256"] for n in order},
        },
    )
    print(f"Verified and plotted {len(order)} held-out evaluations")
