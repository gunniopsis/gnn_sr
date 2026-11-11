"""Verify downloaded depth trials and plot six-panel AOI grids locally."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from plan_depth_experiments import COARSE, FINE, ROOT, load_plan
from plot_loss_heatmaps import METRIC_NOTES, PANELS
from run_width_experiment import comparison_settings


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(root):
    records, evidence, pending = [], {}, []
    common = None
    event_support = None
    for item in load_plan():
        name = item["name"]
        directory = root / name / "fold_1/seed_42"
        status = read_json(directory / "training_status.json")
        config = read_json(directory / "config.json")
        provenance = read_json(directory / "depth_experiment.json")
        history = read_csv(directory / "history.csv")
        require(status["state"] == "complete", f"Incomplete training: {name}")
        require(
            [int(r["epoch"]) for r in history] == list(range(1, status["epoch"] + 1)),
            f"Discontinuous history: {name}",
        )
        require(
            all(np.isfinite(float(r["validation_depth_rmse_m"])) for r in history),
            f"Nonfinite validation RMSE: {name}",
        )
        best = min(history, key=lambda r: float(r["validation_depth_rmse_m"]))
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        require(
            checkpoint["epoch"] == int(best["epoch"])
            and checkpoint["validation_depth_rmse_m"] == float(best["validation_depth_rmse_m"])
            and status["best_validation_depth_rmse_m"] == checkpoint["validation_depth_rmse_m"],
            f"Selected checkpoint/history mismatch: {name}",
        )
        require(
            config["model"]["hidden_dim"] == 64
            and all(config["model"][k] == item[k] for k in ("coarse_layers", "fine_layers"))
            and config["loss"]
            == {"temperature_m": 0.3, "extent_weight": 0.01, "record_diagnostics": True}
            and config["graph"]["coarse_halo_hops"] == 4
            and config["graph"]["fine_halo_hops"] == 6
            and all(
                config["training"][f"debug_max_{s}_samples"] is None
                for s in ("train", "validation", "test")
            ),
            f"Configuration differs from approved experiment: {name}",
        )
        require(
            config["model"] == checkpoint["config"]["model"]
            and config["loss"] == checkpoint["config"]["loss"] == provenance["loss"]
            and comparison_settings(config)
            == comparison_settings(checkpoint["config"])
            == provenance["settings"],
            f"Checkpoint/provenance mismatch: {name}",
        )
        norm_name = Path(checkpoint["normalization_path"].replace("\\", "/")).name
        hashes = {
            "checkpoint_sha256": digest(directory / "best.pt"),
            "checkpoint_epoch": int(best["epoch"]),
            "normalization_sha256": digest(directory / norm_name),
            "history_sha256": digest(directory / "history.csv"),
        }
        shared = copy.deepcopy(provenance)
        shared.pop("trainable_parameters")
        for key in ("coarse_layers", "fine_layers"):
            shared["settings"]["model"].pop(key)
        shared["normalization_sha256"] = hashes["normalization_sha256"]
        if common is None:
            common = shared
        require(shared == common, f"Non-architecture settings/support differ: {name}")
        record = {
            "candidate": name,
            "coarse_layers": item["coarse_layers"],
            "fine_layers": item["fine_layers"],
            "best_epoch": int(best["epoch"]),
            "final_epoch": status["epoch"],
            "validation_rmse_m": float(best["validation_depth_rmse_m"]),
            **{key: None for key, *_ in PANELS if key != "validation_rmse_m"},
        }
        marker = directory / "depth_evaluation.json"
        if not marker.exists():
            pending.append(name)
        else:
            saved = read_json(marker)
            require(
                all(
                    saved[k] == hashes[k]
                    for k in ("checkpoint_sha256", "checkpoint_epoch", "normalization_sha256")
                ),
                f"Evaluation belongs to another checkpoint: {name}",
            )
            events = read_csv(directory / "test_event_metrics.csv")
            support = sorted(
                (
                    r["event_name"],
                    int(r["event_index"]),
                    int(r["group"]),
                    int(r["n_timesteps"]),
                    int(r["n_target_cells"]),
                )
                for r in events
            )
            require(
                len(support) == len({r[0] for r in support}) == saved["events"] == 18
                and all(r[2:] == (1, 383, 2314) for r in support),
                f"Wrong event support: {name}",
            )
            if event_support is None:
                event_support = support
            require(support == event_support, f"Different held-out events: {name}")
            expected = {"test_event_metrics.csv", "test_summary.json"} | {
                f"predictions/{r['event_name']}.npz" for r in events
            }
            require(set(saved["artifacts"]) == expected, f"Artifact coverage mismatch: {name}")
            for relative in sorted(expected):
                require(
                    digest(directory / relative) == saved["artifacts"][relative],
                    f"Downloaded artifact hash mismatch: {name}/{relative}",
                )
            require(
                saved["metrics_sha256"] == saved["artifacts"]["test_event_metrics.csv"],
                f"Metric hash mismatch: {name}",
            )
            summary = read_json(directory / "test_summary.json")
            require(summary["events"] == 18, f"Wrong summary count: {name}")
            for key, *_ in PANELS[1:]:
                metric = key.removesuffix("_mean")
                values = np.array([float(r[metric]) for r in events])
                require(np.isfinite(values).all(), f"Undefined event metric: {name}/{metric}")
                mean = float(values.mean())
                require(
                    np.isclose(mean, summary["mean_event_metrics"][metric], rtol=1e-10, atol=1e-12),
                    f"Summary differs from event records: {name}/{metric}",
                )
                record[key] = mean
            hashes["evaluation_marker_sha256"] = digest(marker)
            hashes["verified_artifacts"] = saved["artifacts"]
        records.append(record)
        evidence[name] = hashes
        print(
            f"Verified {name}: {'test pending' if name in pending else '18 test events'}",
            flush=True,
        )
    return records, {
        "verified": True,
        "validation_models": len(records),
        "evaluated_models": len(records) - len(pending),
        "pending_heldout_evaluations": pending,
        "selection": "Minimum pooled validation depth RMSE only",
        "test_aggregation": "Arithmetic mean over the same 18 held-out events",
        "checks": "History, selected checkpoint, shared provenance and normalization; "
        "SHA-256 of all evaluation metrics, summaries and prediction archives. "
        "No inference or metric recomputation from predictions performed locally.",
        "common_provenance": common,
        "runs": evidence,
    }


def generate(root):
    records, audit = collect(root)
    grid = {(r["coarse_layers"], r["fine_layers"]): r for r in records}
    winner = min(records, key=lambda r: r["validation_rmse_m"])
    baseline = next(r for r in records if r["candidate"] == "c2_f3")
    output = root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metric_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    audit["selected_candidate"] = winner["candidate"]
    audit["reference_candidate"] = baseline["candidate"]
    audit["rows_coarse_layers"], audit["columns_fine_layers"] = COARSE, FINE
    audit["axis_orientation"] = "Fine layers increase rightward; coarse layers increase upward"
    audit["panels"] = {}
    audit["metric_definitions_note"] = METRIC_NOTES
    for mode in ("values", "improvement"):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), layout="constrained")
        for ax, (key, title, direction, precision, scale, units) in zip(axes.flat, PANELS):
            raw = np.array(
                [
                    [grid[c, f][key] if grid[c, f][key] is not None else np.nan for f in FINE]
                    for c in COARSE
                ]
            )
            base = baseline[key]
            require(base is not None, "Reference c2_f3 evaluation is required")
            gain = (
                raw - base
                if direction == "higher"
                else base - raw
                if direction == "lower"
                else abs(base) - np.abs(raw)
            ) * scale
            if mode == "values":
                colored = np.abs(raw) if direction == "zero" else raw
                displayed = raw
                norm = Normalize(float(np.nanmin(colored)), float(np.nanmax(colored)))
                cmap = plt.get_cmap("Blues" if direction == "higher" else "Blues_r").copy()
                preference = {
                    "higher": "higher is better",
                    "lower": "lower is better",
                    "zero": "mean bias closer to 0 is better",
                }[direction]
                subtitle = preference + " | darker blue is better"
                unit = "m" if direction in ("lower", "zero") else "dimensionless"
                label = f"{'Absolute mean bias' if direction == 'zero' else 'Value'} ({unit})"
            else:
                displayed = colored = gain
                limit = max(float(np.nanmax(np.abs(gain))), 1e-12)
                norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
                cmap = plt.get_cmap("RdBu").copy()
                precision = 2
                subtitle = f"Reference c2/f3: {base:.4f}"
                label = f"Improvement ({units})"
            cmap.set_bad("#dedede")
            mesh = ax.imshow(
                np.ma.masked_invalid(colored),
                cmap=cmap,
                norm=norm,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
            )
            for i in range(len(COARSE)):
                for j in range(len(FINE)):
                    value = displayed[i, j]
                    if not np.isfinite(value):
                        ax.text(j, i, "Pending", ha="center", va="center", color="#555555")
                        continue
                    rgb = cmap(norm(colored[i, j]))[:3]
                    light = np.dot(rgb, (0.2126, 0.7152, 0.0722))
                    text = (
                        f"{value:+.{precision}f}"
                        if mode == "improvement"
                        else f"{value:.{precision}f}"
                    )
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if light < 0.5 else "#111111",
                    )
            x, y = FINE.index(winner["fine_layers"]), COARSE.index(winner["coarse_layers"])
            for width, color in ((4, "white"), (1.8, "black")):
                ax.add_patch(
                    Rectangle(
                        (x - 0.48, y - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        linewidth=width,
                        edgecolor=color,
                    )
                )
            ax.set(
                xticks=range(len(FINE)),
                xticklabels=FINE,
                yticks=range(len(COARSE)),
                yticklabels=COARSE,
                xlabel="Fine message-passing layers",
                ylabel="Coarse message-passing layers",
            )
            ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=10)
            bar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.03)
            bar.set_label(label, fontsize=8)
            bar.ax.tick_params(labelsize=8)
            audit["panels"][key] = {
                "raw": [[None if not np.isfinite(v) else float(v) for v in r] for r in raw],
                "reference": base,
                "improvement": [
                    [None if not np.isfinite(v) else float(v) for v in r] for r in gain
                ],
                "improvement_units": units,
            }
        heading = (
            "measured values" if mode == "values" else "improvement over fresh c2/f3 reference"
        )
        fig.suptitle(
            f"Message-passing layer grid: {heading}\n"
            f"Width 64 | T=0.3 m | λ=0.01 | Outline: validation leader "
            f"c{winner['coarse_layers']}/f{winner['fine_layers']}",
            fontsize=16,
        )
        pending = audit["pending_heldout_evaluations"]
        fig.supxlabel(
            "Test cells: equal-event means over 18 held-out events; validation-RMSE-selected checkpoints.\n"
            "Common 4/6-hop halo; metrics inside the 1 km AOI. Independent colour scales; categorical axes.\n"
            + ("Blue: improvement; red: deterioration. " if mode == "improvement" else "")
            + "AvgPeakDiff colour reflects "
            + ("reduction in |mean bias|." if mode == "improvement" else "|mean bias|.")
            + "\n" + METRIC_NOTES
            + (
                f"\nGrey cells: test evaluation pending for {', '.join(pending)}."
                if pending
                else ""
            ),
            fontsize=9,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"coarse_fine_{mode}.{suffix}", dpi=200)
        plt.close(fig)
    (output / "heatmap_verification.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved {output}; pending: {audit['pending_heldout_evaluations']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/v2/message_passing")
    generate(parser.parse_args().root)
