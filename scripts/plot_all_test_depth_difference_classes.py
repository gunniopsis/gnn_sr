"""Pool per-event maximum-depth differences across all 18 held-out events."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from plot_depth_difference_classes import CLASSES, LABELS, MODELS, plt
from export_maximum_extent_gis import DEFAULT_COMPARISON, ROOT, sha256


def classify(values):
    counts = np.array([
        np.count_nonzero(values < -.30),
        np.count_nonzero((values >= -.30) & (values < -.05)),
        np.count_nonzero((values >= -.05) & (values <= .05)),
        np.count_nonzero((values > .05) & (values <= .30)),
        np.count_nonzero(values > .30),
    ], dtype=np.int64)
    if counts.sum() != values.size:
        raise ValueError("Classes do not partition all valid cells")
    return counts


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(comparison: Path, output: Path):
    completion_path = comparison / "completion.json"
    completion = json.loads(completion_path.read_text())
    events = [Path(name).stem for name in completion["prediction_sha256"]]
    if completion["state"] != "complete" or completion["events"] != 18 or len(events) != 18:
        raise ValueError("Expected a completed 18-event evaluation")
    hashes = {}

    def verified(name):
        path = comparison / name
        hashes[name] = sha256(path)
        if hashes[name] != completion["artifacts"][name]:
            raise ValueError(f"Checksum mismatch: {name}")
        return path

    with np.load(verified("mapping_geometry.npz"), allow_pickle=False) as geometry:
        ids = geometry["fine_global_ids"]
        mask = geometry["scoring_mask"]
        if mask.dtype != np.bool_ or mask.shape != ids.shape:
            raise ValueError("Invalid scoring mask")
        if not np.array_equal(np.sort(ids[mask]), np.sort(geometry["lsg_wet_global_ids"])):
            raise ValueError("Scoring mask and LSG IDs disagree")
    n = int(mask.sum())
    totals = {model: np.zeros(5, dtype=np.int64) for model, _, _ in MODELS}
    rows = []
    for event in events:
        reference = None
        for model, label, _ in MODELS:
            suffix = "_lf_peak_maps.npz" if model == "HECRAS_LF" else "_peak_maps.npz"
            with np.load(verified(event + suffix), allow_pickle=False) as data:
                ref, pred = data["reference_peak_depth_m"], data["predicted_peak_depth_m"]
                if any(a.shape != ids.shape or not np.isfinite(a).all() or np.any(a < 0) for a in (ref, pred)):
                    raise ValueError(f"Invalid depths: {event} / {model}")
                if any(not np.array_equal(data[k], np.arange(48, 431)) for k in ("time", "source_timestep_index")):
                    raise ValueError(f"Unexpected evaluation times: {event}")
                if reference is not None and not np.array_equal(reference, ref):
                    raise ValueError(f"LF and GNN references differ: {event}")
                reference = ref.copy()
                # Match the existing Float32 signed GeoTIFF calculations exactly.
                difference = (pred[mask].astype(np.float64) - ref[mask].astype(np.float64)).astype(np.float32).astype(np.float64)
            counts = classify(difference)
            totals[model] += counts
            for category, count in zip(CLASSES, counts):
                rows.append({"event": event, "model": label, "class_m": category,
                             "cell_count": int(count), "area_percent": 100 * int(count) / n})

    # Independent regression check against the previously exported single-event chart.
    old_csv = ROOT / "reports/figures/full_domain_comparison/e1971_real_slmax_1in100/signed_depth_difference_distribution/maximum_depth_difference_classes.csv"
    if old_csv.exists():
        with old_csv.open(newline="") as stream:
            old = {(r["model"], r["class_m"]): int(r["cell_count"]) for r in csv.DictReader(stream)}
        for row in rows:
            if row["event"] == "e1971_real_slmax_1in100" and old[(row["model"], row["class_m"])] != row["cell_count"]:
                raise ValueError("Single-event result changed")

    output.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.96)
    pooled_rows = []
    for index, (model, label, color) in enumerate(MODELS):
        counts = totals[model]
        if counts.sum() != n * len(events):
            raise ValueError("Pooled denominator mismatch")
        percentages = counts / counts.sum() * 100
        if not np.isclose(percentages.sum(), 100):
            raise ValueError("Percentages do not sum to 100")
        bars = ax.bar(np.arange(5) + (index - .5) * .36, percentages, width=.36,
                      label=label, color=color, alpha=.85, edgecolor="black", linewidth=.5)
        ax.bar_label(bars, labels=[f"{p:.1f}" for p in percentages], padding=3, fontsize=9)
        for category, count, percent in zip(CLASSES, counts, percentages):
            pooled_rows.append({"model": label, "class_m": category,
                                "event_cell_count": int(count), "area_percent": float(percent)})
    ax.set_xticks(np.arange(5), LABELS, fontsize=9)
    ax.set_xlabel("Maximum depth difference (m)")
    ax.set_ylabel("Area percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    legend = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(.5)
    stem = output / "maximum_depth_difference_classes_all_18_events"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    write_csv(stem.with_suffix(".csv"), pooled_rows)
    write_csv(output / "per_event_depth_difference_classes.csv", rows)
    details = {
        "events": events, "event_count": len(events), "checkpoint_epoch": completion["checkpoint_epoch"],
        "formula": "model maximum depth - HF TUFLOW maximum depth, separately for each event",
        "aggregation": "Pool event-cell class counts, then divide by 18 times the common scored cell count. Equal event weighting because the mask is identical for every event. This is not a maximum across events.",
        "valid_cells_per_event": n, "event_cells_per_model": n * len(events),
        "zero_differences": "Included; outside-mask cells excluded", "classes_m": CLASSES,
        "evaluation_hours": [48, 430], "dpi": 300,
        "completion_sha256": sha256(completion_path), "source_sha256": hashes,
        "verification": "All source hashes, scoring mask, depth validity, common per-event references, evaluation times, exhaustive class totals, and previous single-event counts checked",
    }
    stem.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(pooled_rows, indent=2))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/full_domain_comparison/all_18_test_events/signed_depth_difference_distribution")
    args = parser.parse_args()
    plot(args.comparison.resolve(), args.output.resolve())
