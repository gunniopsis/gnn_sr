"""Compute and plot hourly CSI/POD/RFA for verified Burnett LF and GNN predictions.

Streams compressed depth arrays in small batches. Does not run model inference.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/v2/full_domain/c2_f2/fold_1/seed_42"
MODELS = [("LF", "LF HEC-RAS", "#B56727", "--"), ("GNN-SR", "GNN-SR", "#252A83", "-")]
THRESHOLD = 0.03


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratios(tp, fp, fn):
    """Undefined ratios stay NaN rather than being assigned perfect scores."""
    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else float("nan")
    return {"CSI": divide(tp, tp + fp + fn), "POD": divide(tp, tp + fn), "RFA": divide(fp, tp + fp)}


def open_array(stack, path, key, expected_shape):
    archive = stack.enter_context(zipfile.ZipFile(path))
    member = stack.enter_context(archive.open(key + ".npy"))
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    if shape != expected_shape or fortran or dtype != np.dtype("float32"):
        raise ValueError(f"Unexpected depth-array format: {path}:{key}")
    return member, dtype


def calculate(run: Path, event: str, wet_union: bool = False):
    comparison = run / "lsg_wet_mask_comparison"
    manifest = json.loads((comparison / "completion.json").read_text())
    if manifest["state"] != "complete":
        raise ValueError("Evaluation comparison is not complete")
    prediction = run / "predictions" / f"{event}.npz"
    lf_prediction = comparison / "lf_predictions" / f"{event}.npz"
    geometry = comparison / "mapping_geometry.npz"
    gnn_peaks = comparison / f"{event}_peak_maps.npz"
    lf_peaks = comparison / f"{event}_lf_peak_maps.npz"
    sources = [(prediction, manifest["prediction_sha256"][f"predictions/{event}.npz"]),
               (lf_prediction, manifest["artifacts"][f"lf_predictions/{event}.npz"])]
    sources += [(p, manifest["artifacts"][p.name]) for p in (geometry, gnn_peaks, lf_peaks)]
    hashes = {}
    for path, expected in sources:
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        hashes[str(path.relative_to(run))] = actual
        print(f"Verified {path.relative_to(run)}", flush=True)
    with np.load(geometry, allow_pickle=False) as source:
        ids, mask, terrain = source["fine_global_ids"], source["scoring_mask"], source["terrain"]
        if not np.array_equal(np.sort(ids[mask]), np.sort(source["lsg_wet_global_ids"])):
            raise ValueError("Invalid common scoring mask")
    if len(ids) != 780785 or mask.sum() != 769709 or len(np.unique(ids)) != len(ids):
        raise ValueError("Unexpected full-domain geometry or mask")
    with np.load(gnn_peaks, allow_pickle=False) as source:
        times = source["time"]
        indices = source["source_timestep_index"]
        reference_peak = source["reference_peak_depth_m"]
        expected_gnn_peak = source["predicted_peak_depth_m"]
        expected_gnn_class = source["maximum_extent_class"]
    with np.load(lf_peaks, allow_pickle=False) as source:
        expected_lf_peak = source["predicted_peak_depth_m"]
        expected_lf_class = source["maximum_extent_class"]
        if not np.array_equal(source["reference_peak_depth_m"], reference_peak) or not np.array_equal(source["time"], times):
            raise ValueError("LF and GNN peaks have different references or times")
    if not np.array_equal(times, np.arange(48, 431)) or not np.array_equal(indices, np.arange(48, 431)):
        raise ValueError("Unexpected event temporal support")
    scoring_mask = mask.copy()
    if wet_union:
        mask = scoring_mask & ((reference_peak >= THRESHOLD) | (expected_gnn_peak >= THRESHOLD) | (expected_lf_peak >= THRESHOLD))
        print(f"Event wet-union mask: {mask.sum()} cells; removed {scoring_mask.sum() - mask.sum()} all-dry cells", flush=True)
    for path in (prediction, lf_prediction):
        with np.load(path, allow_pickle=False) as source:
            if not np.array_equal(source["fine_global_ids"], ids) or not np.array_equal(source["terrain"], terrain):
                raise ValueError("Prediction column IDs/terrain differ from mapping geometry")
            if path == lf_prediction and not np.allclose(source["time_hours"], times, rtol=0, atol=1e-8):
                raise ValueError("LF times do not match HF times")

    records = []
    maxima = [np.zeros(len(ids), dtype=np.float32) for _ in range(3)]
    with ExitStack() as stack:
        streams = [open_array(stack, path, key, (len(times), len(ids))) for path, key in
                   [(prediction, "target_depth"), (prediction, "predicted_depth"), (lf_prediction, "lf_depth")]]
        for start in range(0, len(times), 8):
            stop = min(start + 8, len(times))
            blocks = []
            for i, (stream, dtype) in enumerate(streams):
                size = (stop - start) * len(ids) * dtype.itemsize
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError("Truncated depth array")
                block = np.frombuffer(raw, dtype=dtype).reshape(stop - start, len(ids))
                if not np.isfinite(block).all() or np.any(block < 0):
                    raise ValueError("Invalid physical depth values")
                np.maximum(maxima[i], block.max(axis=0), out=maxima[i])
                blocks.append(block[:, mask] >= THRESHOLD)
            truth, gnn, lf = blocks
            for model, predicted in [("LF", lf), ("GNN-SR", gnn)]:
                tp = np.count_nonzero(truth & predicted, axis=1)
                fp = np.count_nonzero(~truth & predicted, axis=1)
                fn = np.count_nonzero(truth & ~predicted, axis=1)
                tn = int(mask.sum()) - tp - fp - fn
                for j in range(stop - start):
                    scores = ratios(int(tp[j]), int(fp[j]), int(fn[j]))
                    if any(np.isfinite(v) and not 0 <= v <= 1 for v in scores.values()):
                        raise ValueError("Metric outside [0,1]")
                    records.append({"event": event, "model": model, "time_hours": float(times[start + j]),
                                    "source_timestep_index": int(indices[start + j]), "wet_threshold_m": THRESHOLD,
                                    "n_cells": int(mask.sum()), "TP": int(tp[j]), "FP": int(fp[j]),
                                    "FN": int(fn[j]), "TN": int(tn[j]), **scores})
            if stop % 96 == 0 or stop == len(times):
                print(f"Scored {stop}/{len(times)} hourly samples", flush=True)
        for stream, _ in streams:
            if stream.read(1):
                raise ValueError("Unexpected trailing depth-array bytes")
    # An independent spatial reduction must reproduce the earlier peak-map products.
    for actual, expected in zip(maxima, [reference_peak, expected_gnn_peak, expected_lf_peak], strict=True):
        if not np.array_equal(actual, expected):
            raise ValueError("Streamed timestep maxima differ from verified peak maps")
    with (comparison / "event_metrics.csv").open(newline="") as stream:
        event_rows = {r["model"]: r for r in csv.DictReader(stream) if r["event_name"] == event}
    for model, classes in [("LF", expected_lf_class), ("GNN-SR", expected_gnn_class)]:
        for code, name in enumerate(["TN", "TP", "FP", "FN"]):
            if np.count_nonzero(classes[scoring_mask] == code) != int(event_rows[model][name]):
                raise ValueError("Peak classifications disagree with evaluation summary")
    records.sort(key=lambda row: (row["model"], row["time_hours"]))
    if wet_union:
        previous = comparison / "temporal_extent_metrics" / event
        previous_metadata = json.loads((previous / "metadata.json").read_text())
        if sha256(previous / "temporal_extent_metrics.csv") != previous_metadata["csv_sha256"]:
            raise ValueError("Previous temporal CSV checksum mismatch")
        with (previous / "temporal_extent_metrics.csv").open(newline="") as stream:
            original = {(r["model"], float(r["time_hours"])): r for r in csv.DictReader(stream)}
        removed = int(scoring_mask.sum() - mask.sum())
        for row in records:
            old = original[(row["model"], row["time_hours"])]
            if any(row[k] != int(old[k]) for k in ("TP", "FP", "FN")) or row["TN"] != int(old["TN"]) - removed:
                raise ValueError("Wet-union filtering changed more than all-dry true negatives")
            if any(not np.isclose(row[k], float(old[k]), rtol=0, atol=0, equal_nan=True) for k in ("CSI", "POD", "RFA")):
                raise ValueError("Metrics changed unexpectedly after all-dry removal")
    output = comparison / ("temporal_extent_metrics_wet_union" if wet_union else "temporal_extent_metrics") / event
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "temporal_extent_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({k: "" if isinstance(v, float) and np.isnan(v) else v for k, v in row.items()} for row in records)
    details = {"event": event, "wet_threshold_m": THRESHOLD, "wet_rule": "depth >= 0.03 m",
               "spatial_support": f"{int(mask.sum())} cells; " + ("original LSG fold-1 mask intersected with cells wet in any of HF/LF/GNN at any evaluated timestep" if wet_union else "original LSG fold-1 training-derived scoring mask"),
               "times": "383 hourly samples, hours 48-430 from simulation start; warm-up excluded",
               "formulas": {"CSI": "TP/(TP+FP+FN)", "POD": "TP/(TP+FN)", "RFA": "FP/(TP+FP)"},
               "zero_denominator": "Undefined: blank CSV value and a gap in the plot",
               "metric_reference": "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024WR038424",
               "RFA_note": "False-alarm ratio (also called FAR), not FP/(FP+TN)",
               "checkpoint_epoch": manifest["checkpoint_epoch"], "checkpoint_sha256": manifest["checkpoint_sha256"],
               "source_sha256": hashes, "csv_sha256": sha256(csv_path),
               "verification": "Source hashes, cell order, terrain, times, depth ranges and streamed maxima against both saved peak maps verified",
               "undefined_values": {metric: sum(not np.isfinite(row[metric]) for row in records) for metric in ("CSI", "POD", "RFA")}}
    if wet_union:
        details.update(original_scoring_cells=int(scoring_mask.sum()), included_cells=int(mask.sum()),
                       excluded_all_three_dry_cells=removed, mask_is_fixed_across_time=True,
                       comparison_to_original="All 766 rows have identical TP, FP, FN, CSI, POD and RFA; TN reduced by the excluded cell count",
                       original_csv_sha256=previous_metadata["csv_sha256"])
    (output / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    (output / "README.md").write_text(
        f"# Temporal flood extent metrics: {event}\n\n"
        "LF HEC-RAS and validation-selected full-domain GNN-SR are independently compared with HF TUFLOW at each hour. "
        f"Wet means physical depth >= 0.03 m. Spatial support: {details['spatial_support']}. "
        "The 383 evaluation samples span simulation hours 48-430 after warm-up.\n\n"
        "CSI=TP/(TP+FP+FN), POD=TP/(TP+FN), RFA=FP/(TP+FP). Higher CSI/POD and lower RFA are better. "
        "RFA denotes the false-alarm ratio (often FAR), not the false-positive rate FP/(FP+TN). "
        "A zero denominator is undefined, saved blank in CSV and shown as a plot gap.\n\n"
        "The CSV contains per-hour confusion counts and metrics, without smoothing. "
        "These instantaneous extents differ from the union of flooded cells over the entire event used for maximum-extent CSI. "
        "LF uses the original parent projection onto the common fine cells.\n\n"
        "No model inference, training, or source modifications were performed. Source hashes and checks are in metadata.json.\n"
    )
    return records, csv_path


def plot(records, destination: Path, dpi: int = 200):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.4), sharex=True)
    fig.subplots_adjust(left=.13, right=.97, top=.81, bottom=.13, hspace=.15)
    for ax, metric in zip(axes, ["CSI", "POD", "RFA"], strict=True):
        plotted = []
        for model, label, color, style in MODELS:
            selected = sorted((row for row in records if row["model"] == model), key=lambda row: row["time_hours"])
            times = np.array([row["time_hours"] for row in selected], dtype=float)
            values = np.array([row[metric] for row in selected], dtype=float)
            if len(times) != 383 or not np.array_equal(times, np.arange(48, 431)):
                raise ValueError("Plot is missing expected hourly samples")
            ax.plot(times, values, label=label, color=color, alpha=.85)
            plotted.extend(values[np.isfinite(values)])
        lo, hi = min(plotted), max(plotted)
        padding = max(.015, (hi - lo) * .12)
        ax.set_ylim(max(0, lo - padding) if metric != "RFA" else 0, min(1, hi + padding))
        ax.set_ylabel(metric)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, color="#D9DEE5", linewidth=.7)
        ax.set_axisbelow(True)
    axes[-1].set_xlim(48, 430)
    axes[-1].set_xticks([48, 96, 144, 192, 240, 288, 336, 384, 430])
    axes[-1].set_xlabel("Simulation time (hours)")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, .015),
                        ncol=1, frameon=True, fancybox=False, edgecolor="black",
                        facecolor="white", framealpha=1)
    legend.get_frame().set_linewidth(.5)
    stem = destination / "temporal_extent_CSI_POD_RFA"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--event", default="e1971_real_slmax_1in100")
    parser.add_argument("--wet-union", action="store_true", help="Exclude cells dry in all three models throughout the event")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    rows, csv_path = calculate(args.run.resolve(), args.event, wet_union=args.wet_union)
    folder = "temporal_extent_metrics_wet_union" if args.wet_union else "temporal_extent_metrics"
    png = plot(rows, ROOT / "reports/figures/full_domain_comparison" / args.event / folder, dpi=args.dpi)
    print(f"CSV: {csv_path}\nFigure: {png}")
