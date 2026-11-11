"""Hourly depth RMSE and volume on the fixed HF/LF/GNN event wet-union mask."""
import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator
from plot_temporal_extent_metrics import ROOT, DEFAULT_RUN, MODELS, THRESHOLD, sha256, open_array, plt


def calculate(run, event):
    comparison = run / "lsg_wet_mask_comparison"
    manifest = json.loads((comparison / "completion.json").read_text())
    if manifest["state"] != "complete":
        raise ValueError("Evaluation is incomplete")
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
        hashes[str(path.relative_to(ROOT))] = actual
        print(f"Verified {path.name}", flush=True)
    with np.load(geometry) as g:
        ids, scoring, terrain, xy = g["fine_global_ids"], g["scoring_mask"], g["terrain"], g["xy"]
        if not np.array_equal(np.sort(ids[scoring]), np.sort(g["lsg_wet_global_ids"])):
            raise ValueError("Invalid scoring mask")
    with np.load(gnn_peaks) as p:
        times = p["time"]
        peaks = [p["reference_peak_depth_m"], p["predicted_peak_depth_m"]]
        if not np.array_equal(p["source_timestep_index"], np.arange(48, 431)):
            raise ValueError("Unexpected source timesteps")
    with np.load(lf_peaks) as p:
        if not np.array_equal(p["reference_peak_depth_m"], peaks[0]) or not np.array_equal(p["time"], times):
            raise ValueError("References or times disagree")
        peaks.append(p["predicted_peak_depth_m"])
    if not np.array_equal(times, np.arange(48, 431)):
        raise ValueError("Unexpected times")
    mask = scoring & np.logical_or.reduce([p >= THRESHOLD for p in peaks])
    n = int(mask.sum())
    # Actual supplied cell areas, verified against the common fine-cell coordinates.
    area_path = ROOT / "reference_files/BurnettRV/Geometry_data/Tuflow_Geometry_data.npz"
    with np.load(area_path) as g:
        if not np.array_equal(g["XY_coor"][ids], xy):
            raise ValueError("Cell-area geometry does not align")
        area = g["Area"][ids][mask].astype(np.float64)
    if not np.all(area == 400):
        raise ValueError("Expected uniform 20 m cells")
    hashes[str(area_path.relative_to(ROOT))] = sha256(area_path)
    for path in (prediction, lf_prediction):
        with np.load(path) as d:
            if not np.array_equal(d["fine_global_ids"], ids) or not np.array_equal(d["terrain"], terrain):
                raise ValueError("Prediction alignment mismatch")
            if path == lf_prediction and not np.allclose(d["time_hours"], times, rtol=0, atol=1e-8):
                raise ValueError("LF time mismatch")
    maxima = [np.zeros(len(ids), dtype=np.float32) for _ in range(3)]
    records = []
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
                    raise ValueError("Truncated array")
                block = np.frombuffer(raw, dtype=dtype).reshape(stop-start, len(ids))
                if not np.isfinite(block).all() or np.any(block < 0):
                    raise ValueError("Invalid depths")
                np.maximum(maxima[i], block.max(axis=0), out=maxima[i])
                blocks.append(block[:, mask].astype(np.float64))
            hf, gnn, lf = blocks
            volumes = [np.sum(b * area, axis=1) for b in blocks]
            errors = [np.sqrt(np.mean((b - hf)**2, axis=1)) for b in (lf, gnn)]
            for j, hour in enumerate(times[start:stop]):
                records.append({"event": event, "time_hours": int(hour), "n_cells": n,
                                "LF_RMSE_m": float(errors[0][j]), "GNN_SR_RMSE_m": float(errors[1][j]),
                                "HF_volume_m3": float(volumes[0][j]), "GNN_SR_volume_m3": float(volumes[1][j]),
                                "LF_volume_m3": float(volumes[2][j])})
            if stop % 96 == 0 or stop == len(times):
                print(f"Calculated {stop}/{len(times)} hours", flush=True)
        for stream, _ in streams:
            if stream.read(1):
                raise ValueError("Trailing array bytes")
    if any(not np.array_equal(actual, expected) for actual, expected in zip(maxima, peaks)):
        raise ValueError("Streamed maxima disagree with saved maps")
    output = comparison / "temporal_depth_rmse_volume_wet_union" / event
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "temporal_depth_rmse_volume.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    details = {"event": event, "included_cells": n, "cell_area_m2": 400,
               "mask": "LSG fold-1 mask intersected with cells whose event maximum depth is >= 0.03 m in any of HF TUFLOW, LF HEC-RAS or GNN-SR; fixed across all hours",
               "RMSE": "sqrt(mean((model depth - HF depth)^2)) over the same masked cells at each hour; plotted in cm",
               "volume": "sum(depth_m * cell_area_m2) over the same masked fine cells at each hour; plotted in million cubic metres",
               "depth_handling": "Use saved physical depths, retaining instantaneous dry cells in the fixed mask; no extra per-hour wet filter",
               "LF_support": "Parent-projected LF depths on HF terrain and fine grid, not native HEC-RAS mesh volumes",
               "source_sha256": hashes, "csv_sha256": sha256(csv_path),
               "verification": "Source hashes, geometry alignment, cell areas, time alignment, finite nonnegative depths and exact streamed maxima checked",
               "checkpoint_epoch": manifest["checkpoint_epoch"], "dpi": 300}
    (output / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({"cells": n, "max_LF_RMSE_m": max(r["LF_RMSE_m"] for r in records),
                      "max_GNN_RMSE_m": max(r["GNN_SR_RMSE_m"] for r in records)}, indent=2))
    return records


def plot(records, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), sharex=True)
    fig.subplots_adjust(left=.13, right=.97, top=.97, bottom=.22, hspace=.15)
    times = [r["time_hours"] for r in records]
    for model, label, color, _ in MODELS:
        key = "LF" if model == "LF" else "GNN_SR"
        axes[0].plot(times, [r[key + "_RMSE_m"] * 100 for r in records], label=label, color=color, alpha=.85)
        axes[1].plot(times, [r[key + "_volume_m3"] / 1e6 for r in records], label=label, color=color, alpha=.85)
    axes[1].plot(times, [r["HF_volume_m3"] / 1e6 for r in records], label="HF TUFLOW", color="#333333", linestyle="--", linewidth=1.2)
    for ax, label in zip(axes, ["RMSE (cm)", "Volume (10⁶ m³)"]):
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, color="#D9DEE5", linewidth=.7)
        ax.set_axisbelow(True)
    axes[-1].set_xlim(48, 430)
    axes[-1].set_xticks([48, 96, 144, 192, 240, 288, 336, 384, 430])
    axes[-1].set_xlabel("Simulation time (hours)")
    handles, labels = axes[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, .015),
                        ncol=1, frameon=True, fancybox=False, edgecolor="black", facecolor="white", framealpha=1)
    legend.get_frame().set_linewidth(.5)
    stem = destination / "temporal_depth_RMSE_volume"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--event", default="e1971_real_slmax_1in100")
    args = parser.parse_args()
    records = calculate(args.run, args.event)
    plot(records, ROOT / "reports/figures/full_domain_comparison" / args.event / "temporal_depth_rmse_volume_wet_union")
