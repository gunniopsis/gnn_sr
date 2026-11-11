"""Basic overlaid histogram of verified LF/GNN maximum-depth differences."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from export_maximum_extent_gis import DEFAULT_COMPARISON, ROOT, sha256

MODELS = [("HECRAS_LF", "LF HEC-RAS", "#B56727"), ("GNN_SR", "GNN-SR", "#252A83")]


def load_rasters(directory: Path):
    metadata_path = directory / "maximum_depth_signed_difference_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    arrays, hashes, summaries = {}, {}, {}
    common_mask, common_grid = None, None
    for model, label, _ in MODELS:
        matches = [(name, record) for name, record in metadata["outputs"].items() if record["model"] == model]
        if len(matches) != 1:
            raise ValueError(f"Expected one source for {model}")
        name, record = matches[0]
        path = directory / name
        hashes[name] = sha256(path)
        if hashes[name] != record["sha256"]:
            raise ValueError(f"Raster checksum mismatch: {name}")
        with rasterio.open(path) as src:
            mask = src.read_masks(1) > 0
            grid = (src.crs, src.transform, src.width, src.height)
            if common_mask is not None and (grid != common_grid or not np.array_equal(mask, common_mask)):
                raise ValueError("Source grids/masks differ")
            if src.units != ("m",):
                raise ValueError("Expected depth differences in metres")
            common_mask, common_grid = mask, grid
            values = src.read(1)[mask].astype(np.float64)
        if not np.isfinite(values).all() or len(values) != metadata["valid_cells"]:
            raise ValueError("Invalid values or sample count")
        arrays[model] = values
        summaries[model] = {"label": label, "cells": len(values), "zero_cells": int((values == 0).sum()),
                            "minimum_m": float(values.min()), "maximum_m": float(values.max())}
    return metadata, arrays, hashes, summaries


def load_all_events(directory: Path):
    completion = json.loads((directory / "completion.json").read_text())
    events = [Path(name).stem for name in completion["prediction_sha256"]]
    if completion["state"] != "complete" or completion["events"] != 18 or len(set(events)) != 18:
        raise ValueError("Expected a completed evaluation of 18 distinct events")
    hashes = {}

    def verified(name):
        path = directory / name
        hashes[name] = sha256(path)
        if hashes[name] != completion["artifacts"][name]:
            raise ValueError(f"Checksum mismatch: {name}")
        return path

    with np.load(verified("mapping_geometry.npz"), allow_pickle=False) as geometry:
        ids, mask = geometry["fine_global_ids"], geometry["scoring_mask"]
        if mask.dtype != np.bool_ or mask.shape != ids.shape:
            raise ValueError("Invalid scoring mask")
        if not np.array_equal(np.sort(ids[mask]), np.sort(geometry["lsg_wet_global_ids"])):
            raise ValueError("Scoring mask and LSG IDs disagree")
    parts = {model: [] for model, _, _ in MODELS}
    for event in events:
        reference = None
        for model, _, _ in MODELS:
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
                # Match the precision of the original signed difference GeoTIFFs.
                values = (pred[mask].astype(np.float64) - ref[mask].astype(np.float64)).astype(np.float32)
                parts[model].append(values)
    arrays = {model: np.concatenate(values).astype(np.float64) for model, values in parts.items()}
    summaries = {}
    for model, label, _ in MODELS:
        values = arrays[model]
        if len(values) != int(mask.sum()) * 18:
            raise ValueError("Pooled sample count mismatch")
        summaries[model] = {"label": label, "cells": len(values), "zero_cells": int((values == 0).sum()),
                            "minimum_m": float(values.min()), "maximum_m": float(values.max())}
    metadata = {"event": "all_18_test_events", "events": events,
                "formula": "model maximum depth - HF TUFLOW maximum depth, separately for each event",
                "aggregation": "Pool all event-cell differences, with each event contributing the same number of scored cells",
                "valid_cells_per_event": int(mask.sum()), "checkpoint_epoch": completion["checkpoint_epoch"],
                "completion_sha256": sha256(directory / "completion.json")}
    return metadata, arrays, hashes, summaries


def plot(directory: Path, output: Path, all_events: bool = False):
    metadata, arrays, hashes, summaries = load_all_events(directory) if all_events else load_rasters(directory)
    return render_histogram(metadata, arrays, hashes, summaries, output, all_events)


def render_histogram(metadata, arrays, hashes, summaries, output: Path, all_events=False):
    width = .025
    view = (-1.1, 3.0)
    # Symmetric bin around zero avoids arbitrarily grouping exact zeros with one sign.
    lower_index = int(np.floor(min(v.min() for v in arrays.values()) / width - .5))
    upper_index = int(np.ceil(max(v.max() for v in arrays.values()) / width - .5))
    edges = (np.arange(lower_index, upper_index + 1, dtype=float) + .5) * width
    output.mkdir(parents=True, exist_ok=True)
    records = []
    plt.rcdefaults()
    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.subplots_adjust(left=.12, right=.98, bottom=.16, top=.96)
    for model, label, color in MODELS:
        values = arrays[model]
        counts, _ = np.histogram(values, bins=edges)
        if counts.sum() != len(values):
            raise ValueError("Histogram does not account for every valid cell")
        ax.hist(values, bins=edges, label=label, color=color, alpha=.65,
                edgecolor=color, linewidth=.3)
        outside = int(np.count_nonzero((values < view[0]) | (values > view[1])))
        summaries[model].update(outside_display_range_cells=outside,
                                outside_display_range_percent=100 * outside / len(values))
        for i, count in enumerate(counts):
            records.append({"model": label, "lower_m": round(float(edges[i]), 8),
                            "upper_m": round(float(edges[i + 1]), 8), "cell_count": int(count),
                            "percent": 100 * int(count) / len(values)})
    ax.set_xlim(*view)
    ax.set_xticks(np.round(np.arange(-.9, 3.01, .3), 1))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Maximum depth difference (m)")
    ax.set_ylabel("Frequency")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    legend = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(.5)
    stem = output / "maximum_depth_signed_difference_histogram"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    with (output / "histogram_bin_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    details = {"event": metadata["event"], "formula": metadata["formula"], "dpi": 300,
               "bin_width_m": width, "zero_bin_m": [-width / 2, width / 2], "display_range_m": view,
               "bin_range_m": [float(edges[0]), float(edges[-1])],
               "bin_convention": "Left-inclusive, right-exclusive except final upper edge",
               "zero_cells": "Included as valid values; NoData excluded",
               "source_sha256": hashes, "models": summaries,
               "verification": "Input checksums, common masks/grids, units, finite values and histogram totals verified"}
    if all_events:
        details.update(metadata)
        details["verification"] = "All 37 source hashes, scoring mask, per-event reference agreement, finite nonnegative depths, evaluation times and pooled histogram totals verified"
    if "mask_definition" in metadata:
        details.update(metadata)
    (output / "histogram_metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    return stem.with_suffix(".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-events", action="store_true", help="Pool maximum-depth differences from all 18 test events")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.directory or (DEFAULT_COMPARISON if args.all_events else DEFAULT_COMPARISON / "qgis/e1971_real_slmax_1in100")
    event_folder = "all_18_test_events" if args.all_events else "e1971_real_slmax_1in100"
    output = args.output or ROOT / "reports/figures/full_domain_comparison" / event_folder / "signed_depth_difference_distribution"
    print(plot(directory.resolve(), output.resolve(), all_events=args.all_events))
