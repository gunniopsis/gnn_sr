"""Plot peak-depth differences where any of HF, LF or GNN is wet."""
import argparse
import json
from pathlib import Path

import numpy as np
from export_maximum_extent_gis import DEFAULT_COMPARISON, ROOT, sha256
from plot_signed_depth_difference_histogram import MODELS, render_histogram, plt
from plot_depth_difference_classes import CLASSES, LABELS
from plot_all_test_depth_difference_classes import classify, write_csv


def plot(comparison, event, output):
    completion = json.loads((comparison / "completion.json").read_text())
    hashes = {}

    def verified(name):
        path = comparison / name
        hashes[name] = sha256(path)
        if hashes[name] != completion["artifacts"][name]:
            raise ValueError(f"Checksum mismatch: {name}")
        return path

    with np.load(verified("mapping_geometry.npz")) as g:
        ids, scoring = g["fine_global_ids"], g["scoring_mask"]
        if scoring.dtype != np.bool_ or not np.array_equal(np.sort(ids[scoring]), np.sort(g["lsg_wet_global_ids"])):
            raise ValueError("LSG mask mismatch")
    depths = {}
    for model, _, _ in MODELS:
        suffix = "_lf_peak_maps.npz" if model == "HECRAS_LF" else "_peak_maps.npz"
        with np.load(verified(event + suffix)) as d:
            ref, pred = d["reference_peak_depth_m"], d["predicted_peak_depth_m"]
            if any(a.shape != ids.shape or not np.isfinite(a).all() or np.any(a < 0) for a in (ref, pred)):
                raise ValueError("Invalid peak depths")
            if any(not np.array_equal(d[k], np.arange(48, 431)) for k in ("time", "source_timestep_index")):
                raise ValueError("Unexpected evaluation times")
            if "HF" in depths and not np.array_equal(depths["HF"], ref):
                raise ValueError("References differ")
            depths["HF"], depths[model] = ref, pred
    wet_union = np.logical_or.reduce([depth >= .03 for depth in depths.values()])
    mask = scoring & wet_union
    n = int(mask.sum())
    if n == 0 or np.any(mask & ~scoring):
        raise ValueError("Invalid union mask")
    arrays, summaries = {}, {}
    for model, label, _ in MODELS:
        values = (depths[model][mask].astype(np.float64) - depths["HF"][mask].astype(np.float64)).astype(np.float32).astype(np.float64)
        arrays[model] = values
        summaries[model] = {"label": label, "cells": n, "zero_cells": int((values == 0).sum()),
                            "minimum_m": float(values.min()), "maximum_m": float(values.max())}
    metadata = {"event": event, "formula": "model maximum depth - HF TUFLOW maximum depth",
                "mask_definition": "Original LSG fold-1 scoring mask AND (HF maximum depth >= 0.03 m OR LF maximum depth >= 0.03 m OR GNN maximum depth >= 0.03 m)",
                "wet_threshold_m": .03, "original_scoring_cells": int(scoring.sum()), "included_cells": n,
                "excluded_all_three_dry_cells": int((scoring & ~wet_union).sum()),
                "same_mask_for_both_models": True, "zero_cells": "Retained if at least one of the three models is wet",
                "evaluation_hours": [48, 430], "dpi": 300,
                "verification": "Source hashes, original scoring mask, common HF reference, evaluation times, finite nonnegative peak depths and common wet-union mask verified"}
    render_histogram(metadata, arrays, hashes, summaries, output)
    plt.rcdefaults()
    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.96)
    rows = []
    for index, (model, label, color) in enumerate(MODELS):
        counts = classify(arrays[model])
        percentages = counts / n * 100
        assert np.isclose(percentages.sum(), 100)
        bars = ax.bar(np.arange(5) + (index - .5) * .36, percentages, width=.36,
                      label=label, color=color, alpha=.85, edgecolor="black", linewidth=.5)
        ax.bar_label(bars, labels=[f"{p:.1f}" for p in percentages], padding=3, fontsize=9)
        for category, count, percent in zip(CLASSES, counts, percentages):
            rows.append({"model": label, "class_m": category, "cell_count": int(count), "area_percent": float(percent)})
    ax.set_xticks(np.arange(5), LABELS, fontsize=9)
    ax.set_xlabel("Maximum depth difference (m)")
    ax.set_ylabel("Area percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    legend = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(.5)
    stem = output / "maximum_depth_difference_classes"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    write_csv(stem.with_suffix(".csv"), rows)
    metadata.update(source_sha256=hashes, classes_m=CLASSES)
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="e1971_real_slmax_1in100")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "reports/figures/full_domain_comparison" / args.event / "wet_union_depth_difference_distribution"
    plot(args.comparison, args.event, output)
