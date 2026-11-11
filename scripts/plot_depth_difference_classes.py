"""Plot area percentages in five signed maximum-depth difference classes."""
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
from plot_signed_depth_difference_histogram import MODELS

CLASSES = ["d < -0.30", "-0.30 <= d < -0.05", "-0.05 <= d <= 0.05",
           "0.05 < d <= 0.30", "d > 0.30"]
LABELS = ["< -0.30\nUnderestimated", "-0.30 to -0.05\nComparable",
          "-0.05 to 0.05\nComparable", "0.05 to 0.30\nComparable",
          "> 0.30\nOverestimated"]


def plot(directory: Path, output: Path):
    metadata = json.loads((directory / "maximum_depth_signed_difference_metadata.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    common_mask, common_grid = None, None
    plt.rcdefaults()
    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.96)
    for index, (model, label, color) in enumerate(MODELS):
        matches = [(name, record) for name, record in metadata["outputs"].items()
                   if record["model"] == model]
        if len(matches) != 1:
            raise ValueError(f"Expected one raster for {model}")
        name, record = matches[0]
        path = directory / name
        hashes[name] = sha256(path)
        if hashes[name] != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
        with rasterio.open(path) as src:
            mask = src.read_masks(1) > 0
            grid = (src.crs, src.transform, src.width, src.height)
            if common_mask is not None and (grid != common_grid or not np.array_equal(mask, common_mask)):
                raise ValueError("Source grids or masks differ")
            if src.units != ("m",):
                raise ValueError("Expected metres")
            common_mask, common_grid = mask, grid
            values = src.read(1)[mask].astype(np.float64)
            cell_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
        if not np.isfinite(values).all() or len(values) != metadata["valid_cells"]:
            raise ValueError("Invalid values or sample count")
        counts = np.array([
            np.count_nonzero(values < -.30),
            np.count_nonzero((values >= -.30) & (values < -.05)),
            np.count_nonzero((values >= -.05) & (values <= .05)),
            np.count_nonzero((values > .05) & (values <= .30)),
            np.count_nonzero(values > .30),
        ])
        if counts.sum() != len(values):
            raise ValueError("Classes must account for every valid cell exactly once")
        percentages = counts / len(values) * 100
        assert np.isclose(percentages.sum(), 100)
        bars = ax.bar(np.arange(5) + (index - .5) * .36, percentages, width=.36,
                      label=label, color=color, alpha=.85, edgecolor="black", linewidth=.5)
        ax.bar_label(bars, labels=[f"{p:.1f}" for p in percentages], padding=3, fontsize=9)
        for category, count, percent in zip(CLASSES, counts, percentages):
            rows.append({"model": label, "class_m": category, "cell_count": int(count),
                         "area_km2": int(count) * cell_area / 1e6, "area_percent": float(percent)})
    ax.set_xticks(np.arange(5), LABELS, fontsize=9)
    ax.set_xlabel("Maximum depth difference (m)")
    ax.set_ylabel("Area percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    legend = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(.5)
    stem = output / "maximum_depth_difference_classes"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    details = {"event": metadata["event"], "formula": metadata["formula"],
               "classes_m": CLASSES, "valid_cells_per_model": metadata["valid_cells"],
               "denominator": "All valid cells on the common scoring mask, including zero differences",
               "area_method": "Equal-area raster cells; percentage of cells equals percentage of area",
               "source_sha256": hashes,
               "verification": "Source hashes, common grids/masks, units, finite values and exhaustive class totals verified"}
    stem.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    return stem.with_suffix(".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_COMPARISON / "qgis/e1971_real_slmax_1in100")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/full_domain_comparison/e1971_real_slmax_1in100/signed_depth_difference_distribution")
    args = parser.parse_args()
    print(plot(args.directory.resolve(), args.output.resolve()))
