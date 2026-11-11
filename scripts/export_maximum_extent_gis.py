"""Export hash-verified Burnett LF/GNN peak classifications as unstyled GeoTIFFs.

Requires numpy and rasterio. No interpolation, inference, or source edits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = ROOT / "outputs/v2/full_domain/c2_f2/fold_1/seed_42/lsg_wet_mask_comparison"
CLASS_CODES = {0: "TN", 1: "TP", 2: "FP", 3: "FN"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(comparison: Path, event: str) -> Path:
    completion = json.loads((comparison / "completion.json").read_text())
    if completion["state"] != "complete":
        raise ValueError("Comparison is not complete")
    sources = ["mapping_geometry.npz", f"{event}_peak_maps.npz", f"{event}_lf_peak_maps.npz"]
    source_hashes = {}
    for name in sources:
        actual = sha256(comparison / name)
        if actual != completion["artifacts"][name]:
            raise ValueError(f"Downloaded file fails checksum verification: {name}")
        source_hashes[name] = actual
    with np.load(comparison / sources[0], allow_pickle=False) as data:
        xy = data["xy"]
        ids = data["fine_global_ids"]
        scoring_mask = data["scoring_mask"]
        crs = str(data["crs"].item())
        if not np.array_equal(np.sort(ids[scoring_mask]), np.sort(data["lsg_wet_global_ids"])):
            raise ValueError("Scoring mask and stored LSG cell IDs disagree")
    if xy.shape != (len(ids), 2) or scoring_mask.shape != ids.shape:
        raise ValueError("Geometry dimensions disagree")
    if not np.isfinite(xy).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid coordinates or duplicate IDs")
    if crs != "EPSG:28356":
        raise ValueError(f"Unexpected Burnett CRS: {crs}")

    spacing = 20.0
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    cols_float = (xy[:, 0] - xmin) / spacing
    rows_float = (ymax - xy[:, 1]) / spacing
    cols, rows = np.rint(cols_float).astype(int), np.rint(rows_float).astype(int)
    if not np.allclose(cols_float, cols, rtol=0, atol=1e-6) or not np.allclose(rows_float, rows, rtol=0, atol=1e-6):
        raise ValueError("Coordinates are not on the expected 20 m grid")
    width, height = int(cols.max()) + 1, int(rows.max()) + 1
    if len(np.unique(rows * width + cols)) != len(ids):
        raise ValueError("Multiple source cells occupy the same raster pixel")
    transform = from_origin(xmin - spacing / 2, ymax + spacing / 2, spacing, spacing)
    with (comparison / "event_metrics.csv").open(newline="") as stream:
        metrics = {r["model"]: r for r in csv.DictReader(stream) if r["event_name"] == event}

    # Validate both inputs before writing either output.
    prepared = []
    common_reference = None
    for model, label, source in [("GNN-SR", "GNN_SR", sources[1]), ("LF", "HECRAS_LF", sources[2])]:
        with np.load(comparison / source, allow_pickle=False) as data:
            classes = data["maximum_extent_class"]
            reference = data["reference_peak_depth_m"]
            predicted = data["predicted_peak_depth_m"]
            if any(a.shape != ids.shape for a in (classes, reference, predicted)):
                raise ValueError(f"{model}: unexpected peak-map dimensions")
            if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
                raise ValueError(f"{model}: nonfinite peak depths")
            wet_ref, wet_pred = reference >= 0.03, predicted >= 0.03
            expected = np.zeros(len(ids), dtype=np.uint8)
            expected[wet_ref & wet_pred] = 1
            expected[~wet_ref & wet_pred] = 2
            expected[wet_ref & ~wet_pred] = 3
            if not np.array_equal(classes, expected):
                raise ValueError(f"{model}: saved classes disagree with peak depths")
            if not np.array_equal(data["source_timestep_index"], np.arange(48, 431)):
                raise ValueError(f"{model}: unexpected evaluated timesteps")
            if not np.array_equal(data["time"], np.arange(48, 431)):
                raise ValueError(f"{model}: unexpected event times")
            if common_reference is not None and not np.array_equal(reference, common_reference):
                raise ValueError("LF and GNN reference peaks differ")
            common_reference = reference.copy()
        counts = {name: int(np.count_nonzero(classes[scoring_mask] == code)) for code, name in CLASS_CODES.items()}
        if any(counts[k] != int(metrics[model][k]) for k in counts):
            raise ValueError(f"{model}: class counts do not match saved evaluation metrics")
        csi = counts["TP"] / (counts["TP"] + counts["FP"] + counts["FN"])
        if not np.isclose(csi, float(metrics[model]["maximum_extent_csi"]), rtol=0, atol=1e-12):
            raise ValueError(f"{model}: CSI does not match saved metrics")
        prepared.append((model, label, source, classes, counts, csi))

    output = comparison / "qgis" / event
    output.mkdir(parents=True, exist_ok=True)
    exported = {}
    for model, label, source, classes, counts, csi in prepared:
        grid = np.full((height, width), 255, dtype=np.uint8)
        grid[rows[scoring_mask], cols[scoring_mask]] = classes[scoring_mask]
        path = output / f"{event}_{label}_maximum_extent_classes.tif"
        with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                           count=1, dtype="uint8", crs=crs, transform=transform,
                           nodata=255, compress="deflate", tiled=True) as dst:
            dst.write(grid, 1)
            dst.set_band_description(1, "Maximum inundation extent classification")
            dst.update_tags(event=event, model=model, reference="TUFLOW", threshold_m="0.03",
                            time_hours="48 through 430 inclusive", scoring_cells=str(int(scoring_mask.sum())),
                            class_codes="0=TN;1=TP;2=FP;3=FN;255=NoData", source_file=source,
                            checkpoint_epoch=str(completion["checkpoint_epoch"]) if model == "GNN-SR" else "not applicable")
        with rasterio.open(path) as check:
            actual = check.read(1)
            if check.crs.to_epsg() != 28356 or check.transform != transform or check.nodata != 255:
                raise ValueError(f"Raster georeferencing failed: {path}")
            if not np.array_equal(actual, grid):
                raise ValueError(f"Raster pixel roundtrip failed: {path}")
            cx, cy = rasterio.transform.xy(check.transform, rows, cols)
            if not np.allclose(np.column_stack([cx, cy]), xy, rtol=0, atol=1e-6):
                raise ValueError(f"Raster cell-centre alignment failed: {path}")
        exported[path.name] = {"model": model, "source": source, "counts": counts,
                               "maximum_extent_csi": csi, "sha256": sha256(path)}
    metadata = {"event": event, "crs": crs, "pixel_size_m": spacing,
                "width": width, "height": height, "class_codes": CLASS_CODES, "nodata": 255,
                "scoring_cells": int(scoring_mask.sum()), "source_cells": len(ids),
                "excluded_source_cells": int((~scoring_mask).sum()),
                "threshold_m": 0.03, "time_hours": [48, 430], "timesteps": 383,
                "gnn_checkpoint_epoch": completion["checkpoint_epoch"],
                "gnn_checkpoint_sha256": completion["checkpoint_sha256"],
                "source_sha256": source_hashes, "outputs": exported,
                "verification": "Source hashes, classes from peaks, shared reference, metric counts/CSI, raster pixels and cell centres verified"}
    (output / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.md").write_text(
        f"# Maximum inundation extent: {event}\n\n"
        "Two unstyled single-band GeoTIFFs: HECRAS_LF versus TUFLOW and GNN_SR versus TUFLOW.\n"
        "Drag the .tif files into QGIS. Native CRS is EPSG:28356 (GDA94 / MGA zone 56), with 20 m pixels.\n"
        "No resampling, interpolation or colour styling has been applied. Use nearest-neighbour resampling for these categorical data.\n\n"
        "| Value | Meaning |\n|---|---|\n| 0 | TN: both dry |\n| 1 | TP: both flooded |\n"
        "| 2 | FP: model flooded, TUFLOW dry |\n| 3 | FN: model dry, TUFLOW flooded |\n"
        "| 255 | NoData: outside the common scoring mask or outside the supplied cells |\n\n"
        "Wet means peak depth >= 0.03 m across the 383 evaluated hourly timesteps (hours 48-430). "
        "The maxima are computed separately at each cell and for each model; this is not a single-time flood snapshot.\n\n"
        "Both maps use the same 769,709-cell LSG fold-1 scoring mask. The other 11,076 supplied cells are NoData, "
        "not classified as dry. TN cells retain value 0 for your own styling and processing.\n\n"
        "GNN-SR uses the validation-selected full-domain validation-selected checkpoint. LF is the existing HEC-RAS "
        "baseline projected onto the same fine cells using the saved model parent mapping and fine-terrain dry processing; "
        "it is not a raster of native coarse-cell polygons.\n\n"
        "For display, choose Paletted/Unique values and Classify in QGIS. Set class 0 transparent if desired. "
        "For polygon processing, polygonize the raster then select values 1, 2 and 3.\n\n"
        "Source checksums, class counts, CSI and output checksums are recorded in export_metadata.json.\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--event", default="e1971_real_slmax_1in100")
    args = parser.parse_args()
    print(export(args.comparison.resolve(), args.event))
