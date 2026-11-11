"""Export unstyled HF/LF/GNN maximum-depth GeoTIFFs on the existing comparison grid.

Requires numpy and rasterio; preserves saved physical peak depths exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from export_maximum_extent_gis import DEFAULT_COMPARISON, sha256


def export(comparison: Path, event: str) -> Path:
    directory = comparison / "qgis" / event
    extent_metadata_path = directory / "export_metadata.json"
    extent_metadata = json.loads(extent_metadata_path.read_text())
    completion = json.loads((comparison / "completion.json").read_text())
    if completion["state"] != "complete" or extent_metadata["event"] != event:
        raise ValueError("Expected a completed comparison and matching classification export")
    sources = ["mapping_geometry.npz", f"{event}_peak_maps.npz", f"{event}_lf_peak_maps.npz"]
    hashes = {}
    for name in sources:
        hashes[name] = sha256(comparison / name)
        if hashes[name] != completion["artifacts"][name] or hashes[name] != extent_metadata["source_sha256"][name]:
            raise ValueError(f"Source checksum mismatch: {name}")
    with np.load(comparison / sources[0], allow_pickle=False) as source:
        xy, ids = source["xy"], source["fine_global_ids"]
        mask, crs = source["scoring_mask"], str(source["crs"].item())
        if not np.array_equal(np.sort(ids[mask]), np.sort(source["lsg_wet_global_ids"])):
            raise ValueError("Scoring mask does not match the saved LSG cell IDs")
    if xy.shape != (len(ids), 2) or mask.shape != ids.shape or not np.isfinite(xy).all():
        raise ValueError("Invalid geometry dimensions or coordinates")

    class_rasters = {}
    template_profile = None
    for name, record in extent_metadata["outputs"].items():
        path = directory / name
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Classification raster checksum mismatch: {name}")
        hashes[name] = record["sha256"]
        with rasterio.open(path) as src:
            if template_profile is None:
                template_profile = src.profile.copy()
            if any(src.profile[k] != template_profile[k] for k in ("crs", "transform", "width", "height")):
                raise ValueError("Classification grids differ")
            if src.count != 1 or src.nodata != 255 or src.crs.to_string() != crs:
                raise ValueError("Unexpected classification raster metadata")
            class_rasters[record["model"]] = src.read(1)
    if set(class_rasters) != {"LF", "GNN-SR"}:
        raise ValueError("Both LF and GNN classification maps are required")
    transform = template_profile["transform"]
    rows, cols = rasterio.transform.rowcol(transform, xy[:, 0], xy[:, 1])
    rows, cols = np.asarray(rows), np.asarray(cols)
    shape = (template_profile["height"], template_profile["width"])
    if np.any(rows < 0) or np.any(rows >= shape[0]) or np.any(cols < 0) or np.any(cols >= shape[1]):
        raise ValueError("Source cells fall outside the classification raster")
    if len(np.unique(rows * shape[1] + cols)) != len(ids):
        raise ValueError("Duplicate raster cell locations")
    cx, cy = rasterio.transform.xy(transform, rows, cols)
    if not np.allclose(np.column_stack([cx, cy]), xy, rtol=0, atol=1e-6):
        raise ValueError("Source cell centres do not align with the classification grid")
    valid_pixels = np.zeros(shape, dtype=bool)
    valid_pixels[rows[mask], cols[mask]] = True
    if any(not np.array_equal(raster != 255, valid_pixels) for raster in class_rasters.values()):
        raise ValueError("Classification raster mask differs from the source mask")

    depths = {}
    common_reference = None
    for label, model, name in [("GNN_SR", "GNN-SR", sources[1]), ("HECRAS_LF", "LF", sources[2])]:
        with np.load(comparison / name, allow_pickle=False) as source:
            ref, pred = source["reference_peak_depth_m"], source["predicted_peak_depth_m"]
            for values in (ref, pred):
                if values.shape != ids.shape or values.dtype != np.float32 or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError(f"Invalid physical peak depths: {name}")
            if not np.array_equal(source["source_timestep_index"], np.arange(48, 431)) or not np.array_equal(source["time"], np.arange(48, 431)):
                raise ValueError("Unexpected event times")
            if common_reference is not None and not np.array_equal(common_reference, ref):
                raise ValueError("LF and GNN reference depths differ")
            common_reference = ref.copy()
            expected = np.zeros(len(ids), dtype=np.uint8)
            wet_ref, wet_pred = ref >= 0.03, pred >= 0.03
            expected[wet_ref & wet_pred] = 1
            expected[~wet_ref & wet_pred] = 2
            expected[wet_ref & ~wet_pred] = 3
            if not np.array_equal(expected, source["maximum_extent_class"]) or not np.array_equal(expected[mask], class_rasters[model][rows[mask], cols[mask]]):
                raise ValueError(f"Peak depths disagree with existing classifications: {model}")
            depths[label] = (pred.copy(), name, "predicted_peak_depth_m")
    depths["TUFLOW_HF"] = (common_reference, sources[1], "reference_peak_depth_m")

    profile = template_profile.copy()
    profile.update(dtype="float32", nodata=-9999.0, compress="deflate", predictor=3)
    outputs = {}
    for label in ("TUFLOW_HF", "HECRAS_LF", "GNN_SR"):
        values, name, key = depths[label]
        grid = np.full(shape, -9999.0, dtype=np.float32)
        grid[rows[mask], cols[mask]] = values[mask]
        path = directory / f"{event}_{label}_maximum_depth_m.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(grid, 1)
            dst.set_band_description(1, f"{label} maximum flood depth (m)")
            dst.set_band_unit(1, "m")
            dst.update_tags(event=event, model=label, quantity="maximum physical flood depth", units="m",
                            time_hours="48 through 430 inclusive", timesteps="383", source_file=name,
                            source_array=key, dry_cells="0 m retained as valid data",
                            spatial_support="Common fold-1 evaluation mask", cell_count=str(int(mask.sum())))
        with rasterio.open(path) as check:
            if check.crs != profile["crs"] or check.transform != transform or check.units != ("m",):
                raise ValueError("Output georeferencing or units mismatch")
            if not np.array_equal(check.read(1), grid) or not np.array_equal(check.read_masks(1) > 0, valid_pixels):
                raise ValueError("Output pixel values or NoData mask failed roundtrip verification")
        outputs[path.name] = {"model": label, "source": name, "source_array": key,
                              "minimum_m": float(values[mask].min()), "maximum_m": float(values[mask].max()),
                              "zero_depth_cells": int(np.count_nonzero(values[mask] == 0)),
                              "wet_cells_at_0p03_m": int(np.count_nonzero(values[mask] >= 0.03)),
                              "sha256": sha256(path)}
    details = {"event": event, "crs": crs, "pixel_size_m": 20, "units": "m", "nodata": -9999,
               "valid_cells": int(mask.sum()), "timesteps": 383, "time_hours": [48, 430],
               "dry_cells": "Zero is valid data; original peak values are preserved without new thresholding",
               "LF_representation": "Parent-projected HEC-RAS on HF cells with original fine-terrain dry processing",
               "gnn_checkpoint_epoch": completion["checkpoint_epoch"],
               "gnn_checkpoint_sha256": completion["checkpoint_sha256"],
               "source_sha256": hashes, "outputs": outputs,
               "verification": "Source hashes, common reference, grid alignment, scoring mask, existing classifications, units and exact output pixels verified"}
    (directory / "maximum_depth_metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    (directory / "maximum_depth_README.md").write_text(
        f"# Maximum flood depth: {event}\n\n"
        "Three unstyled Float32 GeoTIFFs for HF TUFLOW, LF HEC-RAS and GNN-SR. Drag them into QGIS. "
        "All share EPSG:28356 (GDA94 / MGA zone 56), 20 m pixels and the same extent and alignment as the classification maps.\n\n"
        "Values are physical flood depths in metres above local terrain, not water-surface elevations. "
        "Each cell contains its own temporal maximum across the 383 evaluated hourly samples (hours 48-430 after excluding warm-up). "
        "This is not a simultaneous snapshot. Saved peak values are preserved exactly with no new thresholding or interpolation.\n\n"
        "Zero-depth cells remain valid zero values for analysis. NoData is -9999, covering cells outside the "
        "common 769,709-cell evaluation mask and outside the supplied geometry. Do not interpret NoData as dry.\n\n"
        "LF is the existing parent-projected HEC-RAS baseline on the fine grid, with the original fine-terrain dry processing; "
        "it is not a native coarse-mesh depth map. GNN-SR is the full-domain c2_f2 validation-selected model. "
        "HF references in the LF and GNN source files were checked to be identical.\n\n"
        "The adjacent maximum_depth_metadata.json contains hashes, extrema and wet-cell counts. "
        "The three rasters have identical valid-data masks for direct raster calculations.\n"
    )
    print(json.dumps(outputs, indent=2))
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--event", default="e1971_real_slmax_1in100")
    args = parser.parse_args()
    print(export(args.comparison.resolve(), args.event))
