"""Export unstyled absolute or signed errors between saved maximum-depth GeoTIFFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from export_maximum_extent_gis import DEFAULT_COMPARISON, sha256


def export(directory: Path, signed: bool = False) -> Path:
    kind = "signed_difference" if signed else "absolute_error"
    description = "signed maximum-depth difference" if signed else "absolute maximum-depth error"
    formula = "model maximum depth - HF TUFLOW maximum depth"
    if not signed:
        formula = f"abs({formula})"
    metadata_path = directory / "maximum_depth_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    arrays, profiles, masks, hashes = {}, {}, {}, {}
    for name, record in metadata["outputs"].items():
        model = record["model"]
        path = directory / name
        hashes[name] = sha256(path)
        if hashes[name] != record["sha256"]:
            raise ValueError(f"Source raster checksum mismatch: {name}")
        with rasterio.open(path) as src:
            if src.count != 1 or src.units != ("m",) or src.nodata != -9999:
                raise ValueError(f"Unexpected units, bands or NoData: {name}")
            arrays[model] = src.read(1)
            profiles[model] = src.profile.copy()
            masks[model] = src.read_masks(1) > 0
        values = arrays[model][masks[model]]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid physical depths: {name}")
    if set(arrays) != {"TUFLOW_HF", "HECRAS_LF", "GNN_SR"}:
        raise ValueError("Expected exactly the HF, LF and GNN maximum-depth rasters")
    reference = arrays["TUFLOW_HF"]
    valid = masks["TUFLOW_HF"]
    profile = profiles["TUFLOW_HF"].copy()
    if np.count_nonzero(valid) != metadata["valid_cells"]:
        raise ValueError("Valid cell count disagrees with source metadata")
    for model in ("HECRAS_LF", "GNN_SR"):
        if not np.array_equal(masks[model], valid):
            raise ValueError(f"Valid-data masks differ: {model}")
        if any(profiles[model][k] != profile[k] for k in ("crs", "transform", "width", "height")):
            raise ValueError(f"Input rasters are not aligned: {model}")

    profile.update(dtype="float32", nodata=-9999.0, compress="deflate", predictor=3)
    outputs = {}
    for model in ("HECRAS_LF", "GNN_SR"):
        # Subtract only valid pixels; dry cells (0 m) remain included.
        difference = arrays[model][valid].astype(np.float64) - reference[valid].astype(np.float64)
        error = (difference if signed else np.abs(difference)).astype(np.float32)
        grid = np.full(reference.shape, -9999.0, dtype=np.float32)
        grid[valid] = error
        path = directory / f"{metadata['event']}_{model}_vs_TUFLOW_HF_maximum_depth_{kind}_m.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(grid, 1)
            dst.set_band_description(1, f"{model} {description} against HF TUFLOW (m)")
            dst.set_band_unit(1, "m")
            dst.update_tags(event=metadata["event"], model=model, reference="TUFLOW_HF", units="m",
                            formula=formula,
                            time_hours="48 through 430 inclusive", timesteps="383",
                            spatial_support="Common fold-1 evaluation mask", dry_cells="Included, with depth 0 m")
            if signed:
                dst.update_tags(sign_convention="Positive=overestimation; negative=underestimation; zero=agreement")
        with rasterio.open(path) as check:
            if check.crs != profile["crs"] or check.transform != profile["transform"] or check.units != ("m",):
                raise ValueError("Output georeferencing or units mismatch")
            actual = check.read(1)
            if not np.array_equal(actual, grid) or not np.array_equal(check.read_masks(1) > 0, valid):
                raise ValueError("Output values or NoData mask failed roundtrip verification")
            if not signed and np.any(actual[valid] < 0):
                raise ValueError("Negative absolute error")
            if signed and not np.array_equal(np.sign(actual[valid]), np.sign(difference)):
                raise ValueError("Signed differences have incorrect directions")
            equal_depth = valid & (arrays[model] == reference)
            if np.any(actual[equal_depth] != 0):
                raise ValueError("Equal depths must give zero error")
        outputs[path.name] = {"model": model, "reference": "TUFLOW_HF", "sha256": sha256(path),
                              "minimum_m": float(error.min()), "maximum_m": float(error.max()),
                              "mean_absolute_peak_depth_error_m": float(error.mean(dtype=np.float64)),
                              "zero_error_cells": int(np.count_nonzero(error == 0))}
        # The absolute-error export must agree exactly with the signed magnitudes.
        if signed:
            record = outputs[path.name]
            record.pop("mean_absolute_peak_depth_error_m")
            record.update(mean_signed_peak_depth_difference_m=float(error.mean(dtype=np.float64)),
                          mean_absolute_peak_depth_error_m=float(np.abs(error).mean(dtype=np.float64)),
                          overestimated_cells=int(np.count_nonzero(error > 0)),
                          underestimated_cells=int(np.count_nonzero(error < 0)))
            abs_name = f"{metadata['event']}_{model}_vs_TUFLOW_HF_maximum_depth_absolute_error_m.tif"
            abs_metadata_path = directory / "maximum_depth_absolute_error_metadata.json"
            if (directory / abs_name).exists() and abs_metadata_path.exists():
                abs_metadata = json.loads(abs_metadata_path.read_text())
                if sha256(directory / abs_name) != abs_metadata["outputs"][abs_name]["sha256"]:
                    raise ValueError("Existing absolute-error raster checksum mismatch")
                with rasterio.open(directory / abs_name) as absolute:
                    if absolute.transform != profile["transform"] or absolute.crs != profile["crs"]:
                        raise ValueError("Existing absolute-error raster grid mismatch")
                    if not np.array_equal(absolute.read_masks(1) > 0, valid) or not np.array_equal(absolute.read(1)[valid], np.abs(error)):
                        raise ValueError("Signed magnitudes differ from the existing absolute errors")
                record["verified_against_absolute_error_sha256"] = sha256(directory / abs_name)
    details = {"event": metadata["event"], "formula": formula,
               "crs": metadata["crs"], "pixel_size_m": metadata["pixel_size_m"], "units": "m",
               "nodata": -9999, "valid_cells": int(np.count_nonzero(valid)),
               "time_hours": metadata["time_hours"], "timesteps": metadata["timesteps"],
               "source_metadata_sha256": sha256(metadata_path), "source_sha256": hashes,
               "outputs": outputs, "verification": "Source hashes, grid/mask alignment, units, sign/nonnegativity, zero-error cells and exact raster roundtrip checked"}
    if signed:
        details["sign_convention"] = "Positive = model overestimation; negative = model underestimation"
        details["relation_to_saved_peak_maps"] = "Opposite sign to peak_depth_difference_m and AvgPeakDiff, which use reference minus predicted"
    (directory / f"maximum_depth_{kind}_metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    meaning = (
        "Positive values mean the model overestimates TUFLOW maximum depth; negative values mean underestimation. "
        "Zero means agreement. This is opposite to the reference-minus-predicted convention in the source peak_depth_difference_m and AvgPeakDiff.\n\n"
        if signed else ""
    )
    (directory / f"maximum_depth_{kind}_README.md").write_text(
        f"# {description.capitalize()}: {metadata['event']}\n\n"
        "Two unstyled Float32 GeoTIFFs: LF HEC-RAS against HF TUFLOW, and GNN-SR against HF TUFLOW. "
        f"Drag the .tif files into QGIS. Values are metres: {formula}.\n\n"
        + meaning +
        "These compare each model's independent temporal maximum at each cell across hours 48-430 inclusive "
        "(383 evaluated timesteps). They are not maximum instantaneous errors or time-series MAE.\n\n"
        "EPSG:28356, 20 m pixels, exactly aligned with the source depth rasters. All 769,709 common evaluation "
        "cells are included, including dry cells with depth 0 m. Zero error is valid data. NoData=-9999 "
        "outside the shared evaluation mask; NoData is not treated as zero. No styling or resampling is applied.\n\n"
        "LF remains the parent-projected HEC-RAS baseline on the fine grid. The original three depth rasters "
        f"are unchanged. Source/output checksums and error statistics are in maximum_depth_{kind}_metadata.json.\n"
    )
    print(json.dumps(outputs, indent=2))
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=DEFAULT_COMPARISON / "qgis" / "e1971_real_slmax_1in100")
    parser.add_argument("--signed", action="store_true",
                        help="Export model minus reference: positive overestimation, negative underestimation")
    args = parser.parse_args()
    print(export(args.directory.resolve(), signed=args.signed))
