"""Create an unstyled LF-to-GNN classification change GeoTIFF (numpy, rasterio)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from export_maximum_extent_gis import DEFAULT_COMPARISON, sha256


def classify_change(lf: np.ndarray, gnn: np.ndarray) -> np.ndarray:
    """Ignore TN->TN; retain transitions involving TN when correctness changes."""
    if lf.shape != gnn.shape:
        raise ValueError("Classification dimensions differ")
    if not np.isin(lf, [0, 1, 2, 3, 255]).all() or not np.isin(gnn, [0, 1, 2, 3, 255]).all():
        raise ValueError("Invalid input classification code")
    valid = lf != 255
    if not np.array_equal(valid, gnn != 255):
        raise ValueError("LF and GNN scoring masks differ")
    # TP/FN indicate wet reference; TN/FP indicate dry reference.
    if np.any(valid & (np.isin(lf, [1, 3]) != np.isin(gnn, [1, 3]))):
        raise ValueError("Input classes imply inconsistent TUFLOW reference wetness")
    same = valid & (lf == gnn) & (lf != 0)
    corrected = valid & np.isin(lf, [2, 3]) & np.isin(gnn, [0, 1])
    worsened = valid & np.isin(lf, [0, 1]) & np.isin(gnn, [2, 3])
    ignored_tn = valid & (lf == 0) & (gnn == 0)
    if not np.array_equal(same | corrected | worsened | ignored_tn, valid):
        raise ValueError("Unclassified valid transition")
    result = np.full(lf.shape, 255, dtype=np.uint8)
    result[same], result[corrected], result[worsened] = 1, 2, 3
    return result


def export(directory: Path) -> Path:
    metadata_path = directory / "export_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    inputs, profiles, hashes = {}, {}, {}
    for filename, record in metadata["outputs"].items():
        model = record["model"]
        if model not in ("LF", "GNN-SR"):
            continue
        path = directory / filename
        actual_hash = sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"Input hash mismatch: {filename}")
        hashes[filename] = actual_hash
        with rasterio.open(path) as src:
            if src.count != 1 or src.nodata != 255:
                raise ValueError("Expected a single-band classification raster with NoData=255")
            inputs[model], profiles[model] = src.read(1), src.profile.copy()
        for code, label in enumerate(["TN", "TP", "FP", "FN"]):
            if np.count_nonzero(inputs[model] == code) != record["counts"][label]:
                raise ValueError(f"Input class count mismatch: {model}/{label}")
    if set(inputs) != {"LF", "GNN-SR"}:
        raise ValueError("Both LF and GNN classification rasters are required")
    for key in ("crs", "transform", "width", "height"):
        if profiles["LF"][key] != profiles["GNN-SR"][key]:
            raise ValueError(f"Rasters are not aligned: {key}")
    lf, gnn = inputs["LF"], inputs["GNN-SR"]
    change = classify_change(lf, gnn)
    valid = lf != 255
    matrix = np.bincount((lf[valid].astype(int) * 4 + gnn[valid]), minlength=16).reshape(4, 4)
    unchanged_tn = int(matrix[0, 0])
    counts = {str(code): int(np.count_nonzero(change == code)) for code in (1, 2, 3)}
    if sum(counts.values()) + unchanged_tn != metadata["scoring_cells"]:
        raise ValueError("Output does not account for every scoring cell")

    name = f"{metadata['event']}_LF_to_GNN_SR_classification_change"
    output = directory / f"{name}.tif"
    with rasterio.open(output, "w", **profiles["LF"]) as dst:
        dst.write(change, 1)
        dst.set_band_description(1, "LF to GNN-SR maximum extent classification change")
        dst.update_tags(event=metadata["event"], direction="HEC-RAS LF to GNN-SR",
                        reference="TUFLOW", threshold_m=str(metadata["threshold_m"]),
                        class_codes="1=unchanged TP/FP/FN;2=incorrect to correct;3=correct to incorrect;255=NoData",
                        nodata_meaning="TN to TN, or outside common evaluation mask")
    with rasterio.open(output) as check:
        if not np.array_equal(check.read(1), change):
            raise ValueError("Output raster pixel roundtrip failed")
        if check.transform != profiles["LF"]["transform"] or check.crs != profiles["LF"]["crs"]:
            raise ValueError("Output georeferencing mismatch")
        if check.nodata != 255 or not np.array_equal(check.read_masks(1) > 0, change != 255):
            raise ValueError("Output NoData mask mismatch")
    labels = {"1": "Unchanged classification: TP->TP, FP->FP or FN->FN",
              "2": "Corrected: FP->TN or FN->TP",
              "3": "Made incorrect: TP->FN or TN->FP"}
    details = {"event": metadata["event"], "class_codes": labels, "counts": counts,
               "ignored_unchanged_TN_cells": unchanged_tn, "nodata": 255,
               "nodata_meaning": "Unchanged TN and cells outside the common evaluation mask",
               "transition_matrix": matrix.tolist(), "matrix_row_order_LF": ["TN", "TP", "FP", "FN"],
               "matrix_column_order_GNN": ["TN", "TP", "FP", "FN"],
               "crs": str(profiles["LF"]["crs"]), "pixel_size_m": metadata["pixel_size_m"],
               "threshold_m": metadata["threshold_m"], "time_hours": metadata["time_hours"],
               "source_sha256": hashes, "source_metadata_sha256": sha256(metadata_path),
               "output_sha256": sha256(output)}
    output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    output.with_suffix(".md").write_text(
        f"# LF to GNN-SR maximum extent classification change\n\nEvent: {metadata['event']}\n\n"
        "Unstyled categorical GeoTIFF, EPSG:28356, 20 m cells. Drag into QGIS.\n\n"
        "| Value | Meaning | Cells |\n|---|---|---|\n"
        + "".join(f"| {code} | {label} | {counts[code]} |\n" for code, label in labels.items())
        + f"\nNoData=255: unchanged TN ({unchanged_tn} cells) and cells outside the common evaluation mask.\n\n"
        "Transitions involving TN are retained when correctness changes: FP->TN is corrected, TN->FP is made incorrect. "
        "Class 1 includes persistent FP/FN as well as persistent TP; it does not mean all those cells are correct.\n\n"
        "The inputs share a TUFLOW reference, the 0.03 m wet threshold, and hours 48-430 inclusive. "
        "No resampling or styling was applied. Use nearest-neighbour resampling for categorical processing. "
        "Source checksums, all 16 transition counts and output checksum are in the adjacent JSON file.\n"
    )
    print(json.dumps({"counts": counts, "ignored_TN": unchanged_tn, "transitions": matrix.tolist()}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=DEFAULT_COMPARISON / "qgis" / "e1971_real_slmax_1in100")
    args = parser.parse_args()
    print(export(args.directory.resolve()))
