"""Score completed full-domain predictions on the authors' fold-1 wet mask."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from gnnsr.geometry import GeometryData
from gnnsr.metrics import calculate_event_metrics
from full_domain_lf_export import load_context, export_event

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("outputs/v2/full_domain/c2_f2/fold_1/seed_42")
EVENT_COUNT, FULL_CELL_COUNT, WET_CELL_COUNT, TIMESTEP_COUNT = 18, 780785, 769709, 383
METRICS = {
    "rmse_m": "RMSE", "fidelity_index": "FI", "maximum_depth_r2": "MaxWD_R2",
    "average_peak_difference_m": "peak_diff", "maximum_extent_csi": "CSI",
}
REQUIRED_ARRAYS = {
    "fine_global_ids", "terrain", "target_wse", "target_depth",
    "raw_predicted_wse", "predicted_wse", "predicted_depth",
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def wet_positions(ids, wet_ids):
    """Map global wet-cell IDs into prediction columns, preserving author order."""
    ids, wet_ids = np.asarray(ids), np.asarray(wet_ids)
    if len(np.unique(ids)) != len(ids) or len(np.unique(wet_ids)) != len(wet_ids):
        raise ValueError("Duplicate cell IDs")
    order = np.argsort(ids)
    positions = np.searchsorted(ids[order], wet_ids)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[order[positions]], wet_ids):
        raise ValueError("Predictions do not cover every LSG wet-mask cell")
    return order[positions]


def classification(reference_depth, predicted_depth, threshold=0.03):
    """Same >= threshold convention as the adopted CSI evaluator; 0 TN, 1 TP, 2 FP, 3 FN."""
    reference = reference_depth >= threshold
    predicted = predicted_depth >= threshold
    codes = np.zeros(reference.shape, dtype=np.uint8)
    codes[reference & predicted] = 1
    codes[~reference & predicted] = 2
    codes[reference & ~predicted] = 3
    return codes


def peak_maps(reference_depth, predicted_depth, terrain):
    reference = reference_depth.max(axis=0)
    predicted = predicted_depth.max(axis=0)
    return {
        "reference_peak_depth_m": reference,
        "predicted_peak_depth_m": predicted,
        "reference_peak_wse_m": reference + terrain,
        "predicted_peak_wse_m": predicted + terrain,
        # Matches AvgPeakDiff: positive means underprediction.
        "peak_depth_difference_m": reference - predicted,
        "maximum_extent_class": classification(reference, predicted),
    }


def verify_deployment(root):
    from gnnsr.release import verify_source
    verify_source(root)


def compare(root=ROOT):
    directory = root / RUN
    status = json.loads((directory / "training_status.json").read_text())
    if status["state"] != "complete":
        raise ValueError("Complete the report training run before common-mask evaluation")
    marker_path = directory / "full_domain_evaluation.json"
    marker = json.loads(marker_path.read_text())
    if marker["checkpoint_sha256"] != sha256(directory / "best.pt"):
        raise ValueError("Saved evaluation is not for the current best.pt")
    with (directory / "history.csv").open(newline="") as handle:
        selected = min(csv.DictReader(handle), key=lambda row: float(row["validation_depth_rmse_m"]))
    if int(selected["epoch"]) != marker["checkpoint_epoch"]:
        raise ValueError("Evaluation is not from the validation-RMSE-selected epoch")
    with (directory / "test_event_metrics.csv").open(newline="") as handle:
        timing_rows = list(csv.DictReader(handle))
    dataset = root / "reference_files/BurnettRV"
    events_path = dataset / "BurnettRV_event_summary.csv"
    with events_path.open(newline="", encoding="utf-8-sig") as handle:
        events = [(i, row) for i, row in enumerate(csv.DictReader(handle)) if int(row["Group"]) == 1]
    names = [row["tuflow_evt_name_old"] for _, row in events]
    if len(names) != EVENT_COUNT or len(timing_rows) != EVENT_COUNT or {r["event_name"] for r in timing_rows} != set(names):
        raise ValueError("Expected all 18 fold-1 events")
    for name in ("test_event_metrics.csv", "test_summary.json"):
        if marker["artifacts"].get(name) != sha256(directory / name):
            raise ValueError(f"Evaluation artifact differs: {name}")
    mask_path = dataset / "HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_1.npz"
    results_path = dataset / "Result_data/Validation_results.npz"
    geometry_path = dataset / "Geometry_data/Tuflow_Geometry_data.npz"
    with np.load(mask_path) as archive:
        wet_ids = archive["wet_idx"].astype(np.int64)
    geometry = GeometryData.from_npz(geometry_path)
    if len(wet_ids) != WET_CELL_COUNT or len(geometry.xy) != FULL_CELL_COUNT:
        raise ValueError("Unexpected Burnett fold-1 domain or scoring mask")
    with np.load(results_path) as archive:
        lsg = {key: archive[value][:, 0] for key, value in METRICS.items()}
        lsg_steps = archive["n_timesteps"].reshape(-1)
    output = directory / "lsg_wet_mask_comparison"
    output.mkdir(exist_ok=True)
    completion = output / "completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["evaluation_marker_sha256"] != sha256(marker_path):
            raise ValueError("Comparison exists for a different evaluation")
        for name, expected in saved["source_sha256"].items():
            if sha256(root / name) != expected:
                raise ValueError(f"Comparison source changed: {name}")
        for name, expected in saved["prediction_sha256"].items():
            if sha256(directory / name) != expected:
                raise ValueError(f"Prediction changed: {name}")
        for name, expected in saved["artifacts"].items():
            if sha256(output / name) != expected:
                raise ValueError(f"Completed comparison changed: {name}")
        print(f"Verified existing comparison: {output}")
        return output
    lock = output / ".running"
    with lock.open("x") as handle:
        handle.write("Remove only after confirming no comparison job is running.\n")
    try:
        rows, input_hashes, time_hashes, lf_hashes = [], {}, {}, {}
        lf_context = load_context(root)
        common_ids = None
        for index, event in events:
            name = event["tuflow_evt_name_old"]
            prediction = directory / "predictions" / f"{name}.npz"
            digest = sha256(prediction)
            if marker["artifacts"].get(f"predictions/{name}.npz") != digest:
                raise ValueError(f"Prediction provenance differs: {name}")
            input_hashes[f"predictions/{name}.npz"] = digest
            with np.load(prediction) as archive:
                if not REQUIRED_ARRAYS.issubset(archive.files):
                    raise ValueError(f"WSE/depth mapping arrays missing: {name}")
                ids, terrain = archive["fine_global_ids"], archive["terrain"]
                if len(ids) != len(geometry.xy) or not np.array_equal(np.sort(ids), np.arange(len(ids))):
                    raise ValueError("Predictions must cover the complete domain")
                if not np.array_equal(terrain, geometry.elevation[ids]):
                    raise ValueError("Prediction terrain differs from mapping geometry")
                if common_ids is None:
                    common_ids = ids
                    positions = wet_positions(ids, wet_ids)
                    np.savez_compressed(output / "mapping_geometry.npz", fine_global_ids=ids,
                                        xy=geometry.xy[ids], terrain=terrain,
                                        scoring_mask=np.isin(ids, wet_ids),
                                        lsg_wet_global_ids=wet_ids, crs=np.array("EPSG:28356"))
                elif not np.array_equal(ids, common_ids):
                    raise ValueError("Prediction column order changes between events")
                reference, predicted = archive["target_depth"], archive["predicted_depth"]
            if reference.shape != (TIMESTEP_COUNT, len(ids)) or predicted.shape != reference.shape:
                raise ValueError(f"Unexpected prediction dimensions: {name}")
            if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
                raise ValueError(f"Non-finite depths: {name}")
            if int(lsg_steps[index]) != len(reference):
                raise ValueError("LSG/GNN timestep counts differ")
            time_path = dataset / "HD_model_data/High-fidelity" / f"Paradise_{name}_002.npz"
            with np.load(time_path) as archive:
                times = archive["time_data"][48:]
            if len(times) != len(reference):
                raise ValueError("Time coordinate count differs")
            if times.dtype.hasobject:
                times = times.astype(str)
            time_hashes[name] = hashlib.sha256(times.tobytes()).hexdigest()
            maps = peak_maps(reference, predicted, terrain)
            np.savez_compressed(output / f"{name}_peak_maps.npz", **maps, time=times,
                                source_timestep_index=np.arange(48, 48 + len(times)))
            metrics = calculate_event_metrics(reference[:, positions], predicted[:, positions]).as_dict()
            counts = np.bincount(maps["maximum_extent_class"][positions], minlength=4)
            timing = next(row for row in timing_rows if row["event_name"] == name)
            common = {"event_index": index, "event_name": name, "group": 1,
                      "n_timesteps": len(times), "n_target_cells": len(wet_ids)}
            rows.append({**common, "model": "GNN-SR", **metrics,
                         "TN": int(counts[0]), "TP": int(counts[1]), "FP": int(counts[2]), "FN": int(counts[3]),
                         "model_forward_seconds": timing["model_forward_seconds"],
                         "end_to_end_seconds": timing["end_to_end_seconds"]})
            rows.append({**common, "model": "LSG", **{key: float(value[index]) for key, value in lsg.items()},
                         "TN": "", "TP": "", "FP": "", "FN": "",
                         "model_forward_seconds": "", "end_to_end_seconds": ""})
            del predicted, maps
            lf_depth, lf_source = export_event(lf_context, event, ids, terrain, times,
                                               output / "lf_predictions" / f"{name}.npz")
            lf_hashes[lf_source.relative_to(root).as_posix()] = sha256(lf_source)
            lf_maps = peak_maps(reference, lf_depth, terrain)
            np.savez_compressed(output / f"{name}_lf_peak_maps.npz", **lf_maps, time=times,
                                source_timestep_index=np.arange(48, 48 + len(times)))
            lf_metrics = calculate_event_metrics(reference[:, positions], lf_depth[:, positions]).as_dict()
            counts = np.bincount(lf_maps["maximum_extent_class"][positions], minlength=4)
            rows.append({**common, "model": "LF", **lf_metrics,
                         "TN": int(counts[0]), "TP": int(counts[1]), "FP": int(counts[2]), "FN": int(counts[3]),
                         "model_forward_seconds": "", "end_to_end_seconds": ""})
            del reference, lf_depth, lf_maps
            print(f"Scored and exported maps: {name}", flush=True)
        write_csv(output / "event_metrics.csv", rows)
        summary = []
        for model in ("GNN-SR", "LSG", "LF"):
            selected_rows = [row for row in rows if row["model"] == model]
            summary.append({"model": model, "events": len(selected_rows), **{
                key: float(np.nanmean([row[key] for row in selected_rows])) for key in METRICS}})
        write_csv(output / "mean_event_metrics.csv", summary)
        write_json(output / "mapping_metadata.json", {
            "crs": "EPSG:28356", "cell_size_m": 20, "dry_threshold_m": 0.03,
            "class_codes": {"0": "TN", "1": "TP", "2": "FP", "3": "FN"},
            "classification": "depth >= 0.03 m, matching adopted CSI; peak classes use each cell's temporal maximum",
            "difference_sign": "reference minus predicted; positive = underprediction",
            "peak_difference": "difference of independent temporal maxima, not maximum instantaneous error",
            "mapping_support": "All 780785 cells; exclude scoring_mask=False from benchmark metrics",
            "timestep_mapping": "Prediction row k corresponds to time[k] and original source timestep k+48",
            "wse_retention": "Keep ../predictions/*.npz: raw_predicted_wse, predicted_wse, target_wse, predicted_depth, target_depth, terrain, fine_global_ids",
            "lf_retention": "lf_predictions/*.npz: lf_baseline_wse (exact parent WSE used by GNN), lf_wse and lf_depth (physical outputs on HF terrain), time_hours and cell IDs",
            "lf_mapping": "Existing training graph containing-parent mapping including its original nearest-parent fallbacks; HecEvent rounding and warm-up unchanged",
            "hydrographs": "For an event and fine_global_id, select the same column from each timestep array; use time_hours for LF, GNN and HF. Metrics CSV alone is insufficient.",
            "timing": "GNN original full-domain inference timings; rescoring does not remeasure inference; no comparable LSG timing exported",
            "lsg_source": "Authors' Validation_results.npz column 0, Group 1 rows; no new LSG inference",
            "metric_labels": {"rmse_m": "AvgRMSE", "fidelity_index": "FI", "maximum_depth_r2": "R2",
                              "average_peak_difference_m": "AvgPeakDiff", "maximum_extent_csi": "CSI"},
            "metric_implementation": "Unchanged gnnsr.metrics adopted functions, including pooled RMSE and original FI convention",
        })
        write_json(completion, {
            "state": "complete", "fold": 1, "seed": 42, "events": EVENT_COUNT,
            "checkpoint_epoch": marker["checkpoint_epoch"], "checkpoint_sha256": marker["checkpoint_sha256"],
            "evaluation_marker_sha256": sha256(marker_path), "prediction_sha256": input_hashes,
            "time_array_sha256": time_hashes,
            "source_sha256": {**lf_hashes, **{str(p.relative_to(root)): sha256(p) for p in
                              (mask_path, results_path, geometry_path, events_path, root / "src/gnnsr/metrics.py",
                               root / "scripts/compare_full_domain_lsg.py", root / "scripts/full_domain_lf_export.py",
                               *lf_context["source_paths"])}},
            "artifacts": {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*")
                          if p.suffix in (".csv", ".json", ".npz")},
        })
    finally:
        lock.unlink()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-deployment", action="store_true")
    args = parser.parse_args()
    if args.verify_deployment:
        verify_deployment(ROOT)
    print(compare())
