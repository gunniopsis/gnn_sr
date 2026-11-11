"""Original LSG inference, followed by the deployed GNN-SR scoring functions."""

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.neighbors import BallTree

from gnnsr.data import physical_wse_and_depth
from gnnsr.geometry import GeometryData
from gnnsr.io import EventRecord, HecEvent, TuflowEvent, _decode_hec_times
from gnnsr.lsg import EOFData, SavedSparseGP, _reconstruct_subset
from gnnsr.metrics import calculate_event_metrics

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path("experiments/v2/lsg_full_domain")
RELEASE = Path(".")
OUTPUT = Path("outputs/v2/lsg_full_domain/burnett/fold_1/seed_42")
METRICS = (
    "rmse_m",
    "fidelity_index",
    "maximum_depth_r2",
    "average_peak_difference_m",
    "maximum_extent_csi",
)
EXPECTED_STEPS, EXPECTED_CELLS = 383, 780785


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temp.replace(path)


def write_csv(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def events(root):
    dataset = root / "reference_files/BurnettRV"
    with (dataset / "BurnettRV_event_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for i, row in enumerate(rows):
        if int(row["Group"]) != 1:
            continue
        name, plan = row["tuflow_evt_name_old"], row["HEC_RAS_plan"]
        result.append(
            EventRecord(
                i,
                name,
                1,
                plan,
                dataset / "HD_model_data/High-fidelity" / f"Paradise_{name}_002.npz",
                dataset / "HD_model_data/Low-fidelity" / f"BurnettRV_LFmodelB.{plan}.hdf",
            )
        )
    if len(result) != 18:
        raise ValueError("Expected the same 18 Burnett fold-1 held-out events")
    return result


class Model:
    def __init__(self, dataset, load_gp=True):
        analysis = dataset / "HF_EOF_analysis"
        with np.load(analysis / "Categories_HFdata_ValidateOnGrp_1.npz") as z:
            self.wet, self.tf, self.af = (
                z[k].astype(np.int64) for k in ("wet_idx", "TF_idx", "AF_idx")
            )
        self.wse = EOFData.load(analysis / "EOFanalysis_HFdata_WSE_ValidateOnGrp_1.npz")
        self.ext = EOFData.load(analysis / "EOFanalysis_HFdata_EXT_ValidateOnGrp_1.npz")
        if load_gp:
            self.wse_gp = SavedSparseGP(dataset / "LSG_modeldata/LSG_WSE_ValidateOnGrp_1.npz")
            self.ext_gp = SavedSparseGP(dataset / "LSG_modeldata/LSG_Ext_ValidateOnGrp_1.npz")

    def project(self, projected, terrain):
        # Apply HF terrain filter BEFORE making extent coefficients, as Evaluation.py does.
        projected = np.where(projected > terrain + 0.03, projected, terrain)
        binary = (projected[:, self.tf] >= terrain[self.tf] + 0.03).astype(np.float64)
        # Project all saved modes; saved x_scalers are applied before GP input truncation.
        wc = ((projected[:, self.wet] - self.wse.mean) * self.wse.weights) @ self.wse.modes.T
        ec = ((binary - self.ext.mean) * self.ext.weights) @ self.ext.modes.T
        return wc, ec

    def reconstruct(self, wc, ec, terrain):
        raw = np.broadcast_to(terrain, (len(wc), len(terrain))).astype(np.float64).copy()
        raw[:, self.wet] = _reconstruct_subset(wc, self.wse, np.arange(len(self.wet)))
        wet = np.zeros(raw.shape, bool)
        wet[:, self.af] = True
        wet[:, self.tf] = _reconstruct_subset(ec, self.ext, np.arange(len(self.tf))) > 0.5
        # The two heads jointly determine LSG output. Do not replace this with GNN inference.
        wse = np.where(wet & (raw > terrain + 0.03), raw, terrain)
        return raw, wet, wse


def classify(reference, predicted):
    r, p = reference >= 0.03, predicted >= 0.03
    return np.where(r, np.where(p, 1, 3), np.where(p, 2, 0)).astype(np.uint8)


def score(reference, predicted, wet_ids):
    """Exactly the same function, array dtype and author cell order as the GNN comparison."""
    rows = []
    for mask, ids in (("authors_fold_wet", wet_ids), ("full_domain", slice(None))):
        y, p = reference[:, ids], predicted[:, ids]
        metrics = calculate_event_metrics(y, p).as_dict()
        counts = np.bincount(classify(y.max(axis=0), p.max(axis=0)), minlength=4)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite evaluation metric")
        rows.append(
            dict(
                mask=mask,
                n_target_cells=y.shape[1],
                **metrics,
                TN=int(counts[0]),
                TP=int(counts[1]),
                FP=int(counts[2]),
                FN=int(counts[3]),
            )
        )
    return rows


def evaluate_event(
    event, model, coarse, fine, author_terrain, nearest, directory, identity, chunk=8
):
    marker = directory / "events" / f"{event.name}.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity:
            raise ValueError("Event identity changed")
        for relative, digest in saved["artifacts"].items():
            if sha256(directory / relative) != digest:
                raise ValueError(f"Saved event artifact changed: {relative}")
        return saved["rows"]
    begin = time.perf_counter()
    hf = TuflowEvent(event.tuflow_path, 48)
    if hf.n_timesteps != EXPECTED_STEPS or hf.n_cells != EXPECTED_CELLS:
        raise ValueError("Expected 383 timesteps and 780785 HF cells")
    all_ids = np.arange(len(fine.xy))
    with HecEvent(event.hec_path, "BurnettRV_region", np.arange(len(coarse.xy)), 47.5) as lf:
        dates = _decode_hec_times(
            lf._handle[
                "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp"
            ][:]
        )
        hours = (dates - dates[0]) / np.timedelta64(1, "h")
        if lf.n_timesteps < hf.n_timesteps or not np.allclose(
            hours[lf._kept_timesteps[: hf.n_timesteps]], hf.time, rtol=0, atol=1e-8
        ):
            raise ValueError("LF/HF elapsed event times do not match the GNN workflow")
        wcs, ecs = [], []
        tick = time.perf_counter()
        for start in range(0, hf.n_timesteps, chunk):
            raw = np.stack(
                [
                    np.nan_to_num(lf.read_wse(t), nan=-999.0)
                    for t in range(start, min(start + chunk, hf.n_timesteps))
                ]
            )
            filtered = np.where(raw > coarse.elevation + 0.03, raw, coarse.elevation)
            wc, ec = model.project(filtered[:, nearest], author_terrain)
            wcs.append(wc)
            ecs.append(ec)
        projection_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    wc = model.wse_gp.predict(np.vstack(wcs))
    ec = model.ext_gp.predict(np.vstack(ecs))
    gp_seconds = time.perf_counter() - tick
    final = directory / "predictions" / f"{event.name}.h5"
    partial = final.with_suffix(".partial.h5")
    reconstruction_seconds = 0.0
    with h5py.File(partial, "w") as h:
        h.attrs.update(
            identity=json.dumps(identity, sort_keys=True),
            event_name=event.name,
            dry_threshold_m=0.03,
            classification_codes="0=TN,1=TP,2=FP,3=FN",
            crs="EPSG:28356",
        )
        h["time_hours"] = hf.time
        h["source_timestep_index"] = np.arange(48, 48 + hf.n_timesteps)
        for name in (
            "target_wse",
            "target_depth",
            "raw_predicted_wse",
            "predicted_wse",
            "predicted_depth",
            "predicted_wet",
            "target_wet",
            "extent_class",
            "lsg_extent_head_wet",
        ):
            dtype = (
                "u1"
                if name in ("predicted_wet", "target_wet", "extent_class", "lsg_extent_head_wet")
                else "f4"
            )
            h.create_dataset(
                name,
                shape=(hf.n_timesteps, len(fine.xy)),
                dtype=dtype,
                chunks=(1, min(16384, len(fine.xy))),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
        for start in range(0, hf.n_timesteps, chunk):
            stop = min(start + chunk, hf.n_timesteps)
            sl = slice(start, stop)
            tick = time.perf_counter()
            raw, extent, wse = model.reconstruct(wc[sl], ec[sl], author_terrain)
            # Output arrays use the same float32 and depth projection as GNN predictions.
            wse, depth = physical_wse_and_depth(wse.astype(np.float32), fine.elevation, 0.03)
            reconstruction_seconds += time.perf_counter() - tick
            target = np.stack(
                [np.nan_to_num(hf.read_wse(t, all_ids), nan=-999.0) for t in range(start, stop)]
            )
            target = np.where(target > fine.elevation + 0.03, target, fine.elevation).astype(
                np.float32
            )
            target_depth = np.maximum(target - fine.elevation, 0.0)
            values = {
                "target_wse": target,
                "target_depth": target_depth,
                "raw_predicted_wse": raw,
                "predicted_wse": wse,
                "predicted_depth": depth,
                "predicted_wet": depth >= 0.03,
                "target_wet": target_depth >= 0.03,
                "extent_class": classify(target_depth, depth),
                "lsg_extent_head_wet": extent,
            }
            if not all(np.isfinite(a).all() for a in values.values()):
                raise ValueError("Nonfinite LSG/HF prediction")
            for name, value in values.items():
                h[name][sl] = value
        # Shared metadata lives once in geometry.npz, not repeated in every event.
        reference, prediction = h["target_depth"][:], h["predicted_depth"][:]
        tick = time.perf_counter()
        rows = score(reference, prediction, model.wet)
        metric_seconds = time.perf_counter() - tick
        peaks = h.create_group("peak_maps")
        yp, pp = reference.max(axis=0), prediction.max(axis=0)
        for key, value in {
            "reference_peak_depth_m": yp,
            "predicted_peak_depth_m": pp,
            "reference_peak_wse_m": yp + fine.elevation,
            "predicted_peak_wse_m": pp + fine.elevation,
            "peak_depth_difference_m": yp - pp,
            "maximum_extent_class": classify(yp, pp),
        }.items():
            peaks[key] = value
        h.attrs["metrics_json"] = json.dumps(rows)
    partial.replace(final)
    elapsed = time.perf_counter() - begin
    for row in rows:
        row.update(
            event_index=event.index,
            event_name=event.name,
            group=1,
            model="LSG",
            n_timesteps=hf.n_timesteps,
            model_forward_seconds=gp_seconds + reconstruction_seconds,
            gp_prediction_seconds=gp_seconds,
            reconstruction_seconds=reconstruction_seconds,
            lf_projection_seconds=projection_seconds,
            metric_seconds=metric_seconds,
            end_to_end_seconds=elapsed,
        )
    write_json(
        marker,
        {
            "identity": identity,
            "rows": rows,
            "artifacts": {final.relative_to(directory).as_posix(): sha256(final)},
        },
    )
    return rows


def verify(root, full_hash=False):
    source = json.loads((root / RELEASE / "SOURCE_MANIFEST.json").read_text())
    for row in source["files"]:
        if sha256(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Deployment changed: {row['path']}")
    inputs = json.loads((root / EXPERIMENT / "input_manifest.json").read_text())
    missing = [
        row["path"]
        for row in inputs["files"]
        if not (root / row["path"]).is_file() or (root / row["path"]).stat().st_size != row["bytes"]
    ]
    if missing:
        raise ValueError(
            "Missing/wrong-sized inputs (restore these relative paths):\n" + "\n".join(missing)
        )
    stats = {}
    for number, row in enumerate(inputs["files"], 1):
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Missing/wrong-sized input: {path}")
        if full_hash and sha256(path) != row["sha256"]:
            raise ValueError(f"Input changed: {path}")
        if full_hash:
            print(f"Verified {number}/{len(inputs['files'])}: {path.name}", flush=True)
        stats[row["path"]] = [row["bytes"], path.stat().st_mtime_ns]
    identity = {
        "input_manifest_sha256": sha256(root / EXPERIMENT / "input_manifest.json"),
        "stats": stats,
    }
    receipt = root / "cache/v2/lsg_full_domain/verified_inputs.json"
    if full_hash:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt, identity)
    elif not receipt.exists() or json.loads(receipt.read_text()) != identity:
        raise ValueError("Run verify_lsg_full_domain_inputs.py after uploading inputs")
    return {
        "source_manifest_sha256": sha256(root / RELEASE / "SOURCE_MANIFEST.json"),
        "input_manifest_sha256": identity["input_manifest_sha256"],
    }


def run(root=ROOT, resume=False):
    identity = verify(root)
    from importlib.metadata import version

    identity["package_versions"] = {
        name: version(name) for name in ("numpy", "gpflow", "tensorflow", "scikit-learn", "h5py")
    }
    directory = root / OUTPUT
    if directory.exists() and not resume:
        raise RuntimeError("Output exists; inspect then use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".running"
    with lock.open("x") as handle:
        handle.write("Remove only after verifying that no LSG evaluation job is running.\n")
    try:
        for name in ("predictions", "events"):
            (directory / name).mkdir(exist_ok=True)
        dataset = root / "reference_files/BurnettRV"
        tick = time.perf_counter()
        model = Model(dataset)
        fine = GeometryData.from_npz(dataset / "Geometry_data/Tuflow_Geometry_data.npz")
        coarse = GeometryData.from_npz(dataset / "Geometry_data/HECRAS_Geometry_data.npz")
        with np.load(dataset / "Geometry_data/Tuflow_Geometry_data.npz") as z:
            author_terrain = z["Z_coor"]
        if len(model.wet) != 769709 or len(fine.xy) != 780785:
            raise ValueError("Burnett mask/domain changed")
        nearest = BallTree(coarse.xy, leaf_size=30).query(fine.xy, k=1)[1][:, 0]
        import gpflow
        import tensorflow as tf

        write_json(
            directory / "runtime.json",
            {
                "python": platform.python_version(),
                "gpflow": gpflow.__version__,
                "tensorflow": tf.__version__,
                "hardware": platform.processor(),
                "device": "CPU",
                "setup_seconds": time.perf_counter() - tick,
                "timing_definition": "model_forward_seconds = GP predict_y plus EOF reconstruction/gating/physical depth; excludes LF projection, reading, scoring and export. Compare hardware and timing scope before claiming speedup.",
            },
        )
        np.savez_compressed(
            directory / "geometry.npz",
            fine_global_ids=np.arange(len(fine.xy)),
            xy=fine.xy,
            terrain=fine.elevation,
            area=fine.area,
            authors_wet_ids=model.wet,
            scoring_mask=np.isin(np.arange(len(fine.xy)), model.wet),
            lsg_nearest_lf_ids=nearest,
            crs="EPSG:28356",
        )
        rows = []
        for event in events(root):
            write_json(
                directory / "pipeline_status.json",
                {"state": "evaluating", "event": event.name, "completed_events": len(rows) // 2},
            )
            rows.extend(
                evaluate_event(
                    event, model, coarse, fine, author_terrain, nearest, directory, identity
                )
            )
            write_csv(directory / "event_metrics.csv", rows)
            print(f"Completed {event.name} ({len(rows) // 2}/18)", flush=True)
        summary = []
        for mask in ("authors_fold_wet", "full_domain"):
            selected = [r for r in rows if r["mask"] == mask]
            summary.append(
                dict(
                    model="LSG",
                    mask=mask,
                    events=len(selected),
                    **{k: float(np.mean([r[k] for r in selected])) for k in METRICS},
                )
            )
        write_csv(directory / "mean_event_metrics.csv", summary)
        artifacts = {
            p.relative_to(directory).as_posix(): sha256(p)
            for p in (
                directory / "event_metrics.csv",
                directory / "mean_event_metrics.csv",
                directory / "geometry.npz",
                directory / "runtime.json",
                *sorted((directory / "events").glob("*.json")),
            )
        }
        write_json(
            directory / "completion.json",
            {
                "state": "complete",
                "fold": 1,
                "events": 18,
                "identity": identity,
                "artifacts": artifacts,
            },
        )
        write_json(
            directory / "pipeline_status.json", {"state": "complete", "events": 18, "fold": 1}
        )
    finally:
        lock.unlink()
    return directory
