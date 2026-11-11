"""Reproduce the report: full-domain training up to 100 epochs."""

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import torch
from run_width_experiment import provenance, sha256, write_json

from gnnsr.config import ExperimentConfig
from gnnsr.graph import prepare_aoi_graph
from gnnsr.train import evaluate_model, experiment_directory, train_model

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("experiments/v2/full_domain")
TARGET_CELLS = 780785


def candidate_config(name="c2_f2", preflight=False, root=ROOT):
    if name != "c2_f2":
        raise ValueError(f"Unknown full-domain candidate: {name}")
    config = ExperimentConfig.from_toml(root / EXPERIMENT / "configs/c2_f2.toml")
    reference = ExperimentConfig.from_toml(
        root / "experiments/v2/message_passing/configs/c2_f2.toml"
    )
    for section in ("data", "graph", "model", "loss"):
        if getattr(config, section) != getattr(reference, section):
            raise ValueError(f"Changed approved c2/f2 settings: {section}")
    expected_training = replace(reference.training, max_epochs=100, extend_epoch_cap_if_improving=False)
    if config.training != expected_training:
        raise ValueError("Full-domain settings must match AOI except the fixed 100-epoch cap")
    if config.paths.aoi_path != root / "examples/burnettrv_full_domain.geojson":
        raise ValueError("Expected the full-domain polygon")
    if preflight:
        config = replace(
            config,
            paths=replace(config.paths, output_dir=config.paths.output_dir / "preflight"),
            training=replace(
                config.training,
                max_epochs=1,
                extend_epoch_cap_if_improving=False,
                debug_max_train_samples=20,
                debug_max_validation_samples=5,
                debug_max_test_samples=1,
            ),
        )
    return config


def full_provenance(config):
    record = provenance(config)
    graph, _ = prepare_aoi_graph(config)
    target_ids = graph.fine_global_ids[graph.target_mask]
    if len(target_ids) != TARGET_CELLS or not graph.target_mask.all():
        raise ValueError("Full-domain training must include all 780785 TUFLOW cells")
    record["graph_support"] = {
        "coarse_nodes": len(graph.coarse_global_ids),
        "fine_nodes": len(graph.fine_global_ids),
        "coarse_edges": graph.coarse_edge_index.shape[1],
        "fine_edges": graph.fine_edge_index.shape[1],
        "target_cells": len(target_ids),
        "target_ids_sha256": hashlib.sha256(target_ids.astype("<i8").tobytes()).hexdigest(),
    }
    record["loss"] = config.as_dict()["loss"]
    record["runner_sha256"] = sha256(Path(__file__))
    record["provenance_helper_sha256"] = sha256(ROOT / "scripts/run_width_experiment.py")
    record["save_epoch_checkpoints"] = True
    return record


def check_existing(directory, expected, resume):
    manifest = directory / "full_domain_experiment.json"
    if manifest.exists():
        if not resume:
            raise RuntimeError("Run exists; use --resume only after checking its job state")
        if json.loads(manifest.read_text()) != expected:
            raise ValueError("Run provenance changed; refusing to mix results")
    elif any(directory.glob("*.pt")) or (directory / "history.csv").exists():
        raise RuntimeError("Existing results lack full-domain provenance")


def evaluate_selected(config, directory):
    """Evaluate only a completed run's validation-selected checkpoint; verify saved outputs."""
    status = json.loads((directory / "training_status.json").read_text())
    if status["state"] != "complete":
        raise RuntimeError("Training must complete before held-out evaluation")
    best_path = directory / "best.pt"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    with (directory / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    selected = min(history, key=lambda r: float(r["validation_depth_rmse_m"]))
    if checkpoint["epoch"] != int(selected["epoch"]):
        raise ValueError("best.pt is not the minimum validation-RMSE checkpoint")
    if checkpoint["validation_depth_rmse_m"] != float(selected["validation_depth_rmse_m"]):
        raise ValueError("Checkpoint RMSE differs from training history")
    norm_name = checkpoint["normalization_path"].replace("\\", "/").split("/")[-1]
    fixed = {
        "checkpoint_sha256": sha256(best_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "normalization_sha256": sha256(directory / norm_name),
    }
    marker = directory / "full_domain_evaluation.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if any(saved[k] != v for k, v in fixed.items()):
            raise ValueError("Evaluation marker belongs to a different checkpoint/normalization")
        artifacts = saved.get("artifacts", {})
        if artifacts and all(
            (directory / p).is_file() and sha256(directory / p) == h for p, h in artifacts.items()
        ):
            return saved
    metrics = evaluate_model(config, best_path, save_predictions=True)
    with metrics.open(newline="") as handle:
        events = list(csv.DictReader(handle))
    expected_events = 1 if config.training.debug_max_test_samples else 18
    if len(events) != expected_events or len({r["event_name"] for r in events}) != expected_events:
        raise ValueError("Unexpected held-out event count")
    if any(int(r["group"]) != config.training.fold for r in events):
        raise ValueError("Unexpected held-out group")
    if not config.training.debug_max_test_samples and any(
        int(r["n_timesteps"]) != 383 or int(r["n_target_cells"]) != TARGET_CELLS for r in events
    ):
        raise ValueError("Held-out temporal support or full-domain target count changed")
    summary_path = directory / "test_summary.json"
    if json.loads(summary_path.read_text())["events"] != expected_events:
        raise ValueError("Summary event count differs")
    artifacts = [metrics, summary_path] + [
        directory / "predictions" / f"{r['event_name']}.npz" for r in events
    ]
    saved = {
        **fixed,
        "metrics_sha256": sha256(metrics),
        "events": expected_events,
        "artifacts": {p.relative_to(directory).as_posix(): sha256(p) for p in artifacts},
    }
    write_json(marker, saved)
    return saved


def verify_epoch_archive(directory, epoch):
    records = {}
    for number in range(1, epoch + 1):
        path = directory / "checkpoints" / f"epoch_{number:05d}.pt"
        records[path.relative_to(directory).as_posix()] = sha256(path)
    latest = directory / "latest.pt"
    if records.get(f"checkpoints/epoch_{epoch:05d}.pt") != sha256(latest):
        raise ValueError("Latest checkpoint does not match the final epoch archive")
    write_json(
        directory / "checkpoint_archive.json",
        {
            "epochs": epoch,
            "format": "Complete resumable training checkpoints",
            "artifacts": records,
        },
    )


def write_preflight_report(config, directory):
    from gnnsr.data import load_fold_samples
    from gnnsr.io import EventCatalog

    with (directory / "history.csv").open(newline="") as handle:
        row = list(csv.DictReader(handle))[-1]
    full = candidate_config()
    splits = load_fold_samples(full, EventCatalog(full.paths.dataset_root))
    estimate = float(row["mean_training_step_seconds"]) * len(splits.train) + float(
        row["mean_validation_step_seconds"]
    ) * len(splits.validation)
    report = {
        "target_cells": TARGET_CELLS,
        "train_steps_measured": int(row["training_steps"]),
        "validation_steps_measured": int(row["validation_steps"]),
        "full_training_steps": len(splits.train),
        "full_validation_steps": len(splits.validation),
        "estimated_full_epoch_hours": estimate / 3600,
        "gpu_peak_memory_allocated_gib": float(row["gpu_peak_memory_allocated_bytes"]) / 2**30,
        "gpu_peak_memory_reserved_gib": float(row["gpu_peak_memory_reserved_bytes"]) / 2**30,
        "checkpoint_bytes": (directory / "latest.pt").stat().st_size,
        "note": "Subset timing estimate only; full training refits normalization on all training timesteps.",
    }
    write_json(directory / "preflight_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run(name, mode, resume=False, segment_seconds=6 * 3600):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    config = candidate_config(name, preflight=mode == "preflight")
    if mode == "train":
        preflight = experiment_directory(candidate_config(name, preflight=True))
        marker = preflight / "pipeline_status.json"
        if not marker.exists() or json.loads(marker.read_text())["state"] != "preflight_complete":
            raise RuntimeError("Successful full-domain preflight required before training")
    directory = experiment_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".full_domain_job.lock"
    with lock.open("x") as handle:
        json.dump({"pbs_job_id": os.environ.get("PBS_JOBID"), "pid": os.getpid()}, handle)
    pipeline = directory / "pipeline_status.json"
    writable = False
    try:
        metadata = full_provenance(config)
        check_existing(directory, metadata, resume)
        writable = True
        write_json(directory / "full_domain_experiment.json", metadata)
        status_file = directory / "training_status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if mode != "evaluate" and status.get("state") != "complete":
            write_json(pipeline, {"state": "training", "candidate": name})
            train_model(
                config, resume=resume, segment_seconds=segment_seconds, save_epoch_checkpoints=True
            )
            status = json.loads(status_file.read_text())
        verify_epoch_archive(directory, status["epoch"])
        if not math.isfinite(status["best_validation_depth_rmse_m"]):
            raise ValueError("Nonfinite validation RMSE")
        if status.get("state") != "complete":
            if mode == "evaluate" or status.get("reason") != "segment_limit":
                raise RuntimeError(f"Unexpected incomplete training state: {status}")
            state = "segment_complete"
        elif mode == "train":
            state = "training_complete"
        else:
            write_json(pipeline, {"state": "evaluating", "candidate": name})
            evaluate_selected(config, directory)
            state = "preflight_complete" if mode == "preflight" else "complete"
            if mode == "preflight":
                write_preflight_report(config, directory)
        write_json(
            pipeline,
            {
                "state": state,
                "candidate": name,
                "training_epoch": status.get("epoch"),
                "best_validation_depth_rmse_m": status.get("best_validation_depth_rmse_m"),
            },
        )
        print(f"{name}: {state}; {directory}")
        return directory
    except Exception as error:
        # Preserve existing runs on provenance refusal; failures remain visible in the PBS log.
        if writable:
            write_json(
                pipeline, {"state": "failed", "candidate": name, "phase": mode, "error": str(error)}
            )
        print(f"{name}: failed: {error}")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="c2_f2", choices=("c2_f2",))
    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-seconds", type=float, default=6 * 3600)
    args = parser.parse_args()
    if args.segment_seconds <= 0:
        parser.error("segment-seconds must be positive")
    run(args.candidate, args.mode, args.resume, args.segment_seconds)
