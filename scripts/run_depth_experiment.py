"""Train or evaluate one AOI depth trial; shared scientific implementation is unchanged."""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
from plan_depth_experiments import EXPERIMENT, ROOT, load_plan
from run_width_experiment import provenance, sha256, write_json

from gnnsr.config import ExperimentConfig
from gnnsr.graph import prepare_aoi_graph
from gnnsr.train import evaluate_model, experiment_directory, train_model


def candidate_config(name, preflight=False):
    if name not in {p["name"] for p in load_plan(stage="initial")}:
        raise ValueError(f"Unknown candidate: {name}")
    config = ExperimentConfig.from_toml(ROOT / EXPERIMENT / "configs" / f"{name}.toml")
    if preflight:
        config = replace(
            config,
            paths=replace(config.paths, output_dir=config.paths.output_dir / "preflight"),
            training=replace(
                config.training,
                max_epochs=1,
                extend_epoch_cap_if_improving=False,
                debug_max_train_samples=2,
                debug_max_validation_samples=1,
                debug_max_test_samples=1,
            ),
        )
    return config


def depth_provenance(config):
    record = provenance(config)
    graph, _ = prepare_aoi_graph(config)
    target_ids = graph.fine_global_ids[graph.target_mask]
    if len(target_ids) != 2314:
        raise ValueError("The approved AOI must retain its 2314 target cells")
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
    record["plan_helper_sha256"] = sha256(ROOT / "scripts/plan_depth_experiments.py")
    return record


def check_existing(directory, expected, resume):
    manifest = directory / "depth_experiment.json"
    if manifest.exists():
        if not resume:
            raise RuntimeError("Run exists; use --resume only after checking its job state")
        if json.loads(manifest.read_text()) != expected:
            raise ValueError("Run provenance changed; refusing to mix results")
    elif any(directory.glob("*.pt")) or (directory / "history.csv").exists():
        raise RuntimeError("Existing results lack depth provenance")


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
    marker = directory / "depth_evaluation.json"
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
        int(r["n_timesteps"]) != 383 or int(r["n_target_cells"]) != 2314 for r in events
    ):
        raise ValueError("Held-out temporal support or AOI target count changed")
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


def run(name, mode, resume=False, segment_hours=43):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    config = candidate_config(name, preflight=mode == "preflight")
    directory = experiment_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".depth_job.lock"
    with lock.open("x") as handle:
        json.dump({"pbs_job_id": os.environ.get("PBS_JOBID"), "pid": os.getpid()}, handle)
    pipeline = directory / "pipeline_status.json"
    writable = False
    try:
        metadata = depth_provenance(config)
        check_existing(directory, metadata, resume)
        writable = True
        write_json(directory / "depth_experiment.json", metadata)
        status_file = directory / "training_status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if mode != "evaluate" and status.get("state") != "complete":
            write_json(pipeline, {"state": "training", "candidate": name})
            train_model(config, resume=resume, segment_seconds=segment_hours * 3600)
            status = json.loads(status_file.read_text())
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
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-hours", type=float, default=43)
    args = parser.parse_args()
    if args.segment_hours <= 0:
        parser.error("segment-hours must be positive")
    run(args.candidate, args.mode, args.resume, args.segment_hours)
