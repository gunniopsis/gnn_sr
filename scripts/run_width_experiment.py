"""Run one MSE-only width experiment, or its separate one-epoch GPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import replace
from pathlib import Path

import torch

from gnnsr.config import ExperimentConfig
from gnnsr.data import load_fold_samples
from gnnsr.io import EventCatalog
from gnnsr.model import TwoScaleFloodGNN
from gnnsr.train import evaluate_model, experiment_directory, train_model

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (64, 128, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def config_for_width(width: int, preflight: bool = False) -> ExperimentConfig:
    if width not in WIDTHS:
        raise ValueError(f"Unsupported width: {width}")
    config = ExperimentConfig.from_toml(
        ROOT / "experiments/v2/hidden_width/configs" / f"width_{width}.toml"
    )
    if config.model.hidden_dim != width:
        raise ValueError("Width does not match the configuration")
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


def comparison_settings(config: dict) -> dict:
    """Settings that must agree across widths; locations are audited by content hashes."""
    model = {k: v for k, v in config["model"].items() if k != "hidden_dim"}
    return {
        "data": config["data"],
        "graph": config["graph"],
        "model": model,
        "training": config["training"],
    }


def provenance(config: ExperimentConfig) -> dict:
    splits = load_fold_samples(config, EventCatalog(config.paths.dataset_root))
    split_records = {}
    for name in ("train", "validation", "test"):
        refs = getattr(splits, name)
        content = "\n".join(f"{ref.event_index},{ref.timestep}" for ref in refs).encode()
        split_records[name] = {
            "samples": len(refs),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    split_file = (
        config.paths.dataset_root
        / "Train_test_split_data"
        / (f"Train_test_split_ValidateOnGrp_{config.training.fold}.npz")
    )
    return {
        "schema": 1,
        "hidden_dim": config.model.hidden_dim,
        "trainable_parameters": sum(
            p.numel() for p in TwoScaleFloodGNN(config.model).parameters() if p.requires_grad
        ),
        "settings": comparison_settings(config.as_dict()),
        "aoi_sha256": sha256(config.paths.aoi_path),
        "split_file_sha256": sha256(split_file),
        "splits": split_records,
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted((ROOT / "src/gnnsr").glob("*.py"))
        },
        "runner_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def verify_existing_run(run_dir: Path, expected: dict, resume: bool) -> None:
    manifest = run_dir / "width_experiment.json"
    if manifest.exists():
        if not resume:
            raise RuntimeError("Run already exists; use --resume to continue or evaluate it")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError("Run provenance changed; refusing to mix configurations or source")
    elif any(run_dir.glob("*.pt")) or (run_dir / "history.csv").exists():
        raise RuntimeError("Existing run has no width-experiment provenance; refusing to overwrite")


def run(width: int, preflight: bool, resume: bool, segment_hours: float) -> Path:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This experiment requires exactly one visible CUDA GPU")
    config = config_for_width(width, preflight)
    run_dir = experiment_directory(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".width_job.lock"
    # Exclusive creation also protects identical widths submitted concurrently.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pbs_job_id": os.environ.get("PBS_JOBID"), "pid": os.getpid()}))
    try:
        metadata = provenance(config)
        verify_existing_run(run_dir, metadata, resume)
        write_json(run_dir / "width_experiment.json", metadata)
        print(json.dumps(metadata, indent=2), flush=True)
        status_path = run_dir / "training_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        best = run_dir / "best.pt"
        if status.get("state") != "complete":
            best = train_model(config, resume=resume, segment_seconds=segment_hours * 3600)
            status = json.loads(status_path.read_text())
        if status["state"] != "complete":
            print("Segment saved. Resubmit this width with GNNSR_RESUME=1 to continue.")
            return run_dir
        if preflight:
            print(f"GPU preflight complete: {run_dir}")
            return run_dir
        if not best.is_file():
            raise FileNotFoundError(best)
        evaluation_path = run_dir / "width_evaluation.json"
        checkpoint_hash = sha256(best)
        if evaluation_path.exists():
            saved = json.loads(evaluation_path.read_text())
            metrics_path = run_dir / "test_event_metrics.csv"
            if (
                saved["checkpoint_sha256"] == checkpoint_hash
                and metrics_path.is_file()
                and sha256(metrics_path) == saved["metrics_sha256"]
            ):
                print(f"Training and evaluation already complete: {run_dir}")
                return run_dir
        metrics_path = evaluate_model(config, best, save_predictions=True)
        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        normalization = run_dir / Path(checkpoint["normalization_path"]).name
        write_json(
            evaluation_path,
            {
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_epoch": checkpoint["epoch"],
                "normalization_sha256": sha256(normalization),
                "metrics_sha256": sha256(metrics_path),
            },
        )
        print(f"Training and held-out evaluation complete: {run_dir}")
        return run_dir
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=WIDTHS, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-hours", type=float, default=43)
    args = parser.parse_args()
    if args.segment_hours <= 0:
        parser.error("--segment-hours must be positive")
    run(args.width, args.preflight, args.resume, args.segment_hours)


if __name__ == "__main__":
    main()
