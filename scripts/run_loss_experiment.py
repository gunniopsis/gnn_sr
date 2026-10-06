"""Width-64 composite-loss calibration, preflight, training and explicit evaluation."""

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from run_width_experiment import ROOT, provenance, sha256, write_json

from gnnsr.config import ExperimentConfig
from gnnsr.data import EventDataAccess, NormalizationBundle, load_fold_samples
from gnnsr.geometry import load_burnett_geometries
from gnnsr.graph import prepare_aoi_graph, to_heterodata
from gnnsr.io import EventCatalog
from gnnsr.model import TwoScaleFloodGNN
from gnnsr.train import (
    _normalized_tensors,
    _sample_loss,
    _validate_checkpoint_compatibility,
    evaluate_model,
    experiment_directory,
    set_reproducible_seed,
    train_model,
)

EXPERIMENT = ROOT / "experiments/v2/composite_loss"


def candidate_config(name, preflight=False):
    candidates = json.loads((EXPERIMENT / "candidates.json").read_text())
    if name not in {item["name"] for item in candidates}:
        raise ValueError(f"Unknown candidate: {name}")
    config = ExperimentConfig.from_toml(EXPERIMENT / "configs" / f"{name}.toml")
    if config.model.hidden_dim != 64 or not config.loss.record_diagnostics:
        raise ValueError("This experiment requires width 64 and diagnostics")
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


def loss_provenance(config):
    result = provenance(config)
    result["loss"] = config.as_dict()["loss"]
    result["runner_sha256"] = sha256(Path(__file__))
    result["provenance_helper_sha256"] = sha256(ROOT / "scripts/run_width_experiment.py")
    return result


def check_existing(directory, metadata, resume):
    manifest = directory / "loss_experiment.json"
    if manifest.exists():
        if not resume:
            raise RuntimeError("Run exists; use --resume")
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError("Run provenance changed; use a separate experiment")
    elif any(directory.glob("*.pt")) or (directory / "history.csv").exists():
        raise RuntimeError("Untracked existing run; refusing to overwrite")


def run(name, mode, resume=False, segment_hours=43):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    config = candidate_config(name, mode == "preflight")
    directory = experiment_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".loss_job.lock"
    with lock.open("x") as handle:
        handle.write(json.dumps({"pbs_job_id": os.environ.get("PBS_JOBID"), "pid": os.getpid()}))
    try:
        metadata = loss_provenance(config)
        check_existing(directory, metadata, resume)
        write_json(directory / "loss_experiment.json", metadata)
        status_path = directory / "training_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if mode == "evaluate":
            if status.get("state") != "complete":
                raise RuntimeError("Complete training before explicitly evaluating a finalist")
            best = directory / "best.pt"
            marker = directory / "loss_evaluation.json"
            if marker.exists():
                saved = json.loads(marker.read_text())
                if (
                    saved["checkpoint_sha256"] == sha256(best)
                    and sha256(directory / "test_event_metrics.csv") == saved["metrics_sha256"]
                ):
                    print("Evaluation already complete")
                    return directory
            metrics = evaluate_model(config, best, save_predictions=True)
            checkpoint = torch.load(best, map_location="cpu", weights_only=False)
            norm_name = checkpoint["normalization_path"].replace("\\", "/").split("/")[-1]
            write_json(
                marker,
                {
                    "checkpoint_sha256": sha256(best),
                    "checkpoint_epoch": checkpoint["epoch"],
                    "metrics_sha256": sha256(metrics),
                    "normalization_sha256": sha256(directory / norm_name),
                },
            )
            print(f"Finalist evaluation complete: {directory}")
            return directory
        if status.get("state") != "complete":
            train_model(config, resume=resume, segment_seconds=segment_hours * 3600)
            status = json.loads(status_path.read_text())
        print(f"{mode}: {status['state']}; {directory}")
        print(
            "No test evaluation during loss calibration/search. Selection remains validation RMSE."
        )
        return directory
    finally:
        lock.unlink()


def calibrate(samples=24):
    """Training-only gradients at random initialization and the frozen MSE control.

    No optimizer updates, no validation/test predictions, and no automatic choice
    of lambda. Reports norms before clipping and gradient alignment for 6 T values.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Calibration requires one visible CUDA GPU")
    config = candidate_config("control")
    device = torch.device("cuda")
    base = ROOT / "outputs/v2/hidden_width/width_64/fold_1/seed_42"
    checkpoint = torch.load(base / "best.pt", map_location="cpu", weights_only=False)
    _validate_checkpoint_compatibility(config, checkpoint)
    marker = json.loads((base / "width_evaluation.json").read_text())
    if sha256(base / "best.pt") != marker["checkpoint_sha256"]:
        raise ValueError("Frozen MSE checkpoint hash differs")
    normalization_name = checkpoint["normalization_path"].replace("\\", "/").split("/")[-1]
    normalization = NormalizationBundle.load(base / normalization_name)
    if sha256(base / normalization_name) != marker["normalization_sha256"]:
        raise ValueError("Frozen normalization hash differs")
    graph, _ = prepare_aoi_graph(config)
    catalog = EventCatalog(config.paths.dataset_root)
    splits = load_fold_samples(config, catalog)
    refs = sorted(splits.train, key=lambda r: (r.event_index, r.timestep))
    refs = [
        refs[i]
        for i in np.unique(np.linspace(0, len(refs) - 1, min(samples, len(refs)), dtype=int))
    ]
    static = to_heterodata(graph, normalization.normalized_edges(graph)).to(device)
    _, fine_geometry = load_burnett_geometries(config.paths.dataset_root)
    terrain = torch.from_numpy(
        fine_geometry.elevation[graph.fine_global_ids[graph.target_mask]]
    ).to(device)
    target_local = torch.from_numpy(graph.target_local_ids).to(device)
    records = []
    for phase in ("initialization", "mse_checkpoint"):
        set_reproducible_seed(config.training.seed)
        model = TwoScaleFloodGNN(config.model).to(device)
        if phase == "mse_checkpoint":
            model.load_state_dict(checkpoint["model_state"])
        model.eval()
        parameters = tuple(model.parameters())
        with EventDataAccess(config, graph, catalog) as access:
            for ref in refs:
                sample = access.sample(ref)
                hec, fine, target = _normalized_tensors(sample, normalization, device)
                prediction = model(static, hec, fine)[target_local]
                for temperature in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
                    cfg = replace(config, loss=replace(config.loss, temperature_m=temperature))
                    terms = _sample_loss(
                        prediction, target, sample, graph, normalization, terrain, cfg, device
                    )
                    mse_grad = torch.autograd.grad(
                        terms.residual_mse, parameters, retain_graph=True
                    )
                    extent_grad = torch.autograd.grad(terms.extent, parameters, retain_graph=True)
                    mse_norm = torch.stack([g.square().sum() for g in mse_grad]).sum().sqrt().item()
                    extent_norm = (
                        torch.stack([g.square().sum() for g in extent_grad]).sum().sqrt().item()
                    )
                    dot = (
                        torch.stack([(a * b).sum() for a, b in zip(mse_grad, extent_grad)])
                        .sum()
                        .item()
                    )
                    cosine = (
                        dot / (mse_norm * extent_norm)
                        if min(mse_norm, extent_norm) > 1e-12
                        else None
                    )
                    for weight in (0.001, 0.01, 0.1, 1.0, 10.0):
                        records.append(
                            {
                                "phase": phase,
                                "event_index": ref.event_index,
                                "timestep": ref.timestep,
                                "temperature_m": temperature,
                                "extent_weight": weight,
                                "residual_mse": terms.residual_mse.item(),
                                "extent_loss": terms.extent.item(),
                                "weighted_extent_loss": weight * terms.extent.item(),
                                "mse_gradient_norm": mse_norm,
                                "extent_gradient_norm": extent_norm,
                                "weighted_gradient_ratio": weight * extent_norm / mse_norm
                                if mse_norm > 1e-12
                                else None,
                                "gradient_cosine": cosine,
                                "soft_saturated_fraction": (
                                    (terms.soft_wet < 0.01) | (terms.soft_wet > 0.99)
                                )
                                .float()
                                .mean()
                                .item(),
                                "true_wet_cells": int(
                                    np.count_nonzero(sample.target_depth >= 0.03)
                                ),
                            }
                        )
                del prediction, terms, mse_grad, extent_grad
        del model
    output = ROOT / "outputs/v2/composite_loss/calibration" / os.environ.get("PBS_JOBID", "local")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "training_gradient_probe.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(
        output / "calibration.json",
        {
            "training_only": True,
            "sample_count": len(refs),
            "sample_selection": "Evenly spaced indices from sorted training refs; not representative of all wetness regimes",
            "checkpoint_sha256": marker["checkpoint_sha256"],
            "normalization_sha256": marker["normalization_sha256"],
            "provenance": loss_provenance(config),
            "note": "Gradient calibration only; no updates and no automatic hyperparameter selection.",
        },
    )
    print(f"Training-only calibration saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="control")
    parser.add_argument(
        "--mode", choices=("calibration", "preflight", "train", "evaluate"), required=True
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-hours", type=float, default=43)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    if args.segment_hours <= 0:
        parser.error("segment-hours must be positive")
    if args.mode == "calibration":
        calibrate(args.samples)
    else:
        run(args.candidate, args.mode, args.resume, args.segment_hours)


if __name__ == "__main__":
    main()
