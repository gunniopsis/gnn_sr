from __future__ import annotations

import csv
import gzip
import json
import random
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .data import (
    EventDataAccess,
    FloodSample,
    NormalizationBundle,
    SampleRef,
    fit_normalization,
    grouped_shuffled_refs,
    load_fold_samples,
)
from .diagnostics import STEP_FIELDS, SUMMARY_FIELDS, TimestepDiagnostics
from .geometry import load_burnett_geometries
from .graph import GraphArrays, prepare_aoi_graph, to_heterodata
from .io import EventCatalog
from .losses import composite_loss
from .metrics import calculate_event_metrics, mean_event_metrics
from .model import TwoScaleFloodGNN, physical_outputs


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch installation cannot access a CUDA GPU. "
                "On Gadi, submit through a GPU queue with ngpus=1 and use a CUDA-enabled "
                "PyTorch installation."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible."
            )
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available.")
    return device


def _device_summary(device: torch.device) -> dict[str, str | int | bool | None]:
    summary: dict[str, str | int | bool | None] = {
        "training_device": str(device),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": torch.cuda.device_count(),
        "cuda_device_name": None,
    }
    if device.type == "cuda":
        summary["cuda_device_name"] = torch.cuda.get_device_name(device)
    return summary


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _process_peak_rss_bytes() -> int | None:
    """Return Linux's process high-water RSS, or None when unavailable."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def experiment_directory(config: ExperimentConfig) -> Path:
    return config.paths.output_dir / f"fold_{config.training.fold}" / f"seed_{config.training.seed}"


def _normalized_tensors(
    sample: FloodSample, normalization: NormalizationBundle, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hec = torch.from_numpy(normalization.hec_nodes.transform(sample.hec_features)).to(device)
    fine = torch.from_numpy(normalization.fine_nodes.transform(sample.fine_features)).to(device)
    target = normalization.target_residual.transform(sample.target_residual[:, None])[:, 0]
    target_tensor = torch.from_numpy(target).to(device)
    return hec, fine, target_tensor


def _predicted_physical_depth(
    prediction_normalized: torch.Tensor,
    sample: FloodSample,
    graph: GraphArrays,
    normalization: NormalizationBundle,
    target_terrain: torch.Tensor,
    dry_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    target_local = torch.from_numpy(graph.target_local_ids).to(device)
    prediction = prediction_normalized[target_local]
    mean = torch.as_tensor(normalization.target_residual.mean[0], device=device)
    scale = torch.as_tensor(normalization.target_residual.scale[0], device=device)
    residual = prediction * scale + mean
    baseline = torch.from_numpy(sample.baseline_wse[graph.target_mask]).to(device)
    raw_wse = baseline + residual
    _, depth = physical_outputs(raw_wse, target_terrain, dry_threshold)
    return depth


@torch.no_grad()
def validation_depth_rmse(
    model: TwoScaleFloodGNN,
    static_graph,
    graph: GraphArrays,
    normalization: NormalizationBundle,
    access: EventDataAccess,
    refs: Sequence[SampleRef],
    target_terrain: torch.Tensor,
    dry_threshold: float,
    device: torch.device,
    diagnostics: TimestepDiagnostics | None = None,
    config: ExperimentConfig | None = None,
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for ref in sorted(refs, key=lambda item: (item.event_index, item.timestep)):
        sample = access.sample(ref)
        hec, fine, target = _normalized_tensors(sample, normalization, device)
        prediction = model(static_graph, hec, fine)
        predicted_depth = _predicted_physical_depth(
            prediction,
            sample,
            graph,
            normalization,
            target_terrain,
            dry_threshold,
            device,
        )
        target_depth = torch.from_numpy(sample.target_depth).to(device)
        if diagnostics is not None:
            selected = prediction[torch.from_numpy(graph.target_local_ids).to(device)]
            terms = _sample_loss(selected, target, sample, graph, normalization,
                                 target_terrain, config, device)
            diagnostics.add(terms, target_depth, ref)
        squared_error += torch.sum(torch.square(predicted_depth - target_depth)).item()
        count += target_depth.numel()
    return float(np.sqrt(squared_error / count))


def _sample_loss(prediction, target, sample, graph, normalization, terrain, config, device):
    return composite_loss(
        prediction, target,
        torch.from_numpy(sample.baseline_wse[graph.target_mask]).to(device), terrain,
        torch.from_numpy(sample.target_depth).to(device),
        torch.as_tensor(normalization.target_residual.mean[0], device=device),
        torch.as_tensor(normalization.target_residual.scale[0], device=device),
        config.loss.temperature_m, config.loss.extent_weight, config.data.dry_threshold_m,
    )


def _save_checkpoint(
    path: Path,
    model: TwoScaleFloodGNN,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    validation_rmse: float,
    best_rmse: float,
    bad_epochs: int,
    config: ExperimentConfig,
    graph_path: Path,
    normalization_path: Path,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "validation_depth_rmse_m": validation_rmse,
        "best_validation_depth_rmse_m": best_rmse,
        "bad_epochs": bad_epochs,
        "config": config.as_dict(),
        "graph_path": str(graph_path),
        "normalization_path": str(normalization_path),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _restore_random_state(checkpoint: dict) -> None:
    python_state = checkpoint.get("python_random_state")
    numpy_state = checkpoint.get("numpy_random_state")
    torch_state = checkpoint.get("torch_random_state")
    cuda_states = checkpoint.get("cuda_random_state_all")
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu())
    if cuda_states and torch.cuda.is_available():
        # Checkpoints loaded with ``map_location=device`` also move these saved
        # RNG tensors to CUDA. PyTorch's CUDA generator API nevertheless
        # requires CPU ByteTensors when restoring its state.
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _validate_resume_checkpoint(config: ExperimentConfig, checkpoint: dict) -> None:
    _validate_checkpoint_compatibility(config, checkpoint)
    from .config import LossConfig
    saved_loss = checkpoint["config"].get("loss", vars(LossConfig()))
    if saved_loss != config.as_dict()["loss"]:
        raise ValueError("Resume checkpoint has different loss settings")
    saved_training = checkpoint["config"].get("training", {})
    current_training = config.as_dict()["training"]
    operational_fields = {"device", "max_epochs", "extend_epoch_cap_if_improving"}
    mismatches = [
        field
        for field, current_value in current_training.items()
        if field not in operational_fields and saved_training.get(field) != current_value
    ]
    if mismatches:
        raise ValueError(
            "Resume checkpoint does not match the current training configuration: "
            + ", ".join(f"training.{field}" for field in mismatches)
        )


def _json_safe(values):
    if isinstance(values, dict):
        return {key: _json_safe(value) for key, value in values.items()}
    if isinstance(values, list):
        return [_json_safe(value) for value in values]
    if isinstance(values, float) and not np.isfinite(values):
        return None
    return values


def _restored_epoch_limit(config: ExperimentConfig, epoch: int, bad_epochs: int) -> int:
    """Recover the automatic epoch-cap extensions when a job resumes."""
    limit = config.training.max_epochs
    patience = config.training.early_stopping_patience
    if config.training.extend_epoch_cap_if_improving and bad_epochs < patience and epoch >= limit:
        limit += ((epoch - limit) // patience + 1) * patience
    return limit


def train_model(
    config: ExperimentConfig,
    force_graph: bool = False,
    resume: bool = False,
    segment_seconds: float | None = None,
) -> Path:
    if segment_seconds is not None and segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    invocation_started = time.perf_counter()
    set_reproducible_seed(config.training.seed)
    device = resolve_device(config.training.device)
    print(json.dumps(_device_summary(device)))
    run_dir = experiment_directory(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    graph, graph_path = prepare_aoi_graph(config, force=force_graph)
    catalog = EventCatalog(config.paths.dataset_root)
    splits = load_fold_samples(config, catalog)
    normalization_path = run_dir / f"normalization_{graph_path.stem}.json"
    if normalization_path.exists():
        normalization = NormalizationBundle.load(normalization_path)
    else:
        normalization = fit_normalization(config, graph, catalog, splits.train)
        normalization.save(normalization_path)
    static_graph = to_heterodata(graph, normalization.normalized_edges(graph)).to(device)
    model = TwoScaleFloodGNN(config.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=config.training.minimum_learning_rate,
    )
    _, fine_geometry = load_burnett_geometries(config.paths.dataset_root)
    target_terrain = torch.from_numpy(
        fine_geometry.elevation[graph.fine_global_ids[graph.target_mask]]
    ).to(device)
    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"
    history_path = run_dir / "history.csv"
    status_path = run_dir / "training_status.json"
    best_rmse = float("inf")
    bad_epochs = 0
    epoch_limit = config.training.max_epochs
    epoch = 0
    fieldnames = [
        "epoch",
        "training_mse",
        "validation_depth_rmse_m",
        "learning_rate",
        "mean_gradient_norm_before_clipping",
        "clipped_step_fraction",
        "training_steps",
        "validation_steps",
        "training_seconds",
        "validation_seconds",
        "mean_training_step_seconds",
        "mean_validation_step_seconds",
        "process_peak_rss_bytes",
        "gpu_peak_memory_allocated_bytes",
        "gpu_peak_memory_reserved_bytes",
        "epoch_seconds",
    ]
    diagnostics_enabled = config.loss.record_diagnostics
    if diagnostics_enabled:
        fieldnames += [f"{phase}_{key}" for phase in ("train", "validation")
                       for key in SUMMARY_FIELDS]
        fieldnames += ["loss_temperature_m", "loss_extent_weight"]

    resume_checkpoint: dict | None = None
    resume_path: Path | None = None
    if resume:
        if latest_path.exists():
            resume_path = latest_path
        elif best_path.exists():
            resume_path = best_path
        elif history_path.exists():
            raise RuntimeError(
                f"Cannot resume because {history_path} exists but no checkpoint is available"
            )
        if resume_path is not None:
            resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            _validate_resume_checkpoint(config, resume_checkpoint)
            model.load_state_dict(resume_checkpoint["model_state"])
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            epoch = int(resume_checkpoint["epoch"])
            best_rmse = float(
                resume_checkpoint.get(
                    "best_validation_depth_rmse_m",
                    resume_checkpoint["validation_depth_rmse_m"],
                )
            )
            bad_epochs = int(resume_checkpoint.get("bad_epochs", 0))
            epoch_limit = _restored_epoch_limit(config, epoch, bad_epochs)
            _restore_random_state(resume_checkpoint)
            print(
                json.dumps(
                    {
                        "resume_checkpoint": str(resume_path),
                        "resume_epoch": epoch,
                        "best_validation_depth_rmse_m": best_rmse,
                        "bad_epochs": bad_epochs,
                    }
                )
            )

    if resume_checkpoint is not None and not history_path.exists():
        raise RuntimeError(f"Cannot resume epoch {epoch} because {history_path} is missing")

    if resume_checkpoint is not None:
        with history_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            if reader.fieldnames != fieldnames:
                raise ValueError("Existing training history columns do not match this code version")
        retained_rows = [row for row in existing_rows if int(row["epoch"]) <= epoch]
        if len(retained_rows) != len(existing_rows):
            temporary_history = history_path.with_suffix(".csv.tmp")
            with temporary_history.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(retained_rows)
            temporary_history.replace(history_path)
            print(
                f"Discarded {len(existing_rows) - len(retained_rows)} history row(s) "
                f"newer than resume epoch {epoch}."
            )

    history_mode = "a" if resume_checkpoint is not None else "w"
    stop_reason: str | None = None
    with history_path.open(history_mode, newline="", encoding="utf-8") as history_handle:
        writer = csv.DictWriter(history_handle, fieldnames=fieldnames)
        if history_mode == "w":
            writer.writeheader()
        with EventDataAccess(config, graph, catalog) as access, ExitStack() as files:
            while (
                epoch < epoch_limit
                and bad_epochs < config.training.early_stopping_patience
            ):
                epoch += 1
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                _synchronize_device(device)
                started = time.perf_counter()
                model.train()
                loss_sum = 0.0
                steps = 0
                gradient_norm_sum = 0.0
                clipped_steps = 0
                train_diagnostics = validation_diagnostics = timestep_handle = None
                if diagnostics_enabled:
                    diagnostics_dir = run_dir / "timestep_metrics"
                    diagnostics_dir.mkdir(exist_ok=True)
                    timestep_handle = files.enter_context(gzip.open(
                        diagnostics_dir / f"epoch_{epoch:05d}.csv.gz", "wt",
                        newline="", encoding="utf-8"))
                    timestep_writer = csv.DictWriter(timestep_handle, fieldnames=[
                        "epoch", "phase", "event_index", "timestep", *STEP_FIELDS])
                    timestep_writer.writeheader()
                    train_diagnostics = TimestepDiagnostics(
                        config.data.dry_threshold_m, timestep_writer, epoch, "train")
                    validation_diagnostics = TimestepDiagnostics(
                        config.data.dry_threshold_m, timestep_writer, epoch, "validation")
                refs = grouped_shuffled_refs(splits.train, config.training.seed + epoch)
                target_local = torch.from_numpy(graph.target_local_ids).to(device)
                for ref in refs:
                    sample = access.sample(ref)
                    hec, fine, target = _normalized_tensors(sample, normalization, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(static_graph, hec, fine)[target_local]
                    if diagnostics_enabled or config.loss.extent_weight > 0:
                        terms = _sample_loss(prediction, target, sample, graph, normalization,
                                             target_terrain, config, device)
                        loss = terms.total
                        mse = terms.residual_mse
                        if train_diagnostics is not None:
                            train_diagnostics.add(
                                terms, torch.from_numpy(sample.target_depth).to(device), ref)
                    else:
                        loss = mse = nn.functional.mse_loss(prediction, target)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.gradient_clip_norm
                    )
                    gradient_norm_value = float(gradient_norm.item())
                    gradient_norm_sum += gradient_norm_value
                    clipped_steps += int(gradient_norm_value > config.training.gradient_clip_norm)
                    optimizer.step()
                    loss_sum += float(mse.item())
                    steps += 1

                _synchronize_device(device)
                training_seconds = time.perf_counter() - started
                validation_started = time.perf_counter()
                validation_rmse = validation_depth_rmse(
                    model,
                    static_graph,
                    graph,
                    normalization,
                    access,
                    splits.validation,
                    target_terrain,
                    config.data.dry_threshold_m,
                    device,
                    diagnostics=validation_diagnostics,
                    config=config,
                )
                if timestep_handle is not None:
                    timestep_handle.close()
                _synchronize_device(device)
                validation_seconds = time.perf_counter() - validation_started
                scheduler.step(validation_rmse)
                improved = validation_rmse < best_rmse
                if improved:
                    best_rmse = validation_rmse
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                row = {
                    "epoch": epoch,
                    "training_mse": loss_sum / max(steps, 1),
                    "validation_depth_rmse_m": validation_rmse,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "mean_gradient_norm_before_clipping": gradient_norm_sum / max(steps, 1),
                    "clipped_step_fraction": clipped_steps / max(steps, 1),
                    "training_steps": steps,
                    "validation_steps": len(splits.validation),
                    "training_seconds": training_seconds,
                    "validation_seconds": validation_seconds,
                    "mean_training_step_seconds": training_seconds / max(steps, 1),
                    "mean_validation_step_seconds": validation_seconds
                    / max(len(splits.validation), 1),
                    "process_peak_rss_bytes": _process_peak_rss_bytes(),
                    "gpu_peak_memory_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                    ),
                    "gpu_peak_memory_reserved_bytes": (
                        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                    ),
                    "epoch_seconds": time.perf_counter() - started,
                }
                if diagnostics_enabled:
                    for phase, diagnostics in (("train", train_diagnostics),
                                               ("validation", validation_diagnostics)):
                        row.update({f"{phase}_{key}": value
                                    for key, value in diagnostics.summary().items()})
                    row.update(loss_temperature_m=config.loss.temperature_m,
                               loss_extent_weight=config.loss.extent_weight)
                writer.writerow(row)
                history_handle.flush()
                print(json.dumps(row))
                if improved:
                    _save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        validation_rmse,
                        best_rmse,
                        bad_epochs,
                        config,
                        graph_path,
                        normalization_path,
                    )
                _save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    validation_rmse,
                    best_rmse,
                    bad_epochs,
                    config,
                    graph_path,
                    normalization_path,
                )
                if bad_epochs >= config.training.early_stopping_patience:
                    stop_reason = "early_stopping"
                    break
                elapsed = time.perf_counter() - invocation_started
                if segment_seconds is not None and elapsed + row["epoch_seconds"] >= segment_seconds:
                    stop_reason = "segment_limit"
                    print(
                        f"Stopping this segment after epoch {epoch}: elapsed time plus one "
                        f"estimated epoch reached the {segment_seconds / 3600:.2f}-hour limit."
                    )
                    break
                if (
                    config.training.extend_epoch_cap_if_improving
                    and epoch == epoch_limit
                    and bad_epochs < config.training.early_stopping_patience
                ):
                    epoch_limit += config.training.early_stopping_patience
                    print(
                        f"Validation is still within early-stopping patience at epoch {epoch}; "
                        f"extending the cap to {epoch_limit}."
                    )
    if stop_reason is None:
        if bad_epochs >= config.training.early_stopping_patience:
            stop_reason = "early_stopping"
        elif epoch >= epoch_limit:
            stop_reason = "max_epochs"
        else:  # pragma: no cover - defensive state guard
            stop_reason = "unknown"

    complete = stop_reason in {"early_stopping", "max_epochs"}
    status = {
        "state": "complete" if complete else "segment_complete",
        "reason": stop_reason,
        "epoch": epoch,
        "max_epochs": epoch_limit,
        "best_validation_depth_rmse_m": best_rmse,
        "bad_epochs": bad_epochs,
        "early_stopping_patience": config.training.early_stopping_patience,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    temporary_status = status_path.with_suffix(".json.tmp")
    temporary_status.write_text(json.dumps(_json_safe(status), indent=2), encoding="utf-8")
    temporary_status.replace(status_path)
    print(json.dumps({"training_status": _json_safe(status)}))
    if not best_path.exists():
        raise RuntimeError("Training completed without producing a checkpoint")
    return best_path


def _load_model_from_checkpoint(
    config: ExperimentConfig, checkpoint_path: str | Path, device: torch.device
) -> tuple[TwoScaleFloodGNN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TwoScaleFloodGNN(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _validate_checkpoint_compatibility(config: ExperimentConfig, checkpoint: dict) -> None:
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise TypeError("The checkpoint does not contain its training configuration")
    current = config.as_dict()
    mismatches = [
        section
        for section in ("data", "graph", "model")
        if saved.get(section) != current.get(section)
    ]
    saved_training = saved.get("training", {})
    if saved_training.get("fold") != config.training.fold:
        mismatches.append("training.fold")
    saved_aoi = Path(saved.get("paths", {}).get("aoi_path", "")).name
    current_aoi = config.paths.aoi_path.name if config.paths.aoi_path is not None else ""
    if not saved_aoi or saved_aoi != current_aoi:
        mismatches.append("paths.aoi_path")
    if mismatches:
        raise ValueError(
            "The checkpoint configuration is incompatible with this evaluation configuration: "
            + ", ".join(mismatches)
        )


def _resolve_checkpoint_artifact(checkpoint_path: Path, recorded_path: str | Path) -> Path:
    recorded = Path(recorded_path)
    if recorded.exists():
        return recorded
    companion = checkpoint_path.parent / recorded.name
    if companion.exists():
        return companion
    raise FileNotFoundError(
        f"Checkpoint artifact not found at its recorded path ({recorded}) or beside the "
        f"downloaded checkpoint ({companion})"
    )


@torch.no_grad()
def evaluate_model(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    save_predictions: bool = True,
) -> Path:
    device = resolve_device(config.training.device)
    graph, _ = prepare_aoi_graph(config)
    catalog = EventCatalog(config.paths.dataset_root)
    splits = load_fold_samples(config, catalog)
    checkpoint_path = Path(checkpoint_path).resolve()
    model, checkpoint = _load_model_from_checkpoint(config, checkpoint_path, device)
    _validate_checkpoint_compatibility(config, checkpoint)
    normalization_path = _resolve_checkpoint_artifact(
        checkpoint_path, checkpoint["normalization_path"]
    )
    normalization = NormalizationBundle.load(normalization_path)
    static_graph = to_heterodata(graph, normalization.normalized_edges(graph)).to(device)
    _, fine_geometry = load_burnett_geometries(config.paths.dataset_root)
    target_terrain_np = fine_geometry.elevation[graph.fine_global_ids[graph.target_mask]]
    target_terrain = torch.from_numpy(target_terrain_np).to(device)
    target_local = torch.from_numpy(graph.target_local_ids).to(device)
    run_dir = Path(checkpoint_path).resolve().parent
    prediction_dir = run_dir / "predictions"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    event_rows: list[dict[str, float | int | str]] = []
    event_metric_objects = []
    grouped: dict[int, list[SampleRef]] = {}
    for ref in splits.test:
        grouped.setdefault(ref.event_index, []).append(ref)
    with EventDataAccess(config, graph, catalog) as access:
        for event_index, refs in grouped.items():
            refs.sort(key=lambda ref: ref.timestep)
            predicted_depths = []
            predicted_wses = []
            raw_wses = []
            target_depths = []
            target_wses = []
            forward_seconds = 0.0
            event_started = time.perf_counter()
            for ref in refs:
                sample = access.sample(ref)
                hec, fine, _ = _normalized_tensors(sample, normalization, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                prediction_normalized = model(static_graph, hec, fine)[target_local]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                forward_seconds += time.perf_counter() - started
                mean = torch.as_tensor(normalization.target_residual.mean[0], device=device)
                scale = torch.as_tensor(normalization.target_residual.scale[0], device=device)
                residual = prediction_normalized * scale + mean
                baseline = torch.from_numpy(sample.baseline_wse[graph.target_mask]).to(device)
                raw_wse = baseline + residual
                physical_wse, depth = physical_outputs(
                    raw_wse, target_terrain, config.data.dry_threshold_m
                )
                raw_wses.append(raw_wse.cpu().numpy())
                predicted_wses.append(physical_wse.cpu().numpy())
                predicted_depths.append(depth.cpu().numpy())
                target_wses.append(sample.target_wse)
                target_depths.append(sample.target_depth)
            predicted_depth = np.stack(predicted_depths)
            predicted_wse = np.stack(predicted_wses)
            raw_wse = np.stack(raw_wses)
            target_depth = np.stack(target_depths)
            target_wse = np.stack(target_wses)
            metrics = calculate_event_metrics(
                target_depth, predicted_depth, config.data.dry_threshold_m
            )
            event_metric_objects.append(metrics)
            event = catalog.records[event_index]
            row: dict[str, float | int | str] = {
                "event_index": event_index,
                "event_name": event.name,
                "group": event.group,
                **metrics.as_dict(),
                "model_forward_seconds": forward_seconds,
                "end_to_end_seconds": time.perf_counter() - event_started,
                "n_timesteps": len(refs),
                "n_target_cells": len(target_terrain_np),
            }
            event_rows.append(row)
            if save_predictions:
                np.savez_compressed(
                    prediction_dir / f"{event.name}.npz",
                    fine_global_ids=graph.fine_global_ids[graph.target_mask],
                    terrain=target_terrain_np,
                    target_wse=target_wse,
                    target_depth=target_depth,
                    raw_predicted_wse=raw_wse,
                    predicted_wse=predicted_wse,
                    predicted_depth=predicted_depth,
                )
    metrics_path = run_dir / "test_event_metrics.csv"
    if event_rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
            writer.writeheader()
            writer.writerows(event_rows)
        summary = mean_event_metrics(event_metric_objects)
        summary_payload = {
            "mean_event_metrics": summary.as_dict(),
            "total_model_forward_seconds": sum(
                float(row["model_forward_seconds"]) for row in event_rows
            ),
            "total_end_to_end_seconds": sum(float(row["end_to_end_seconds"]) for row in event_rows),
            "events": len(event_rows),
        }
        (run_dir / "test_summary.json").write_text(
            json.dumps(_json_safe(summary_payload), indent=2), encoding="utf-8"
        )
    return metrics_path
