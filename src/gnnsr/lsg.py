from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .config import ExperimentConfig
from .data import load_fold_samples, physical_wse_and_depth
from .geometry import load_burnett_geometries
from .graph import prepare_aoi_graph
from .io import EventCatalog, HecEvent, TuflowEvent
from .metrics import calculate_event_metrics, mean_event_metrics


class SavedSparseGP:
    """Minimal loader for the authors' saved GPflow SGPR ensemble."""

    def __init__(self, path: str | Path):
        try:
            import gpflow
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LSG evaluation requires the optional dependencies. Install with "
                '`python -m pip install -e ".[lsg]"` in a compatible environment.'
            ) from exc
        gpflow.config.set_default_float(np.float64)
        gpflow.config.set_default_jitter(1e-1)
        with np.load(path, allow_pickle=True) as archive:
            x_data = np.asarray(archive["x_data"], dtype=np.float64)
            y_data = archive["y_data_list"]
            parameters = archive["params_list"][()]
            self.x_scaler = archive["x_scaler"][()]
            self.y_scaler = archive["y_scaler"][()]
        self.input_modes = x_data.shape[1]
        self.models = []
        for output_index, parameter_values in enumerate(parameters):
            kernel = gpflow.kernels.Exponential(
                variance=1,
                lengthscales=parameter_values[".kernel.lengthscales"],
            )
            model = gpflow.models.SGPR(
                data=(x_data, y_data[output_index]),
                kernel=kernel,
                inducing_variable=parameter_values[".inducing_variable.Z"],
            )
            gpflow.utilities.multiple_assign(model, parameter_values)
            self.models.append(model)

    def predict(self, low_fidelity_coefficients: np.ndarray) -> np.ndarray:
        scaled = self.x_scaler.transform(low_fidelity_coefficients)
        scaled = scaled[:, : self.input_modes]
        predictions = [model.predict_y(scaled)[0].numpy() for model in self.models]
        return self.y_scaler.inverse_transform(np.column_stack(predictions))


@dataclass(frozen=True)
class EOFData:
    modes: np.ndarray
    mean: np.ndarray
    weights: np.ndarray
    n_modes: int

    @classmethod
    def load(cls, path: str | Path) -> EOFData:
        with np.load(path, allow_pickle=True) as archive:
            return cls(
                modes=np.asarray(archive["EOFs"]),
                mean=np.asarray(archive["mean_train"]),
                weights=np.asarray(archive["Area"]),
                n_modes=int(archive["n_ECs"]),
            )

    def coefficients(self, values: np.ndarray) -> np.ndarray:
        centered_weighted = (values - self.mean) * self.weights
        return centered_weighted @ self.modes[: self.n_modes].T


def _reconstruct_subset(
    coefficients: np.ndarray, eof: EOFData, subset_positions: np.ndarray
) -> np.ndarray:
    return (coefficients @ eof.modes[: coefficients.shape[1], subset_positions]) / eof.weights[
        subset_positions
    ] + eof.mean[subset_positions]


def _low_fidelity_coefficients(
    config: ExperimentConfig,
    event,
    nearest_coarse: np.ndarray,
    coarse_elevation: np.ndarray,
    fine_elevation: np.ndarray,
    wse_wet_indices: np.ndarray,
    extent_tf_indices: np.ndarray,
    wse_eof: EOFData,
    extent_eof: EOFData,
    required_timesteps: int | None = None,
    chunk_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    all_coarse = np.arange(len(coarse_elevation), dtype=np.int64)
    tuflow = TuflowEvent(event.tuflow_path, config.data.tuflow_warmup_steps)
    wse_coefficients = []
    extent_coefficients = []
    with HecEvent(
        event.hec_path,
        config.data.flow_area_name,
        all_coarse,
        config.data.hec_warmup_hours,
    ) as hec:
        if hec.n_timesteps < tuflow.n_timesteps:
            raise ValueError(f"HEC-RAS is shorter than TUFLOW for {event.name}")
        total_timesteps = tuflow.n_timesteps
        if required_timesteps is not None:
            total_timesteps = min(total_timesteps, required_timesteps)
        for start in range(0, total_timesteps, chunk_size):
            stop = min(start + chunk_size, total_timesteps)
            hec_chunk = np.stack([hec.read_wse(index) for index in range(start, stop)])
            hec_chunk = np.where(
                hec_chunk > coarse_elevation + config.data.dry_threshold_m,
                hec_chunk,
                coarse_elevation,
            )
            wet_values = hec_chunk[:, nearest_coarse[wse_wet_indices]]
            wet_values = np.where(
                wet_values > fine_elevation[wse_wet_indices] + config.data.dry_threshold_m,
                wet_values,
                fine_elevation[wse_wet_indices],
            )
            tf_values = hec_chunk[:, nearest_coarse[extent_tf_indices]]
            tf_binary = (
                tf_values >= fine_elevation[extent_tf_indices] + config.data.dry_threshold_m
            ).astype(np.float32)
            wse_coefficients.append(wse_eof.coefficients(wet_values))
            extent_coefficients.append(extent_eof.coefficients(tf_binary))
    return np.vstack(wse_coefficients), np.vstack(extent_coefficients)


def evaluate_lsg_aoi(config: ExperimentConfig) -> Path:
    """Evaluate the authors' full-domain LSG models on the configured AOI cells only."""
    graph, _ = prepare_aoi_graph(config)
    catalog = EventCatalog(config.paths.dataset_root)
    splits = load_fold_samples(config, catalog)
    fold = config.training.fold
    root = config.paths.dataset_root
    analysis = root / "HF_EOF_analysis"
    models = root / "LSG_modeldata"
    categories_path = analysis / f"Categories_HFdata_ValidateOnGrp_{fold}.npz"
    with np.load(categories_path) as categories:
        wet_indices = np.asarray(categories["wet_idx"], dtype=np.int64)
        tf_indices = np.asarray(categories["TF_idx"], dtype=np.int64)
        af_indices = np.asarray(categories["AF_idx"], dtype=np.int64)
    wse_eof = EOFData.load(analysis / f"EOFanalysis_HFdata_WSE_ValidateOnGrp_{fold}.npz")
    extent_eof = EOFData.load(analysis / f"EOFanalysis_HFdata_EXT_ValidateOnGrp_{fold}.npz")
    wse_gp = SavedSparseGP(models / f"LSG_WSE_ValidateOnGrp_{fold}.npz")
    extent_gp = SavedSparseGP(models / f"LSG_Ext_ValidateOnGrp_{fold}.npz")
    coarse, fine = load_burnett_geometries(root)
    nearest_coarse = cKDTree(coarse.xy).query(fine.xy, k=1)[1]
    target_ids = graph.fine_global_ids[graph.target_mask]
    target_elevation = fine.elevation[target_ids]

    wet_position = np.full(len(fine.xy), -1, dtype=np.int64)
    wet_position[wet_indices] = np.arange(len(wet_indices))
    tf_position = np.full(len(fine.xy), -1, dtype=np.int64)
    tf_position[tf_indices] = np.arange(len(tf_indices))
    target_wet = wet_position[target_ids] >= 0
    target_tf = tf_position[target_ids] >= 0
    target_af = np.isin(target_ids, af_indices)

    grouped = {}
    for ref in splits.test:
        grouped.setdefault(ref.event_index, []).append(ref)
    output_dir = config.paths.output_dir / "lsg" / f"fold_{fold}" / "aoi_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metric_objects = []
    for event_index, requested_refs in grouped.items():
        event = catalog.records[event_index]
        event_started = time.perf_counter()
        low_wse, low_extent = _low_fidelity_coefficients(
            config,
            event,
            nearest_coarse,
            coarse.elevation,
            fine.elevation,
            wet_indices,
            tf_indices,
            wse_eof,
            extent_eof,
            required_timesteps=max(ref.timestep for ref in requested_refs) + 1,
        )
        prediction_started = time.perf_counter()
        high_wse_coefficients = wse_gp.predict(low_wse)
        high_extent_coefficients = extent_gp.predict(low_extent)
        predicted_wse_all = np.broadcast_to(
            target_elevation, (len(low_wse), len(target_ids))
        ).copy()
        if target_wet.any():
            reconstructed = _reconstruct_subset(
                high_wse_coefficients, wse_eof, wet_position[target_ids[target_wet]]
            )
            predicted_wse_all[:, target_wet] = np.where(
                reconstructed > target_elevation[target_wet] + config.data.dry_threshold_m,
                reconstructed,
                target_elevation[target_wet],
            )
        predicted_extent = np.zeros_like(predicted_wse_all, dtype=bool)
        predicted_extent[:, target_af] = True
        if target_tf.any():
            reconstructed_extent = _reconstruct_subset(
                high_extent_coefficients, extent_eof, tf_position[target_ids[target_tf]]
            )
            predicted_extent[:, target_tf] = reconstructed_extent > 0.5
        predicted_wse_all = np.where(predicted_extent, predicted_wse_all, target_elevation)
        prediction_seconds = time.perf_counter() - prediction_started

        timestep_ids = np.asarray([ref.timestep for ref in requested_refs], dtype=np.int64)
        predicted_wse = predicted_wse_all[timestep_ids]
        predicted_wse, predicted_depth = physical_wse_and_depth(
            predicted_wse, target_elevation, config.data.dry_threshold_m
        )
        tuflow = TuflowEvent(event.tuflow_path, config.data.tuflow_warmup_steps)
        target_wse = np.stack([tuflow.read_wse(int(step), target_ids) for step in timestep_ids])
        target_wse = np.where(
            target_wse > target_elevation + config.data.dry_threshold_m,
            target_wse,
            target_elevation,
        )
        _, target_depth = physical_wse_and_depth(
            target_wse, target_elevation, config.data.dry_threshold_m
        )
        metrics = calculate_event_metrics(
            target_depth, predicted_depth, config.data.dry_threshold_m
        )
        metric_objects.append(metrics)
        rows.append(
            {
                "event_index": event_index,
                "event_name": event.name,
                "group": event.group,
                **metrics.as_dict(),
                "lsg_gp_and_reconstruction_seconds": prediction_seconds,
                "end_to_end_seconds": time.perf_counter() - event_started,
                "n_timesteps": len(timestep_ids),
                "n_target_cells": len(target_ids),
            }
        )
        np.savez_compressed(
            output_dir / f"{event.name}.npz",
            fine_global_ids=target_ids,
            terrain=target_elevation,
            target_wse=target_wse,
            target_depth=target_depth,
            predicted_wse=predicted_wse,
            predicted_depth=predicted_depth,
        )
    metrics_path = output_dir.parent / "event_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = mean_event_metrics(metric_objects)
    safe_metrics = {
        name: value if np.isfinite(value) else None for name, value in summary.as_dict().items()
    }
    payload = {
        "mean_event_metrics": safe_metrics,
        "note": "LSG was trained on the full supplied domain; metrics use AOI target cells only.",
    }
    (output_dir.parent / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics_path
