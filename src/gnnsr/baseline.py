from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import EventDataAccess, SampleRef, load_fold_samples, physical_wse_and_depth
from .geometry import load_burnett_geometries
from .graph import prepare_aoi_graph
from .io import EventCatalog
from .metrics import calculate_event_metrics, mean_event_metrics


def project_parent_wse_to_fine_depth(
    coarse_wse: np.ndarray,
    fine_parent_local: np.ndarray,
    fine_elevation: np.ndarray,
    dry_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each fine cell its parent HEC-RAS WSE, then derive physical depth."""
    coarse_wse = np.asarray(coarse_wse)
    fine_parent_local = np.asarray(fine_parent_local, dtype=np.int64)
    fine_elevation = np.asarray(fine_elevation)
    if fine_parent_local.shape != fine_elevation.shape:
        raise ValueError("Parent indices and fine elevations must have equal shape")
    if np.any((fine_parent_local < 0) | (fine_parent_local >= len(coarse_wse))):
        raise ValueError("Fine-to-coarse parent index is outside the HEC-RAS array")
    inherited_wse = coarse_wse[fine_parent_local]
    return physical_wse_and_depth(inherited_wse, fine_elevation, dry_threshold_m)


def evaluate_hec_ras_aoi(config: ExperimentConfig) -> Path:
    """Evaluate parent-cell-projected HEC-RAS results against TUFLOW in the AOI."""
    graph, _ = prepare_aoi_graph(config)
    catalog = EventCatalog(config.paths.dataset_root)
    splits = load_fold_samples(config, catalog)
    fold = config.training.fold
    _, fine_geometry = load_burnett_geometries(config.paths.dataset_root)
    target_local = graph.target_local_ids
    target_ids = graph.fine_global_ids[target_local]
    target_elevation = fine_geometry.elevation[target_ids]
    target_parents = graph.fine_parent_local[target_local]

    grouped: dict[int, list[SampleRef]] = {}
    for ref in splits.test:
        grouped.setdefault(ref.event_index, []).append(ref)

    rows: list[dict[str, float | int | str]] = []
    metric_objects = []
    with EventDataAccess(config, graph, catalog) as access:
        for event_index, refs in grouped.items():
            refs.sort(key=lambda ref: ref.timestep)
            projected_depths = []
            target_depths = []
            for ref in refs:
                sample = access.sample(ref)
                _, projected_depth = project_parent_wse_to_fine_depth(
                    sample.hec_features[:, 2],
                    target_parents,
                    target_elevation,
                    config.data.dry_threshold_m,
                )
                projected_depths.append(projected_depth)
                target_depths.append(sample.target_depth)

            projected_depth = np.stack(projected_depths)
            target_depth = np.stack(target_depths)
            metrics = calculate_event_metrics(
                target_depth, projected_depth, config.data.dry_threshold_m
            )
            metric_objects.append(metrics)
            event = catalog.records[event_index]
            rows.append(
                {
                    "event_index": event_index,
                    "event_name": event.name,
                    "group": event.group,
                    **metrics.as_dict(),
                    "n_timesteps": len(refs),
                    "n_target_cells": len(target_ids),
                }
            )

    output_dir = config.paths.output_dir / "hec_ras" / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "event_metrics.csv"
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
        "events": len(rows),
        "timesteps_per_event": sorted({int(row["n_timesteps"]) for row in rows}),
        "target_cells": len(target_ids),
        "mapping_method": (
            "Each AOI TUFLOW cell inherits the WSE of its containing parent HEC-RAS cell; "
            "depth is then calculated against the TUFLOW cell elevation and filtered with "
            f"the {config.data.dry_threshold_m:.2f} m dry threshold."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics_path
