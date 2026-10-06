from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EventMetrics:
    rmse_m: float
    fidelity_index: float
    maximum_depth_r2: float
    average_peak_difference_m: float
    maximum_extent_csi: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def rmse(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - true))))


def fidelity_index(
    true_depth: np.ndarray,
    predicted_depth: np.ndarray,
    depth_tolerance_m: float = 0.05,
    time_tolerance_fraction: float = 0.05,
) -> float:
    """Replicate the LSG fidelity index using the closest time-shifted target."""
    shifts = round(time_tolerance_fraction * len(true_depth))
    if shifts <= 0 or 2 * shifts >= len(true_depth):
        return float(np.mean(np.abs(predicted_depth - true_depth) < depth_tolerance_m))
    core = slice(shifts, -shifts)
    minimum_difference = np.full_like(true_depth[core], np.inf, dtype=np.float64)
    for shift in range(-shifts, shifts):
        difference = np.abs(predicted_depth[core] - np.roll(true_depth, shift, axis=0)[core])
        minimum_difference = np.minimum(minimum_difference, difference)
    return float(np.mean(minimum_difference < depth_tolerance_m))


def coefficient_of_determination(true: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum(np.square(true - predicted)))
    total = float(np.sum(np.square(true - np.mean(true))))
    return float("nan") if total == 0 else 1.0 - residual / total


def critical_success_index(true_extent: np.ndarray, predicted_extent: np.ndarray) -> float:
    true_extent = np.asarray(true_extent, dtype=bool)
    predicted_extent = np.asarray(predicted_extent, dtype=bool)
    hits = np.count_nonzero(true_extent & predicted_extent)
    misses = np.count_nonzero(true_extent & ~predicted_extent)
    false_alarms = np.count_nonzero(~true_extent & predicted_extent)
    denominator = hits + misses + false_alarms
    return float("nan") if denominator == 0 else hits / denominator


def calculate_event_metrics(
    true_depth: np.ndarray,
    predicted_depth: np.ndarray,
    dry_threshold_m: float = 0.03,
) -> EventMetrics:
    if true_depth.shape != predicted_depth.shape:
        raise ValueError(
            f"Metric arrays must have equal shape, got {true_depth.shape} and {predicted_depth.shape}"
        )
    true_peak = np.max(true_depth, axis=0)
    predicted_peak = np.max(predicted_depth, axis=0)
    true_extent = true_peak >= dry_threshold_m
    predicted_extent = predicted_peak >= dry_threshold_m
    return EventMetrics(
        rmse_m=rmse(true_depth, predicted_depth),
        fidelity_index=fidelity_index(true_depth, predicted_depth),
        maximum_depth_r2=coefficient_of_determination(true_peak, predicted_peak),
        average_peak_difference_m=float(np.mean(true_peak - predicted_peak)),
        maximum_extent_csi=critical_success_index(true_extent, predicted_extent),
    )


def mean_event_metrics(metrics: list[EventMetrics]) -> EventMetrics:
    if not metrics:
        raise ValueError("No event metrics supplied")
    values = {}
    for name in EventMetrics.__dataclass_fields__:
        candidates = np.asarray([getattr(item, name) for item in metrics], dtype=float)
        finite = candidates[np.isfinite(candidates)]
        values[name] = float(np.mean(finite)) if len(finite) else float("nan")
    return EventMetrics(**values)
