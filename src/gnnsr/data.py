from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from .config import ExperimentConfig
from .geometry import load_burnett_geometries
from .graph import GraphArrays
from .io import EventCatalog, HecEvent, TuflowEvent, flatten_event_timesteps


@dataclass(frozen=True)
class SampleRef:
    event_index: int
    timestep: int


@dataclass(frozen=True)
class FoldSamples:
    train: tuple[SampleRef, ...]
    validation: tuple[SampleRef, ...]
    test: tuple[SampleRef, ...]


def load_fold_samples(config: ExperimentConfig, catalog: EventCatalog) -> FoldSamples:
    fold = config.training.fold
    training_events = catalog.training_events(fold)
    pooled = [
        SampleRef(*item)
        for item in flatten_event_timesteps(training_events, config.data.tuflow_warmup_steps)
    ]
    split_path = (
        config.paths.dataset_root
        / "Train_test_split_data"
        / f"Train_test_split_ValidateOnGrp_{fold}.npz"
    )
    with np.load(split_path) as archive:
        train_indices = np.asarray(archive["idx_train"], dtype=np.int64)
        validation_indices = np.asarray(archive["idx_test"], dtype=np.int64)
    if max(train_indices.max(), validation_indices.max()) >= len(pooled):
        raise ValueError(f"Saved fold split expects more timesteps than were found ({len(pooled)})")
    test = [
        SampleRef(*item)
        for item in flatten_event_timesteps(
            catalog.test_events(fold), config.data.tuflow_warmup_steps
        )
    ]
    if config.training.debug_max_test_samples is not None:
        test = test[: config.training.debug_max_test_samples]
    train = tuple(pooled[index] for index in train_indices)
    validation = tuple(pooled[index] for index in validation_indices)
    if config.training.debug_max_train_samples is not None:
        train = train[: config.training.debug_max_train_samples]
    if config.training.debug_max_validation_samples is not None:
        validation = validation[: config.training.debug_max_validation_samples]
    return FoldSamples(
        train=train,
        validation=validation,
        test=tuple(test),
    )


def grouped_shuffled_refs(refs: Sequence[SampleRef], seed: int) -> list[SampleRef]:
    """Shuffle events and timesteps while retaining disk-friendly event blocks."""
    rng = random.Random(seed)
    grouped: dict[int, list[SampleRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.event_index, []).append(ref)
    event_ids = list(grouped)
    rng.shuffle(event_ids)
    result: list[SampleRef] = []
    for event_id in event_ids:
        rng.shuffle(grouped[event_id])
        result.extend(grouped[event_id])
    return result


@dataclass(frozen=True)
class FloodSample:
    hec_features: np.ndarray
    fine_features: np.ndarray
    target_residual: np.ndarray
    target_wse: np.ndarray
    target_depth: np.ndarray
    baseline_wse: np.ndarray
    event_index: int
    timestep: int


def physical_wse_and_depth(
    raw_wse: np.ndarray, terrain: np.ndarray, dry_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    raw_depth = raw_wse - terrain
    depth = np.where(raw_depth >= dry_threshold, raw_depth, 0.0)
    return terrain + depth, depth


class EventDataAccess:
    """Read one timestep at a time and keep only the current event open."""

    def __init__(self, config: ExperimentConfig, graph: GraphArrays, catalog: EventCatalog):
        self.config = config
        self.graph = graph
        self.catalog = catalog
        coarse_geometry, fine_geometry = load_burnett_geometries(config.paths.dataset_root)
        self.coarse_elevation = coarse_geometry.elevation[graph.coarse_global_ids]
        self.fine_elevation = fine_geometry.elevation[graph.fine_global_ids]
        self.target_local = graph.target_local_ids
        self.target_global = graph.fine_global_ids[self.target_local]
        self.target_elevation = self.fine_elevation[self.target_local]
        self._current_event_id: int | None = None
        self._tuflow: TuflowEvent | None = None
        self._hec: HecEvent | None = None

    def _open(self, event_index: int) -> tuple[TuflowEvent, HecEvent]:
        if self._current_event_id == event_index:
            assert self._tuflow is not None and self._hec is not None
            return self._tuflow, self._hec
        self.close()
        event = self.catalog.records[event_index]
        self._tuflow = TuflowEvent(event.tuflow_path, self.config.data.tuflow_warmup_steps)
        self._hec = HecEvent(
            event.hec_path,
            self.config.data.flow_area_name,
            self.graph.coarse_global_ids,
            self.config.data.hec_warmup_hours,
        )
        if self._hec.n_timesteps < self._tuflow.n_timesteps:
            raise ValueError(
                f"HEC-RAS event {event.name} has {self._hec.n_timesteps} post-warmup steps "
                f"but TUFLOW has {self._tuflow.n_timesteps}"
            )
        self._current_event_id = event_index
        return self._tuflow, self._hec

    def sample(self, ref: SampleRef) -> FloodSample:
        tuflow, hec = self._open(ref.event_index)
        threshold = self.config.data.dry_threshold_m
        raw_hec = np.nan_to_num(hec.read_wse(ref.timestep), nan=-999.0)
        hec_wse = np.where(
            raw_hec > self.coarse_elevation + threshold,
            raw_hec,
            self.coarse_elevation,
        ).astype(np.float32)
        hec_depth = np.maximum(hec_wse - self.coarse_elevation, 0.0)
        baseline_wse = hec_wse[self.graph.fine_parent_local]
        baseline_depth = np.maximum(baseline_wse - self.fine_elevation, 0.0)

        raw_target = np.nan_to_num(tuflow.read_wse(ref.timestep, self.target_global), nan=-999.0)
        target_wse = np.where(
            raw_target > self.target_elevation + threshold,
            raw_target,
            self.target_elevation,
        ).astype(np.float32)
        target_depth = np.maximum(target_wse - self.target_elevation, 0.0)
        target_residual = target_wse - baseline_wse[self.target_local]
        hec_features = np.column_stack(
            (
                self.graph.coarse_static,
                hec_wse,
                hec_depth,
            )
        ).astype(np.float32)
        fine_features = np.column_stack(
            (
                self.graph.fine_static,
                baseline_wse,
                baseline_depth,
            )
        ).astype(np.float32)
        return FloodSample(
            hec_features=hec_features,
            fine_features=fine_features,
            target_residual=target_residual.astype(np.float32),
            target_wse=target_wse,
            target_depth=target_depth.astype(np.float32),
            baseline_wse=baseline_wse.astype(np.float32),
            event_index=ref.event_index,
            timestep=ref.timestep,
        )

    def close(self) -> None:
        if self._hec is not None:
            self._hec.close()
        self._hec = None
        self._tuflow = None
        self._current_event_id = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


class RunningMoments:
    def __init__(self, n_features: int):
        self.count = 0
        self.sum = np.zeros(n_features, dtype=np.float64)
        self.sum_squares = np.zeros(n_features, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, len(self.sum))
        if not np.isfinite(values).all():
            raise ValueError("Non-finite value encountered while fitting normalization statistics")
        self.count += len(values)
        self.sum += values.sum(axis=0)
        self.sum_squares += np.square(values).sum(axis=0)

    def finalize(self) -> Standardizer:
        if self.count == 0:
            raise ValueError("Cannot finalize empty normalization statistics")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_squares / self.count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-8] = 1.0
        return Standardizer(mean.astype(np.float32), scale.astype(np.float32))


@dataclass(frozen=True)
class NormalizationBundle:
    hec_nodes: Standardizer
    fine_nodes: Standardizer
    target_residual: Standardizer
    coarse_edges: Standardizer
    cross_edges: Standardizer
    fine_edges: Standardizer

    def save(self, path: str | Path) -> None:
        values: dict[str, list[float]] = {}
        for name in self.__dataclass_fields__:
            scaler = getattr(self, name)
            values[f"{name}_mean"] = scaler.mean.tolist()
            values[f"{name}_scale"] = scaler.scale.tolist()
        Path(path).write_text(json.dumps(values, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NormalizationBundle:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        kwargs = {}
        for name in cls.__dataclass_fields__:
            kwargs[name] = Standardizer(
                np.asarray(values[f"{name}_mean"], dtype=np.float32),
                np.asarray(values[f"{name}_scale"], dtype=np.float32),
            )
        return cls(**kwargs)

    def normalized_edges(self, graph: GraphArrays) -> dict[str, np.ndarray]:
        return {
            "coarse": self.coarse_edges.transform(graph.coarse_edge_attr),
            "cross": self.cross_edges.transform(graph.cross_edge_attr),
            "fine": self.fine_edges.transform(graph.fine_edge_attr),
        }


def fit_normalization(
    config: ExperimentConfig,
    graph: GraphArrays,
    catalog: EventCatalog,
    train_refs: Sequence[SampleRef],
) -> NormalizationBundle:
    hec_moments = RunningMoments(4)
    fine_moments = RunningMoments(4)
    target_moments = RunningMoments(1)
    ordered = grouped_shuffled_refs(train_refs, config.training.seed)
    with EventDataAccess(config, graph, catalog) as access:
        for ref in ordered:
            sample = access.sample(ref)
            hec_moments.update(sample.hec_features)
            fine_moments.update(sample.fine_features)
            target_moments.update(sample.target_residual[:, None])
    edge_scalers = []
    for values in (graph.coarse_edge_attr, graph.cross_edge_attr, graph.fine_edge_attr):
        moments = RunningMoments(5)
        moments.update(values)
        edge_scalers.append(moments.finalize())
    return NormalizationBundle(
        hec_nodes=hec_moments.finalize(),
        fine_nodes=fine_moments.finalize(),
        target_residual=target_moments.finalize(),
        coarse_edges=edge_scalers[0],
        cross_edges=edge_scalers[1],
        fine_edges=edge_scalers[2],
    )
