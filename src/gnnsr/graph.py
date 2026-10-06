from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .geometry import (
    GeometryData,
    containing_hec_parents,
    edge_attributes,
    hec_cell_polygons,
    load_burnett_geometries,
    load_or_build_topology,
    select_aoi_centres,
)

HEC_RELATION = ("hec", "hydraulic_neighbour", "hec")
CROSS_RELATION = ("hec", "containing_parent", "tuflow")
TUFLOW_RELATION = ("tuflow", "hydraulic_neighbour", "tuflow")


@dataclass(frozen=True)
class GraphArrays:
    coarse_global_ids: np.ndarray
    fine_global_ids: np.ndarray
    target_mask: np.ndarray
    coarse_static: np.ndarray
    fine_static: np.ndarray
    coarse_edge_index: np.ndarray
    coarse_edge_attr: np.ndarray
    cross_edge_index: np.ndarray
    cross_edge_attr: np.ndarray
    fine_edge_index: np.ndarray
    fine_edge_attr: np.ndarray
    fine_parent_local: np.ndarray
    fallback_parent_mask: np.ndarray

    @property
    def target_local_ids(self) -> np.ndarray:
        return np.flatnonzero(self.target_mask)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: str | Path) -> GraphArrays:
        with np.load(path) as archive:
            return cls(**{name: np.asarray(archive[name]) for name in cls.__dataclass_fields__})

    def summary(self) -> dict[str, int]:
        return {
            "coarse_nodes": len(self.coarse_global_ids),
            "fine_context_nodes": len(self.fine_global_ids),
            "fine_target_nodes": int(self.target_mask.sum()),
            "coarse_directed_edges": self.coarse_edge_index.shape[1],
            "cross_directed_edges": self.cross_edge_index.shape[1],
            "fine_directed_edges": self.fine_edge_index.shape[1],
            "nearest_parent_fallbacks": int(self.fallback_parent_mask.sum()),
        }


def expand_halo(seed_mask: np.ndarray, edge_index: np.ndarray, hops: int) -> np.ndarray:
    selected = np.asarray(seed_mask, dtype=bool).copy()
    source, destination = edge_index
    frontier = selected.copy()
    for _ in range(hops):
        next_frontier = np.zeros_like(selected)
        next_frontier[destination[frontier[source]]] = True
        next_frontier &= ~selected
        selected |= next_frontier
        frontier = next_frontier
        if not frontier.any():
            break
    return selected


def induced_edges(
    edge_index: np.ndarray, selected_global_ids: np.ndarray, n_global: int
) -> np.ndarray:
    global_to_local = np.full(n_global, -1, dtype=np.int64)
    global_to_local[selected_global_ids] = np.arange(len(selected_global_ids), dtype=np.int64)
    source_local = global_to_local[edge_index[0]]
    destination_local = global_to_local[edge_index[1]]
    keep = (source_local >= 0) & (destination_local >= 0)
    return np.stack((source_local[keep], destination_local[keep]), axis=0)


def _local_geometry(geometry: GeometryData, ids: np.ndarray) -> GeometryData:
    return GeometryData(geometry.xy[ids], geometry.elevation[ids], geometry.area[ids])


def _static_features(geometry: GeometryData) -> np.ndarray:
    if np.any(geometry.area <= 0):
        raise ValueError("Cell areas must be positive before log transformation")
    return np.column_stack((geometry.elevation, np.log(geometry.area))).astype(np.float32)


def _cross_attributes(
    edge_index: np.ndarray, coarse_geometry: GeometryData, fine_geometry: GeometryData
) -> np.ndarray:
    source, destination = edge_index
    delta_xy = fine_geometry.xy[destination] - coarse_geometry.xy[source]
    distance = np.linalg.norm(delta_xy, axis=1)
    delta_z = fine_geometry.elevation[destination] - coarse_geometry.elevation[source]
    slope = np.divide(delta_z, distance, out=np.zeros_like(delta_z), where=distance > 0)
    return np.column_stack((delta_xy, distance, delta_z, slope)).astype(np.float32)


def graph_cache_path(config: ExperimentConfig) -> Path:
    aoi = config.paths.aoi_path
    assert aoi is not None
    identity = {
        "aoi": str(aoi.resolve()),
        "aoi_mtime_ns": aoi.stat().st_mtime_ns,
        "fine_hops": config.graph.fine_halo_hops,
        "coarse_hops": config.graph.coarse_halo_hops,
        "spacing": config.graph.fine_grid_spacing_m,
        "model_crs": config.data.model_crs,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return config.paths.cache_dir / f"aoi_graph_{digest}.npz"


def prepare_aoi_graph(config: ExperimentConfig, force: bool = False) -> tuple[GraphArrays, Path]:
    cache_path = graph_cache_path(config)
    if cache_path.exists() and not force:
        return GraphArrays.load(cache_path), cache_path

    coarse_geometry, fine_geometry = load_burnett_geometries(config.paths.dataset_root)
    coarse_full_edges, fine_full_edges = load_or_build_topology(
        config.paths.dataset_root,
        config.paths.cache_dir,
        config.data.flow_area_name,
        config.graph.fine_grid_spacing_m,
        config.graph.fine_spacing_tolerance_m,
    )
    target_global_mask = select_aoi_centres(
        fine_geometry.xy, config.paths.aoi_path, config.data.model_crs
    )
    if not target_global_mask.any():
        raise ValueError("The AOI contains no TUFLOW cell centres")
    fine_selected_mask = expand_halo(
        target_global_mask, fine_full_edges, config.graph.fine_halo_hops
    )
    fine_global_ids = np.flatnonzero(fine_selected_mask)

    polygons, coarse_centres = hec_cell_polygons(config.paths.dataset_root)
    parent_global, fallback_mask = containing_hec_parents(
        fine_geometry.xy[fine_global_ids], polygons, coarse_centres
    )
    coarse_seed_mask = np.zeros(len(coarse_geometry.xy), dtype=bool)
    coarse_seed_mask[np.unique(parent_global)] = True
    coarse_selected_mask = expand_halo(
        coarse_seed_mask, coarse_full_edges, config.graph.coarse_halo_hops
    )
    coarse_global_ids = np.flatnonzero(coarse_selected_mask)

    coarse_global_to_local = np.full(len(coarse_geometry.xy), -1, dtype=np.int64)
    coarse_global_to_local[coarse_global_ids] = np.arange(len(coarse_global_ids))
    fine_parent_local = coarse_global_to_local[parent_global]
    if np.any(fine_parent_local < 0):
        raise RuntimeError("A cross-scale parent was omitted from the coarse context graph")

    coarse_edges = induced_edges(coarse_full_edges, coarse_global_ids, len(coarse_geometry.xy))
    fine_edges = induced_edges(fine_full_edges, fine_global_ids, len(fine_geometry.xy))
    cross_edges = np.stack(
        (fine_parent_local, np.arange(len(fine_global_ids), dtype=np.int64)), axis=0
    )
    coarse_local_geometry = _local_geometry(coarse_geometry, coarse_global_ids)
    fine_local_geometry = _local_geometry(fine_geometry, fine_global_ids)
    graph = GraphArrays(
        coarse_global_ids=coarse_global_ids,
        fine_global_ids=fine_global_ids,
        target_mask=target_global_mask[fine_global_ids],
        coarse_static=_static_features(coarse_local_geometry),
        fine_static=_static_features(fine_local_geometry),
        coarse_edge_index=coarse_edges,
        coarse_edge_attr=edge_attributes(coarse_edges, coarse_local_geometry),
        cross_edge_index=cross_edges,
        cross_edge_attr=_cross_attributes(cross_edges, coarse_local_geometry, fine_local_geometry),
        fine_edge_index=fine_edges,
        fine_edge_attr=edge_attributes(fine_edges, fine_local_geometry),
        fine_parent_local=fine_parent_local,
        fallback_parent_mask=fallback_mask,
    )
    graph.save(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(graph.summary(), indent=2), encoding="utf-8")
    return graph, cache_path


def to_heterodata(graph: GraphArrays, normalized_edges: dict[str, np.ndarray] | None = None):
    """Create the agreed PyG HeteroData relation structure from cached arrays."""
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Creating HeteroData requires torch and torch-geometric") from exc
    edges = normalized_edges or {
        "coarse": graph.coarse_edge_attr,
        "cross": graph.cross_edge_attr,
        "fine": graph.fine_edge_attr,
    }
    data = HeteroData()
    data["hec"].num_nodes = len(graph.coarse_global_ids)
    data["tuflow"].num_nodes = len(graph.fine_global_ids)
    data["tuflow"].target_mask = torch.from_numpy(graph.target_mask)
    data[HEC_RELATION].edge_index = torch.from_numpy(graph.coarse_edge_index).long()
    data[HEC_RELATION].edge_attr = torch.from_numpy(edges["coarse"]).float()
    data[CROSS_RELATION].edge_index = torch.from_numpy(graph.cross_edge_index).long()
    data[CROSS_RELATION].edge_attr = torch.from_numpy(edges["cross"]).float()
    data[TUFLOW_RELATION].edge_index = torch.from_numpy(graph.fine_edge_index).long()
    data[TUFLOW_RELATION].edge_attr = torch.from_numpy(edges["fine"]).float()
    return data
