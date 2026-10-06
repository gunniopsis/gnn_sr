from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeometryData:
    xy: np.ndarray
    elevation: np.ndarray
    area: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> GeometryData:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                xy=np.asarray(archive["XY_coor"], dtype=np.float64),
                elevation=np.asarray(archive["Z_coor"], dtype=np.float32),
                area=np.asarray(archive["Area"], dtype=np.float32),
            )


def load_burnett_geometries(dataset_root: str | Path) -> tuple[GeometryData, GeometryData]:
    root = Path(dataset_root) / "Geometry_data"
    coarse = GeometryData.from_npz(root / "HECRAS_Geometry_data.npz")
    fine = GeometryData.from_npz(root / "Tuflow_Geometry_data.npz")
    return coarse, fine


def build_regular_grid_edges(
    xy: np.ndarray, spacing: float = 20.0, tolerance: float = 0.05
) -> np.ndarray:
    """Build bidirectional face-neighbour links for an axis-aligned regular grid."""
    xy = np.asarray(xy, dtype=np.float64)
    origin = xy.min(axis=0)
    ij_float = (xy - origin) / spacing
    ij = np.rint(ij_float).astype(np.int64)
    residual = np.abs(ij_float - ij) * spacing
    if residual.max(initial=0.0) > tolerance:
        raise ValueError(
            "TUFLOW coordinates do not lie on the configured regular grid: "
            f"maximum residual is {residual.max():.6f} m"
        )
    width = int(ij[:, 0].max()) + 1
    keys = ij[:, 1] * width + ij[:, 0]
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Duplicate TUFLOW cell-centre coordinates were found")
    order = np.argsort(keys)
    sorted_keys = keys[order]

    pairs: list[np.ndarray] = []
    for offset in (1, width):
        candidate = keys + offset
        positions = np.searchsorted(sorted_keys, candidate)
        valid = positions < len(sorted_keys)
        matched = np.zeros(len(keys), dtype=bool)
        matched[valid] = sorted_keys[positions[valid]] == candidate[valid]
        if offset == 1:
            matched &= ij[:, 0] + 1 < width
        source = np.flatnonzero(matched)
        destination = order[positions[matched]]
        pairs.append(np.stack((source, destination), axis=0))

    physical = np.concatenate(pairs, axis=1)
    reverse = physical[::-1]
    return np.concatenate((physical, reverse), axis=1).astype(np.int64, copy=False)


def extract_hec_edges(hdf_path: str | Path, flow_area_name: str) -> np.ndarray:
    """Extract bidirectional cell links from HEC-RAS Faces Cell Indexes."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Extracting HEC-RAS topology requires h5py") from exc
    with h5py.File(hdf_path, "r") as handle:
        geometry = handle[f"Geometry/2D Flow Areas/{flow_area_name}"]
        raw_elevation = np.asarray(geometry["Cells Minimum Elevation"])
        faces = np.asarray(geometry["Faces Cell Indexes"], dtype=np.int64)
    valid_raw = np.flatnonzero(np.isfinite(raw_elevation))
    raw_to_compact = np.full(len(raw_elevation), -1, dtype=np.int64)
    raw_to_compact[valid_raw] = np.arange(len(valid_raw), dtype=np.int64)
    valid_faces = (faces[:, 0] >= 0) & (faces[:, 1] >= 0)
    pairs = raw_to_compact[faces[valid_faces]]
    pairs = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    directed = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    return directed.T.astype(np.int64, copy=False)


def edge_attributes(edge_index: np.ndarray, geometry: GeometryData) -> np.ndarray:
    source, destination = edge_index
    delta_xy = geometry.xy[destination] - geometry.xy[source]
    distance = np.linalg.norm(delta_xy, axis=1)
    delta_z = geometry.elevation[destination] - geometry.elevation[source]
    slope = np.divide(delta_z, distance, out=np.zeros_like(delta_z), where=distance > 0)
    return np.column_stack((delta_xy, distance, delta_z, slope)).astype(np.float32)


def select_aoi_centres(
    xy: np.ndarray, aoi_path: str | Path, model_crs: str = "EPSG:28356"
) -> np.ndarray:
    """Return a centre-inside-or-on-boundary mask for a polygon GIS dataset."""
    try:
        import fiona
        import shapely
        from pyproj import CRS, Transformer
        from shapely.geometry import shape
        from shapely.ops import transform, unary_union
    except ImportError as exc:  # pragma: no cover
        raise ImportError("AOI selection requires fiona, shapely, and pyproj") from exc

    aoi_path = Path(aoi_path)
    with fiona.open(aoi_path) as source:
        geometries = [shape(feature["geometry"]) for feature in source if feature["geometry"]]
        source_crs = source.crs_wkt or source.crs
    if not geometries:
        raise ValueError(f"No polygon geometries found in {aoi_path}")
    polygon = unary_union(geometries)
    if polygon.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"AOI must contain polygons, got {polygon.geom_type}")
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if not source_crs:
        raise ValueError(f"AOI {aoi_path} has no CRS metadata")
    source_reference = CRS.from_user_input(source_crs)
    target_reference = CRS.from_user_input(model_crs)
    if source_reference != target_reference:
        transformer = Transformer.from_crs(source_reference, target_reference, always_xy=True)
        polygon = transform(transformer.transform, polygon)

    points = shapely.points(xy[:, 0], xy[:, 1])
    return np.asarray(shapely.covers(polygon, points), dtype=bool)


def hec_cell_polygons(dataset_root: str | Path) -> tuple[list[object], np.ndarray]:
    """Construct HEC-RAS cell polygons from the supplied face-point indices."""
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:  # pragma: no cover
        raise ImportError("HEC-RAS containment mapping requires shapely") from exc
    path = Path(dataset_root) / "Geometry_data" / "HECRAS_Geometry_data.npz"
    with np.load(path, allow_pickle=True) as archive:
        face_points = np.asarray(archive["Face_XY_coor"], dtype=np.float64)
        cell_vertices = np.asarray(archive["Cell_face_idx"], dtype=np.int64)
        centres = np.asarray(archive["XY_coor"], dtype=np.float64)
    polygons: list[object] = []
    for row in cell_vertices:
        indices = row[row >= 0]
        vertices = face_points[indices]
        polygon = Polygon(vertices)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygons.append(polygon)
    return polygons, centres


def containing_hec_parents(
    fine_xy: np.ndarray, polygons: list[object], coarse_centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map each fine centre to one containing HEC cell, with nearest-centre fallback."""
    try:
        import shapely
        from scipy.spatial import cKDTree
        from shapely.strtree import STRtree
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Cross-scale containment mapping requires shapely and scipy") from exc
    points = shapely.points(fine_xy[:, 0], fine_xy[:, 1])
    tree = STRtree(polygons)
    matches = tree.query(points, predicate="intersects")
    parents = np.full(len(fine_xy), -1, dtype=np.int64)
    if matches.size:
        point_ids, polygon_ids = matches
        order = np.argsort(point_ids, kind="stable")
        point_ids = point_ids[order]
        polygon_ids = polygon_ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(point_ids)) + 1]
        stops = np.r_[starts[1:], len(point_ids)]
        for start, stop in zip(starts, stops, strict=True):
            point_id = int(point_ids[start])
            candidates = polygon_ids[start:stop]
            if len(candidates) == 1:
                parents[point_id] = candidates[0]
            else:
                delta = coarse_centres[candidates] - fine_xy[point_id]
                parents[point_id] = candidates[np.argmin(np.einsum("ij,ij->i", delta, delta))]
    fallback = parents < 0
    if fallback.any():
        nearest = cKDTree(coarse_centres).query(fine_xy[fallback], k=1)[1]
        parents[fallback] = nearest
    return parents, fallback


def load_or_build_topology(
    dataset_root: str | Path,
    cache_dir: str | Path,
    flow_area_name: str,
    spacing: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build immutable full-domain within-scale topology once and cache it."""
    dataset_root = Path(dataset_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fine_path = cache_dir / "tuflow_face_edges.npz"
    coarse_path = cache_dir / "hecras_face_edges.npz"
    coarse_geometry, fine_geometry = load_burnett_geometries(dataset_root)
    if fine_path.exists():
        with np.load(fine_path) as archive:
            fine_edges = np.asarray(archive["edge_index"], dtype=np.int64)
    else:
        fine_edges = build_regular_grid_edges(fine_geometry.xy, spacing, tolerance)
        np.savez(fine_path, edge_index=fine_edges)
    if coarse_path.exists():
        with np.load(coarse_path) as archive:
            coarse_edges = np.asarray(archive["edge_index"], dtype=np.int64)
    else:
        first_hdf = next((dataset_root / "HD_model_data" / "Low-fidelity").glob("*.hdf"))
        coarse_edges = extract_hec_edges(first_hdf, flow_area_name)
        if coarse_edges.max(initial=-1) >= len(coarse_geometry.xy):
            raise ValueError("HEC-RAS topology indices do not match supplied compact geometry")
        np.savez(coarse_path, edge_index=coarse_edges)
    metadata = {
        "fine_nodes": len(fine_geometry.xy),
        "fine_directed_edges": fine_edges.shape[1],
        "coarse_nodes": len(coarse_geometry.xy),
        "coarse_directed_edges": coarse_edges.shape[1],
    }
    (cache_dir / "topology_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return coarse_edges, fine_edges
