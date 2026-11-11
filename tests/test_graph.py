import numpy as np

from gnnsr.geometry import GeometryData, build_regular_grid_edges, edge_attributes
from gnnsr.graph import expand_halo, induced_edges


def test_regular_grid_edges_are_face_neighbours_and_bidirectional():
    xy = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
    edges = build_regular_grid_edges(xy, spacing=20)
    assert edges.shape == (2, 8)
    links = set(map(tuple, edges.T.tolist()))
    assert links == {
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
    }


def test_halo_and_induced_edges():
    xy = np.array([[0, 0], [20, 0], [40, 0], [60, 0]], dtype=float)
    edges = build_regular_grid_edges(xy, spacing=20)
    seed = np.array([True, False, False, False])
    selected = expand_halo(seed, edges, hops=2)
    np.testing.assert_array_equal(selected, [True, True, True, False])
    local = induced_edges(edges, np.flatnonzero(selected), len(xy))
    assert set(map(tuple, local.T.tolist())) == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_signed_edge_attributes_flip_on_reverse_edges():
    geometry = GeometryData(
        xy=np.array([[0, 0], [20, 0]], dtype=float),
        elevation=np.array([1, 3], dtype=np.float32),
        area=np.array([400, 400], dtype=np.float32),
    )
    edges = np.array([[0, 1], [1, 0]])
    attrs = edge_attributes(edges, geometry)
    np.testing.assert_allclose(attrs[0], [20, 0, 20, 2, 0.1])
    np.testing.assert_allclose(attrs[1], [-20, 0, 20, -2, -0.1])
