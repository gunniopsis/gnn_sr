import numpy as np

from gnnsr.data import physical_wse_and_depth
from gnnsr.metrics import calculate_event_metrics, critical_success_index


def test_physical_postprocessing_enforces_threshold_and_non_negative_depth():
    terrain = np.array([5.0, 5.0, 5.0, 5.0])
    raw = np.array([4.0, 5.0, 5.029, 5.03])
    wse, depth = physical_wse_and_depth(raw, terrain, 0.03)
    np.testing.assert_allclose(depth, [0.0, 0.0, 0.0, 0.03])
    np.testing.assert_allclose(wse, terrain + depth)


def test_perfect_event_metrics():
    depth = np.array([[0.0, 0.1], [0.2, 0.4]])
    metrics = calculate_event_metrics(depth, depth)
    assert metrics.rmse_m == 0.0
    assert metrics.fidelity_index == 1.0
    assert metrics.maximum_depth_r2 == 1.0
    assert metrics.average_peak_difference_m == 0.0
    assert metrics.maximum_extent_csi == 1.0


def test_csi_counts_hits_misses_and_false_alarms():
    true = np.array([1, 1, 0, 0], dtype=bool)
    predicted = np.array([1, 0, 1, 0], dtype=bool)
    assert critical_success_index(true, predicted) == 1 / 3
