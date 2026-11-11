"""Scientific checks for common-mask scoring and mapping semantics."""
import sys
import csv
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_full_domain_lsg import classification, peak_maps, wet_positions
from gnnsr.metrics import calculate_event_metrics
import compare_full_domain_lsg as comparison
from full_domain_lf_export import project_step, export_event
from plot_cell_hydrograph import load_hydrograph


def test_mask_joins_by_global_id_and_omits_large_outside_error():
    ids = np.array([30, 10, 20, 40])
    positions = wet_positions(ids, np.array([20, 10]))
    np.testing.assert_array_equal(positions, [2, 1])
    reference = np.array([[1., 0., 1., 0.], [2., 0., 2., 0.]])
    predicted = reference.copy()
    predicted[:, [0, 3]] += 100
    matched = calculate_event_metrics(reference[:, positions], predicted[:, positions])
    assert matched.rmse_m == 0
    assert matched.maximum_extent_csi == 1
    assert calculate_event_metrics(reference, predicted).rmse_m > 0


def test_missing_and_duplicate_ids_fail():
    with pytest.raises(ValueError, match="cover"):
        wet_positions([1, 3], [2])
    with pytest.raises(ValueError, match="cover"):
        wet_positions([1, 3], [4])
    with pytest.raises(ValueError, match="Duplicate"):
        wet_positions([1, 1], [1])
    with pytest.raises(ValueError, match="Duplicate"):
        wet_positions([1, 2], [1, 1])


def test_classification_threshold_and_peak_vs_timestep():
    reference = np.array([0., .03, 0., .03])
    predicted = np.array([0., .03, .03, 0.])
    np.testing.assert_array_equal(classification(reference, predicted), [0, 1, 2, 3])
    # A peak occurring at different times is a hit for maximum extent.
    reference = np.array([[1., 0.], [0., 0.]])
    predicted = np.array([[0., 0.], [.5, 0.]])
    maps = peak_maps(reference, predicted, np.array([10., 20.]))
    np.testing.assert_array_equal(maps["maximum_extent_class"], [1, 0])
    np.testing.assert_array_equal(maps["peak_depth_difference_m"], [.5, 0.])
    np.testing.assert_array_equal(maps["predicted_peak_wse_m"], [10.5, 20.])
    assert classification(reference[0], predicted[0])[0] == 3
    assert classification(reference[1], predicted[1])[0] == 2


def test_peak_map_counts_reproduce_csi_and_bias():
    reference = np.array([[1., 0., .1, 0.], [.5, 0., .2, 0.]])
    predicted = np.array([[.8, .2, 0., 0.], [.5, .1, 0., 0.]])
    maps = peak_maps(reference, predicted, np.zeros(4))
    _, tp, fp, fn = np.bincount(maps["maximum_extent_class"], minlength=4)
    metrics = calculate_event_metrics(reference, predicted)
    assert metrics.maximum_extent_csi == tp / (tp + fp + fn)
    assert metrics.average_peak_difference_m == pytest.approx(maps["peak_depth_difference_m"].mean())


def test_completed_evaluation_rescore_and_provenance(tmp_path, monkeypatch):
    for key, value in {"EVENT_COUNT": 2, "FULL_CELL_COUNT": 4,
                       "WET_CELL_COUNT": 2, "TIMESTEP_COUNT": 4}.items():
        monkeypatch.setattr(comparison, key, value)
    root = tmp_path
    run = root / comparison.RUN
    (run / "predictions").mkdir(parents=True)
    dataset = root / "reference_files/BurnettRV"
    for folder in ("Geometry_data", "HF_EOF_analysis", "Result_data", "HD_model_data/High-fidelity"):
        (dataset / folder).mkdir(parents=True)
    for name in ("scripts/compare_full_domain_lsg.py", "scripts/full_domain_lf_export.py", "src/gnnsr/metrics.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((comparison.ROOT / name).read_bytes())
    comparison.write_csv(dataset / "BurnettRV_event_summary.csv", [
        {"tuflow_evt_name_old": "a", "Group": 1},
        {"tuflow_evt_name_old": "training_only", "Group": 2},
        {"tuflow_evt_name_old": "b", "Group": 1}])
    np.savez(dataset / "Geometry_data/Tuflow_Geometry_data.npz",
             XY_coor=np.zeros((4, 2)), Z_coor=np.zeros(4, dtype=np.float32), Area=np.ones(4))
    np.savez(dataset / "HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_1.npz", wet_idx=[2, 0])
    np.savez(dataset / "Result_data/Validation_results.npz", n_timesteps=np.full(3, 4),
             **{name: np.array([[.1], [999.], [.2]]) for name in comparison.METRICS.values()})
    comparison.write_json(run / "training_status.json", {"state": "complete"})
    comparison.write_csv(run / "history.csv", [{"epoch": 1, "validation_depth_rmse_m": .2},
                                               {"epoch": 2, "validation_depth_rmse_m": .1}])
    (run / "best.pt").write_bytes(b"checkpoint identity")
    comparison.write_csv(run / "test_event_metrics.csv", [
        {"event_name": name, "model_forward_seconds": 5, "end_to_end_seconds": 10} for name in ("a", "b")])
    comparison.write_json(run / "test_summary.json", {"events": 2})
    ids = np.array([3, 1, 0, 2])
    reference = np.tile([0., 0., 1., 0.], (4, 1)).astype(np.float32)
    predicted = reference.copy()
    predicted[:, 0] = 100  # outside the benchmark mask
    for name in ("a", "b"):
        np.savez_compressed(run / "predictions" / f"{name}.npz", fine_global_ids=ids, terrain=np.zeros(4, dtype=np.float32),
                            target_depth=reference, predicted_depth=predicted,
                            target_wse=reference, predicted_wse=predicted, raw_predicted_wse=predicted)
        np.savez(dataset / "HD_model_data/High-fidelity" / f"Paradise_{name}_002.npz", time_data=np.arange(52))
    artifacts = ["test_event_metrics.csv", "test_summary.json", "predictions/a.npz", "predictions/b.npz"]
    comparison.write_json(run / "full_domain_evaluation.json", {
        "checkpoint_epoch": 2, "checkpoint_sha256": comparison.sha256(run / "best.pt"),
        "artifacts": {name: comparison.sha256(run / name) for name in artifacts}})
    source = dataset / "fake_lf_source.hdf"
    source.write_bytes(b"LF source")
    monkeypatch.setattr(comparison, "load_context", lambda root: {"source_paths": []})
    def fake_export(context, event, ids, terrain, times, path):
        path.parent.mkdir(exist_ok=True)
        np.savez(path, lf_depth=reference)
        return reference.copy(), source
    monkeypatch.setattr(comparison, "export_event", fake_export)
    output = comparison.compare(root)
    with (output / "event_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert [float(row["rmse_m"]) for row in rows] == [0., .1, 0., 0., .2, 0.]
    assert all(int(row["n_target_cells"]) == 2 for row in rows)
    assert int(rows[0]["TP"]) == 1 and int(rows[0]["TN"]) == 1
    with np.load(output / "a_peak_maps.npz") as maps:
        np.testing.assert_array_equal(maps["source_timestep_index"], [48, 49, 50, 51])
        np.testing.assert_array_equal(maps["time"], [48, 49, 50, 51])
        assert maps["peak_depth_difference_m"][0] == -100
    assert comparison.compare(root) == output
    (run / "predictions/a.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Prediction changed"):
        comparison.compare(root)


def test_lf_coarse_and_fine_terrain_processing():
    baseline, wse, depth = project_step(np.array([np.nan, 10.02, 12.]),
                                        np.array([5., 10., 10.]), np.array([0, 1, 2, 2]),
                                        np.array([6., 9., 13., 11.]))
    np.testing.assert_array_equal(baseline, [5., 10., 12., 12.])
    np.testing.assert_array_equal(wse, [6., 10., 13., 12.])
    np.testing.assert_array_equal(depth, [0., 1., 0., 1.])


def test_lf_export_uses_cached_parent_ids_and_time_alignment(tmp_path):
    from types import SimpleNamespace
    import h5py
    dataset = tmp_path / "data"
    source = dataset / "HD_model_data/Low-fidelity/BurnettRV_LFmodelB.p01.hdf"
    source.parent.mkdir(parents=True)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("Geometry/2D Flow Areas/test/Cells Minimum Elevation", data=[1., np.nan, 2.])
        base = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
        handle.create_dataset(base + "Time Date Stamp", data=np.array([
            b"01Jan2000 00:00:00", b"03Jan2000 00:00:00", b"03Jan2000 01:00:00"]))
        handle.create_dataset(base + "2D Flow Areas/test/Water Surface", data=[[1., 0., 2.], [3., 0., 4.], [5., 0., 6.]])
    config = SimpleNamespace(paths=SimpleNamespace(dataset_root=dataset), data=SimpleNamespace(
        flow_area_name="test", hec_warmup_hours=47.5, dry_threshold_m=.03))
    context = {"config": config, "fine_ids": np.array([10, 20]), "coarse_ids": np.array([0, 1]),
               "parents": np.array([1, 0]), "coarse_elevation": np.array([1., 2.]), "fallback": np.array([False, True])}
    output = tmp_path / "lf.npz"
    depth, _ = export_event(context, {"HEC_RAS_plan": "p01"}, np.array([20, 10]),
                            np.array([1., 2.]), np.array([48., 49.]), output)
    np.testing.assert_array_equal(depth, [[2., 2.], [4., 4.]])
    with np.load(output) as archive:
        np.testing.assert_array_equal(archive["lf_baseline_wse"], [[3., 4.], [5., 6.]])
        np.testing.assert_array_equal(archive["coarse_parent_global_ids"], [0, 1])
    with pytest.raises(ValueError, match="times do not align"):
        export_event(context, {"HEC_RAS_plan": "p01"}, np.array([20, 10]),
                     np.array([1., 2.]), np.array([49., 50.]), output)


def test_hydrograph_cell_selection_independent_of_column_order(tmp_path):
    (tmp_path / "predictions").mkdir()
    lf = tmp_path / "lsg_wet_mask_comparison/lf_predictions"
    lf.mkdir(parents=True)
    values = np.array([[10., 20.], [11., 21.]])
    np.savez(tmp_path / "predictions/event.npz", fine_global_ids=[8, 9], terrain=[1., 2.],
             target_wse=values, predicted_wse=values+1, target_depth=values-1, predicted_depth=values)
    np.savez(lf / "event.npz", fine_global_ids=[9, 8], terrain=[2., 1.], time_hours=[48, 49],
             lf_wse=values[:, ::-1]+2, lf_depth=values[:, ::-1]+1, lf_baseline_wse=values[:, ::-1]+3)
    series, terrain = load_hydrograph(tmp_path, "event", 8)
    assert terrain == 1.
    np.testing.assert_array_equal(series["HF_WSE_m"], [10, 11])
    np.testing.assert_array_equal(series["LF_WSE_m"], [12, 13])
    with pytest.raises(ValueError, match="exactly one"):
        load_hydrograph(tmp_path, "event", 999)
