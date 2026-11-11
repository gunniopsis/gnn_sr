"""Read LF results through the deployed GNN parent mapping, without rebuilding it."""
from pathlib import Path

import numpy as np

from gnnsr.config import ExperimentConfig
from gnnsr.geometry import GeometryData
from gnnsr.graph import graph_cache_path
from gnnsr.io import HecEvent, _decode_hec_times

CONFIG = Path("experiments/v2/full_domain/configs/c2_f2.toml")


def load_context(root):
    config = ExperimentConfig.from_toml(root / CONFIG)
    cache = graph_cache_path(config)
    if not cache.is_file():
        raise FileNotFoundError(f"Keep the training graph cache for exact LF mapping: {cache}")
    coarse_path = config.paths.dataset_root / "Geometry_data/HECRAS_Geometry_data.npz"
    coarse = GeometryData.from_npz(coarse_path)
    with np.load(cache) as archive:
        fine_ids = archive["fine_global_ids"]
        coarse_ids = archive["coarse_global_ids"]
        parents = archive["fine_parent_local"]
        if not archive["target_mask"].all():
            raise ValueError("Expected the full-domain training graph")
        fallback = archive["fallback_parent_mask"]
    if len(np.unique(fine_ids)) != len(fine_ids) or np.any((parents < 0) | (parents >= len(coarse_ids))):
        raise ValueError("Invalid training graph parent mapping")
    return {"config": config, "fine_ids": fine_ids, "coarse_ids": coarse_ids,
            "parents": parents, "coarse_elevation": coarse.elevation[coarse_ids],
            "fallback": fallback, "source_paths": [root / CONFIG, cache, coarse_path]}


def project_step(raw_hec, coarse_elevation, parents, terrain, threshold=0.03):
    # Match EventDataAccess.sample exactly: strict coarse filtering, then parent WSE.
    raw_hec = np.nan_to_num(raw_hec, nan=-999.0)
    coarse_wse = np.where(raw_hec > coarse_elevation + threshold,
                          raw_hec, coarse_elevation).astype(np.float32)
    baseline = coarse_wse[parents]
    # Match the existing LF benchmark's physical projection onto fine terrain.
    raw_depth = baseline - terrain
    depth = np.where(raw_depth >= threshold, raw_depth, 0.0).astype(np.float32)
    return baseline, (terrain + depth).astype(np.float32), depth


def export_event(context, event, ids, terrain, times, output):
    config = context["config"]
    order = np.argsort(context["fine_ids"])
    positions = np.searchsorted(context["fine_ids"][order], ids)
    if np.any(positions >= len(order)) or not np.array_equal(context["fine_ids"][order[positions]], ids):
        raise ValueError("Prediction IDs do not match the LF parent graph")
    parents = context["parents"][order[positions]]
    source = config.paths.dataset_root / "HD_model_data/Low-fidelity" / f"BurnettRV_LFmodelB.{event['HEC_RAS_plan']}.hdf"
    shape = (len(times), len(ids))
    baseline, wse, depth = [np.empty(shape, dtype=np.float32) for _ in range(3)]
    with HecEvent(source, config.data.flow_area_name, context["coarse_ids"], config.data.hec_warmup_hours) as hec:
        if hec.n_timesteps < len(times):
            raise ValueError("LF event is shorter than the evaluated HF event")
        time_path = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp"
        dates = _decode_hec_times(hec._handle[time_path][:])
        hours = (dates - dates[0]) / np.timedelta64(1, "h")
        aligned_hours = hours[hec._kept_timesteps[:len(times)]]
        if not np.allclose(aligned_hours, times, rtol=0, atol=1e-8):
            raise ValueError("LF/HF elapsed event times do not align")
        for step in range(len(times)):
            baseline[step], wse[step], depth[step] = project_step(
                hec.read_wse(step), context["coarse_elevation"], parents, terrain,
                config.data.dry_threshold_m)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, fine_global_ids=ids, terrain=terrain, time_hours=aligned_hours,
                            lf_baseline_wse=baseline, lf_wse=wse, lf_depth=depth,
                            coarse_parent_global_ids=context["coarse_ids"][parents],
                            nearest_parent_fallback=context["fallback"][order[positions]])
    temporary.replace(output)
    return depth, source
