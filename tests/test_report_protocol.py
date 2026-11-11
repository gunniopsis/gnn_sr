"""Guard the public release against scope and scientific-protocol regressions."""
import sys
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plan_depth_experiments import load_plan as depth_plan
from plan_loss_experiments import load_plan as loss_plan
from run_full_domain_experiment import candidate_config, evaluate_selected
from gnnsr.model import TwoScaleFloodGNN


def test_report_grid_is_complete_and_keeps_fresh_control():
    assert len(loss_plan(stage="initial")) == 31
    assert len(loss_plan(stage="temperature_extension")) == 10
    assert len(loss_plan(stage="all")) == 41
    assert len(depth_plan()) == 20
    assert not list((ROOT / "experiments/v2/message_passing/configs").glob("c0_*.toml"))
    assert not list((ROOT / "experiments/v2/message_passing/configs").glob("*_f0.toml"))


def test_full_domain_uses_report_cap_and_original_selection_settings():
    cfg = candidate_config(root=ROOT)
    assert cfg.training.max_epochs == 100
    assert cfg.training.extend_epoch_cap_if_improving is False
    assert cfg.training.early_stopping_patience == 30
    assert cfg.training.scheduler_patience == 10
    assert cfg.training.fold == 1 and cfg.training.seed == 42
    assert cfg.model.coarse_layers == cfg.model.fine_layers == 2
    assert cfg.model.hidden_dim == 64
    assert sum(p.numel() for p in TwoScaleFloodGNN(cfg.model).parameters()) == 140481
    assert cfg.loss.temperature_m == .3 and cfg.loss.extent_weight == .01
    assert cfg.paths.output_dir == ROOT / "outputs/v2/full_domain/c2_f2"
    from gnnsr.train import _restored_epoch_limit
    assert _restored_epoch_limit(cfg, epoch=100, bad_epochs=0) == 100


def test_preflight_is_isolated_and_does_not_extend_training():
    cfg = candidate_config(preflight=True, root=ROOT)
    assert cfg.training.max_epochs == 1
    assert cfg.paths.output_dir.name == "preflight"
    assert cfg.training.debug_max_train_samples == 20
    assert cfg.training.debug_max_validation_samples == 5


def test_all_config_inputs_resolve_inside_repository():
    from gnnsr.config import ExperimentConfig
    for path in (ROOT / "experiments").rglob("*.toml"):
        cfg = ExperimentConfig.from_toml(path)
        assert cfg.paths.aoi_path.is_file()
        for value in vars(cfg.paths).values():
            assert value.is_relative_to(ROOT)


def test_incomplete_training_cannot_be_reported_as_final(tmp_path):
    import json
    (tmp_path / "training_status.json").write_text(json.dumps({"state": "segment_complete"}))
    with pytest.raises(RuntimeError, match="complete"):
        evaluate_selected(candidate_config(root=ROOT), tmp_path)


def test_full_evaluation_rejects_non_best_checkpoint_before_inference(tmp_path):
    import csv
    import json
    import torch
    (tmp_path / "training_status.json").write_text(json.dumps({"state": "complete"}))
    with (tmp_path / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "validation_depth_rmse_m"])
        writer.writeheader()
        writer.writerows([{"epoch": 1, "validation_depth_rmse_m": .2},
                          {"epoch": 2, "validation_depth_rmse_m": .3}])
    torch.save({"epoch": 2, "validation_depth_rmse_m": .3}, tmp_path / "best.pt")
    with pytest.raises(ValueError, match="minimum validation"):
        evaluate_selected(candidate_config(root=ROOT), tmp_path)


def test_manifest_verification_rejects_changed_content(tmp_path):
    import hashlib
    import json
    from gnnsr.release import verify_source
    source = tmp_path / "model.py"
    source.write_bytes(b"original")
    record = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (tmp_path / "SOURCE_MANIFEST.json").write_text(json.dumps({"files": [record]}))
    assert verify_source(tmp_path) == 1
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        verify_source(tmp_path)
