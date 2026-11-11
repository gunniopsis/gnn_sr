"""Loss physics, differentiability, diagnostic aggregation and experiment isolation."""

import csv
import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gnnsr.config import ExperimentConfig, LossConfig
from gnnsr.diagnostics import TimestepDiagnostics
from gnnsr.losses import composite_loss
from gnnsr.train import _validate_resume_checkpoint, train_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_loss_experiment import candidate_config, check_existing


def terms(depth, wet, weight=0.1, temperature=0.03):
    prediction = torch.tensor(depth, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(wet, dtype=torch.float64)
    zero = torch.zeros_like(prediction)
    return prediction, composite_loss(
        prediction, target, zero, zero, target, 0.0, 1.0, temperature, weight
    )


def test_lambda_zero_has_exact_mse_value_and_gradient():
    prediction, result = terms([-0.02, 0.05, 0.2], [0.0, 0.08, 0.3], weight=0)
    expected = torch.nn.functional.mse_loss(
        prediction, torch.tensor([0.0, 0.08, 0.3], dtype=prediction.dtype)
    )
    assert torch.equal(result.total, expected)
    assert torch.equal(
        torch.autograd.grad(result.total, prediction)[0],
        torch.autograd.grad(expected, prediction)[0],
    )


def test_physical_reconstruction_and_metres_temperature():
    pred = torch.tensor([0.02], requires_grad=True)
    result = composite_loss(
        pred,
        torch.zeros(1),
        torch.tensor([100.0]),
        torch.tensor([100.0]),
        torch.zeros(1),
        -0.01,
        2.0,
        0.03,
        0.1,
    )
    assert result.raw_depth.item() == pytest.approx(0.03, abs=1e-5)
    assert result.soft_wet.item() == pytest.approx(0.5, abs=1e-4)


def test_soft_extent_gradients_penalize_false_wet_and_missed_wet():
    pred, result = terms([0.04, -0.01], [0.0, 0.1])
    gradient = torch.autograd.grad(result.extent, pred)[0]
    assert gradient[0] > 0  # Gradient descent lowers the false-positive depth.
    assert gradient[1] < 0  # Raw negative depth must still have a useful gradient.
    assert torch.isfinite(gradient).all()


def test_all_dry_uses_mean_probability_and_finite_gradient():
    pred, result = terms([-0.02, 0.05], [0.0, 0.0])
    assert torch.equal(result.extent, result.soft_wet.mean())
    assert (torch.autograd.grad(result.extent, pred)[0] > 0).all()


def test_soft_csi_known_overlap_and_saturated_cases():
    _, result = terms([0.03, 0.03], [0.1, 0.0])
    assert result.extent.item() == pytest.approx(1 - (0.5 + 1e-6) / (1.5 + 1e-6))
    pred, extreme = terms([100.0, -100.0], [1.0, 0.0], temperature=0.001)
    assert extreme.extent.item() == pytest.approx(0.0)
    assert torch.isfinite(torch.autograd.grad(extreme.total, pred)[0]).all()


def test_diagnostics_undefined_csi_and_pooled_rmse():
    diagnostic = TimestepDiagnostics(0.03)
    _, dry = terms([0.0, 0.0], [0.0, 0.0])
    diagnostic.add(dry, torch.zeros(2), SimpleNamespace(event_index=0, timestep=0))
    assert diagnostic.summary()["mean_timestep_csi"] is None
    _, wet = terms([0.04, 0.0], [0.0, 0.1])
    diagnostic.add(
        wet,
        torch.tensor([0.0, 0.1], dtype=torch.float64),
        SimpleNamespace(event_index=0, timestep=1),
    )
    summary = diagnostic.summary()
    assert summary["csi_valid_timesteps"] == 1
    assert summary["csi_both_empty_timesteps"] == 1
    assert summary["mean_timestep_csi"] == 0
    assert summary["false_alarms"] == summary["misses"] == 1
    assert summary["pooled_depth_rmse_m"] == pytest.approx(np.sqrt((0.04**2 + 0.1**2) / 4))
    assert summary["mean_timestep_depth_rmse_m"] != summary["pooled_depth_rmse_m"]


@pytest.mark.parametrize(
    "weight,temperature", [(-1, 0.03), (float("nan"), 0.03), (0, 0), (0, float("inf"))]
)
def test_invalid_loss_configuration(weight, temperature):
    config = candidate_config("control")
    with pytest.raises(ValueError):
        replace(config, loss=LossConfig(weight, temperature)).validate()


def test_candidates_preserve_baseline_science():
    from plan_loss_experiments import commands, load_plan

    baseline = ExperimentConfig.from_toml(
        ROOT / "experiments/v2/hidden_width/configs/width_64.toml"
    )
    candidates = load_plan()
    assert len(candidates) == 31
    all_candidates = json.loads((ROOT / "experiments/v2/composite_loss/candidates.json").read_text())
    assert load_plan(stage="all") == all_candidates
    for mode in ("preflight", "train"):
        planned_commands = list(commands(mode))
        assert len(planned_commands) == len(set(planned_commands)) == 31
        for item, command in zip(candidates, planned_commands):
            assert f"GNNSR_CANDIDATE={item['name']},GNNSR_MODE={mode} " in command
            assert command.endswith("gadi/run.pbs")
    for item in all_candidates:
        config = candidate_config(item["name"])
        for section in ("data", "graph", "model", "training"):
            assert getattr(config, section) == getattr(baseline, section)
        assert config.paths.aoi_path == baseline.paths.aoi_path
        assert config.paths.output_dir != baseline.paths.output_dir
        assert config.loss.extent_weight == item["extent_weight"]


def test_temperature_extension_isolated_from_completed_runs():
    from itertools import product

    from plan_loss_experiments import commands, load_plan

    initial = load_plan()
    extension = load_plan(stage="temperature_extension")
    assert len(extension) == 10
    assert {(c["temperature_m"], c["extent_weight"]) for c in extension} == set(
        product((0.5, 1.0), (0.001, 0.01, 0.1, 1.0, 10.0))
    )
    assert not {c["name"] for c in initial} & {c["name"] for c in extension}
    initial_paths = {candidate_config(c["name"]).paths.output_dir for c in initial}
    for c in extension:
        config = candidate_config(c["name"])
        assert config.paths.output_dir not in initial_paths
        assert config.loss.temperature_m == c["temperature_m"]
        assert config.paths.output_dir.name == config.paths.cache_dir.name == c["name"]
    for mode in ("train", "preflight"):
        jobs = list(commands(mode, stage="temperature_extension"))
        assert len(set(jobs)) == 10
        for c, command in zip(extension, jobs):
            assert f"GNNSR_CANDIDATE={c['name']},GNNSR_MODE={mode} " in command
            assert command.endswith("gadi/run.pbs")


def test_provenance_rejects_changed_loss(tmp_path):
    metadata = {"loss": {"extent_weight": 0.1}}
    (tmp_path / "loss_experiment.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError):
        check_existing(tmp_path, metadata, False)
    with pytest.raises(ValueError):
        check_existing(tmp_path, {"loss": {"extent_weight": 1.0}}, True)


def test_resume_rejects_changed_loss():
    config = candidate_config("control")
    checkpoint = {"config": config.as_dict()}
    with pytest.raises(ValueError, match="loss"):
        _validate_resume_checkpoint(
            replace(config, loss=replace(config.loss, extent_weight=0.1)), checkpoint
        )





def test_heldout_verification_checks_epoch_hashes_and_summary(tmp_path):
    from plot_fold1_boxplots import PANELS
    from report_loss_evaluation import verified_metrics
    from run_width_experiment import sha256

    path = tmp_path / "test_event_metrics.csv"
    rows = [
        {
            "event_name": f"event_{i}",
            "group": 1,
            "n_timesteps": 383,
            "n_target_cells": 2314,
            **{p.key: 0.5 for p in PANELS},
        }
        for i in range(18)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    trial = {
        "directory": tmp_path,
        "name": "fixture",
        "checkpoint_sha256": "checkpoint",
        "normalization_sha256": "norm",
        "best": {"epoch": "7"},
    }
    marker = {
        "checkpoint_sha256": "checkpoint",
        "normalization_sha256": "norm",
        "checkpoint_epoch": 7,
        "metrics_sha256": sha256(path),
    }
    (tmp_path / "loss_evaluation.json").write_text(json.dumps(marker))
    summary = {"events": 18, "mean_event_metrics": {p.key: 0.5 for p in PANELS}}
    (tmp_path / "test_summary.json").write_text(json.dumps(summary))
    assert len(verified_metrics(trial)) == 18
    marker["checkpoint_epoch"] = 8
    (tmp_path / "loss_evaluation.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="selected checkpoint"):
        verified_metrics(trial)
    marker["checkpoint_epoch"] = 7
    (tmp_path / "loss_evaluation.json").write_text(json.dumps(marker))
    summary["mean_event_metrics"]["maximum_extent_csi"] = 0.9
    (tmp_path / "test_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Summary disagrees"):
        verified_metrics(trial)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="selected checkpoint/metrics"):
        verified_metrics(trial)


def test_local_report_ranks_validation_and_renders(tmp_path):
    """Synthetic fixture only; verifies reporting without accessing held-out data."""
    from hashlib import sha256
    digest = lambda data: sha256(data).hexdigest()
    from report_loss_experiments import report
    from run_width_experiment import comparison_settings

    for name, errors in [("control", [0.4, 0.3, 0.32]), ("t0p03_l0p01", [0.4, 0.29, 0.31])]:
        directory = tmp_path / name / "fold_1/seed_42"
        directory.mkdir(parents=True)
        config = candidate_config(name).as_dict()
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "training_status.json").write_text(
            json.dumps({"state": "complete", "epoch": 3})
        )
        provenance = {
            key: "fixture"
            for key in (
                "splits",
                "split_file_sha256",
                "aoi_sha256",
                "source_sha256",
                "runner_sha256",
                "provenance_helper_sha256",
                "python",
                "torch",
                "cuda",
                "gpu",
            )
        }
        provenance.update(settings=comparison_settings(config), loss=config["loss"])
        (directory / "loss_experiment.json").write_text(json.dumps(provenance))
        (directory / "normalization.json").write_text("{}")
        torch.save(
            {
                "epoch": 2,
                "validation_depth_rmse_m": errors[1],
                "config": config,
                "normalization_path": "normalization.json",
            },
            directory / "best.pt",
        )
        rows = []
        for epoch, error in enumerate(errors, 1):
            row = {
                "epoch": epoch,
                "training_mse": 0.01 / epoch,
                "validation_depth_rmse_m": error,
                "learning_rate": 0.001,
                "mean_gradient_norm_before_clipping": 0.5,
                "clipped_step_fraction": 0.1,
                "epoch_seconds": 10,
            }
            for phase in ("train", "validation"):
                row.update(
                    {
                        f"{phase}_{k}": v
                        for k, v in {
                            "residual_mse": 0.01,
                            "extent_loss": 0.5,
                            "weighted_extent_loss": 0.005,
                            "total_loss": 0.015,
                            "mean_timestep_depth_rmse_m": error,
                            "mean_timestep_csi": 0.8,
                            "false_alarms": 10,
                            "misses": 5,
                        }.items()
                    }
                )
            rows.append(row)
        with (directory / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report(tmp_path)
    ranked = list(csv.DictReader((tmp_path / "comparison/validation_ranking.csv").open()))
    assert ranked[0]["candidate"] == "t0p03_l0p01"
    coverage = list(csv.DictReader((tmp_path / "comparison/initial_grid_coverage.csv").open()))
    assert len(coverage) == 31
    assert sum(row["local_status"] == "complete_verified" for row in coverage) == 2
    evidence = json.loads((tmp_path / "comparison/verification.json").read_text())
    assert evidence["initial_grid_complete"] is False
    assert len(evidence["initial_grid_missing_locally_or_incomplete"]) == 29
    assert len(list((tmp_path / "comparison").glob("*.png"))) == 3
    assert digest((tmp_path / "comparison/training.png").read_bytes())


@pytest.mark.skipif(
    not (ROOT / "reference_files/BurnettRV/Geometry_data").exists(), reason="Dataset unavailable"
)
def test_real_aoi_lambda_zero_matches_baseline_and_composite_resumes(tmp_path):
    """Tiny local CPU test, not scientific training or a replacement for GPU preflight."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = candidate_config("control", preflight=True)
        config = replace(
            config,
            paths=replace(
                config.paths, output_dir=tmp_path / "plain", cache_dir=tmp_path / "cache"
            ),
            training=replace(config.training, device="cpu"),
        )
        plain = replace(config, loss=replace(config.loss, record_diagnostics=False))
        plain_path = train_model(plain)
        control = replace(config, paths=replace(config.paths, output_dir=tmp_path / "control"))
        control_path = train_model(control)
        a = torch.load(plain_path, weights_only=False)
        b = torch.load(control_path, weights_only=False)
        for key in a["model_state"]:
            assert torch.equal(a["model_state"][key], b["model_state"][key])
        assert a["validation_depth_rmse_m"] == b["validation_depth_rmse_m"]
        combined = replace(
            config,
            paths=replace(config.paths, output_dir=tmp_path / "combined"),
            loss=replace(config.loss, extent_weight=0.01),
        )
        first = train_model(combined)
        continued = replace(combined, training=replace(combined.training, max_epochs=2))
        train_model(continued, resume=True)
        rows = list(csv.DictReader((first.parent / "history.csv").open()))
        assert [int(r["epoch"]) for r in rows] == [1, 2]
        for row in rows:
            assert float(row["train_total_loss"]) == pytest.approx(
                float(row["train_residual_mse"]) + float(row["train_weighted_extent_loss"])
            )
            assert float(row["validation_depth_rmse_m"]) == float(
                row["validation_pooled_depth_rmse_m"]
            )
        with gzip.open(first.parent / "timestep_metrics/epoch_00002.csv.gz", "rt") as handle:
            steps = list(csv.DictReader(handle))
        assert len(steps) == 3 and [s["phase"] for s in steps] == ["train", "train", "validation"]
    finally:
        torch.set_num_threads(original_threads)
