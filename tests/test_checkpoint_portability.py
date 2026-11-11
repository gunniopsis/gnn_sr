from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnnsr.config import ExperimentConfig, PathsConfig
from gnnsr.train import _resolve_checkpoint_artifact, _validate_checkpoint_compatibility


def test_relocated_checkpoint_finds_companion_artifact(tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    companion = tmp_path / "normalization_graph.json"
    companion.write_text("{}", encoding="utf-8")

    resolved = _resolve_checkpoint_artifact(
        checkpoint_path, "/scratch/project/run/normalization_graph.json"
    )

    assert resolved == companion


def test_checkpoint_rejects_a_different_aoi_filename():
    config = ExperimentConfig(paths=PathsConfig(aoi_path=Path("expected.geojson")))
    saved = config.as_dict()
    saved["paths"]["aoi_path"] = "/remote/path/different.geojson"

    with pytest.raises(ValueError, match="paths.aoi_path"):
        _validate_checkpoint_compatibility(config, {"config": saved})
