from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class PathsConfig:
    dataset_root: Path = Path("reference_files/BurnettRV")
    aoi_path: Path | None = None
    output_dir: Path = Path("outputs/burnettrv_aoi")
    cache_dir: Path = Path("cache/burnettrv")


@dataclass(frozen=True)
class DataConfig:
    flow_area_name: str = "BurnettRV_region"
    tuflow_warmup_steps: int = 48
    hec_warmup_hours: float = 47.5
    dry_threshold_m: float = 0.03
    model_crs: str = "EPSG:28356"


@dataclass(frozen=True)
class GraphConfig:
    fine_grid_spacing_m: float = 20.0
    fine_spacing_tolerance_m: float = 0.05
    fine_halo_hops: int = 3
    coarse_halo_hops: int = 2


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    coarse_layers: int = 2
    fine_layers: int = 3
    leaky_relu_slope: float = 0.01
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    fold: int = 1
    seed: int = 42
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 300
    early_stopping_patience: int = 30
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1e-5
    gradient_clip_norm: float = 1.0
    device: str = "auto"
    extend_epoch_cap_if_improving: bool = True
    debug_max_train_samples: int | None = None
    debug_max_validation_samples: int | None = None
    debug_max_test_samples: int | None = None


@dataclass(frozen=True)
class LossConfig:
    extent_weight: float = 0.0
    temperature_m: float = 0.03
    record_diagnostics: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    source_path: Path | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        source = Path(path).resolve()
        with source.open("rb") as handle:
            values = tomllib.load(handle)
        base = source.parent
        paths = dict(values.get("paths", {}))
        for key in ("dataset_root", "aoi_path", "output_dir", "cache_dir"):
            raw = paths.get(key)
            if raw in (None, ""):
                if key == "aoi_path":
                    paths[key] = None
                continue
            candidate = Path(raw)
            paths[key] = candidate if candidate.is_absolute() else (base / candidate).resolve()
        cfg = cls(
            paths=PathsConfig(**paths),
            data=DataConfig(**values.get("data", {})),
            graph=GraphConfig(**values.get("graph", {})),
            model=ModelConfig(**values.get("model", {})),
            training=TrainingConfig(**values.get("training", {})),
            loss=LossConfig(**values.get("loss", {})),
            source_path=source,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not math.isfinite(self.loss.extent_weight) or self.loss.extent_weight < 0:
            raise ValueError("loss.extent_weight must be finite and nonnegative")
        if not math.isfinite(self.loss.temperature_m) or self.loss.temperature_m <= 0:
            raise ValueError("loss.temperature_m must be finite and positive")
        if not 1 <= self.training.fold <= 4:
            raise ValueError("training.fold must be one of 1, 2, 3, or 4")
        if self.graph.fine_halo_hops < self.model.fine_layers:
            raise ValueError("fine_halo_hops must cover all fine message-passing layers")
        if self.graph.coarse_halo_hops < self.model.coarse_layers:
            raise ValueError("coarse_halo_hops must cover all coarse message-passing layers")
        if self.model.dropout != 0.0:
            raise ValueError("The agreed baseline requires model.dropout = 0.0")
        if self.paths.aoi_path is None:
            raise ValueError("paths.aoi_path must identify the target AOI polygon")

    def as_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "__dataclass_fields__"):
                return {name: convert(getattr(value, name)) for name in value.__dataclass_fields__}
            return value

        return convert(self)
