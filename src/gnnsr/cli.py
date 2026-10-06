from __future__ import annotations

import argparse
import json

from .config import ExperimentConfig
from .geometry import load_burnett_geometries
from .graph import prepare_aoi_graph
from .io import EventCatalog, npz_member_info


def _load_config(path: str) -> ExperimentConfig:
    return ExperimentConfig.from_toml(path)


def command_inspect(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    coarse, fine = load_burnett_geometries(config.paths.dataset_root)
    catalog = EventCatalog(config.paths.dataset_root)
    first = catalog.records[0]
    high_info = npz_member_info(first.tuflow_path, "wl_data")
    payload = {
        "dataset_root": str(config.paths.dataset_root),
        "aoi_path": str(config.paths.aoi_path),
        "events": len(catalog),
        "event_groups": {
            str(group): sum(record.group == group for record in catalog) for group in range(1, 5)
        },
        "hec_cells": len(coarse.xy),
        "tuflow_cells": len(fine.xy),
        "first_event_wse_shape": high_info.shape,
        "first_event_post_warmup_timesteps": high_info.shape[0] - config.data.tuflow_warmup_steps,
        "tuflow_array_is_memory_mappable": True,
        "configured_fold": config.training.fold,
        "configured_seed": config.training.seed,
    }
    print(json.dumps(payload, indent=2))


def command_prepare(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    graph, path = prepare_aoi_graph(config, force=args.force)
    print(json.dumps({"graph_cache": str(path), **graph.summary()}, indent=2))


def command_train(args: argparse.Namespace) -> None:
    from .train import train_model

    config = _load_config(args.config)
    segment_seconds = args.segment_hours * 3600 if args.segment_hours is not None else None
    checkpoint = train_model(
        config,
        force_graph=args.force_graph,
        resume=args.resume,
        segment_seconds=segment_seconds,
    )
    print(f"Best checkpoint: {checkpoint}")


def command_evaluate(args: argparse.Namespace) -> None:
    from .train import evaluate_model

    config = _load_config(args.config)
    metrics = evaluate_model(config, args.checkpoint, save_predictions=not args.no_save_predictions)
    print(f"Event metrics: {metrics}")






def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gnnsr", description="Burnett River two-scale GNN super-resolution"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect paths and dataset dimensions")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.set_defaults(handler=command_inspect)

    prepare_parser = subparsers.add_parser("prepare", help="Build and cache the AOI graph")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(handler=command_prepare)

    train_parser = subparsers.add_parser("train", help="Train one configured fold and seed")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--force-graph", action="store_true")
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest.pt, or start a resumable run when no checkpoint exists",
    )
    train_parser.add_argument(
        "--segment-hours",
        type=float,
        help="Stop cleanly between epochs before this invocation exceeds the given hours",
    )
    train_parser.set_defaults(handler=command_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate held-out events")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--no-save-predictions", action="store_true")
    evaluate_parser.set_defaults(handler=command_evaluate)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
