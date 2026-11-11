"""Validate the report's complete 4-by-5 AOI message-passing grid."""
import argparse
import json
from itertools import product
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("experiments/v2/message_passing")
COARSE = (1, 2, 3, 4)
FINE = (1, 2, 3, 4, 6)


def load_plan(root=ROOT, stage="initial"):
    if stage not in ("initial", "all"):
        raise ValueError("Only the report's 20 configurations are included")
    plan = json.loads((root / EXPERIMENT / "candidates.json").read_text())
    expected = {(f"c{c}_f{f}", c, f) for c, f in product(COARSE, FINE)}
    actual = {(p["name"], p["coarse_layers"], p["fine_layers"]) for p in plan}
    if len(plan) != 20 or actual != expected:
        raise ValueError("Expected all 20 message-passing configurations")
    base = tomllib.loads((root / "experiments/v2/composite_loss/configs/t0p3_l0p01.toml").read_text())
    for item in plan:
        name = item["name"]
        cfg = tomllib.loads((root / EXPERIMENT / "configs" / f"{name}.toml").read_text())
        expected_config = {**base,
            "model": {**base["model"], "coarse_layers": item["coarse_layers"], "fine_layers": item["fine_layers"]},
            "graph": {**base["graph"], "coarse_halo_hops": 4, "fine_halo_hops": 6},
            "paths": {**base["paths"], "output_dir": f"../../../../outputs/v2/message_passing/{name}",
                      "cache_dir": f"../../../../cache/v2/message_passing/{name}"}}
        if cfg != expected_config:
            raise ValueError(f"Configuration differs from the report grid: {name}")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names-only", action="store_true")
    args = parser.parse_args()
    plan = load_plan()
    print("\n".join(p["name"] for p in plan) if args.names_only else json.dumps(plan, indent=2))

