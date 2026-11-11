"""Export one cell's LF, GNN-SR and HF hydrographs from retained event predictions."""
import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/v2/full_domain/c2_f2/fold_1/seed_42"


def cell_column(ids, cell_id):
    matches = np.flatnonzero(ids == cell_id)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cell with global ID {cell_id}")
    return int(matches[0])


def load_hydrograph(run, event, cell_id):
    if Path(event).name != event or any(c in event for c in ("/", "\\", ":")):
        raise ValueError("Use an event name, not a path")
    with np.load(run / "predictions" / f"{event}.npz") as archive:
        column = cell_column(archive["fine_global_ids"], cell_id)
        terrain = float(archive["terrain"][column])
        series = {}
        for output, key in (("HF_WSE_m", "target_wse"), ("GNN_SR_WSE_m", "predicted_wse"),
                            ("HF_depth_m", "target_depth"), ("GNN_SR_depth_m", "predicted_depth")):
            # Load one compressed field at a time and retain only the selected cell.
            series[output] = archive[key][:, column].copy()
    with np.load(run / "lsg_wet_mask_comparison/lf_predictions" / f"{event}.npz") as archive:
        column = cell_column(archive["fine_global_ids"], cell_id)
        if float(archive["terrain"][column]) != terrain:
            raise ValueError("LF/GNN cell terrain differs")
        times = archive["time_hours"]
        for output, key in (("LF_WSE_m", "lf_wse"), ("LF_depth_m", "lf_depth"),
                            ("LF_parent_WSE_m", "lf_baseline_wse")):
            series[output] = archive[key][:, column].copy()
    if any(len(values) != len(times) for values in series.values()):
        raise ValueError("Hydrograph time dimensions differ")
    return {"time_hours": times, **series}, terrain


def plot(run, event, cell_id, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, terrain = load_hydrograph(run, event, cell_id)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"{event}_cell_{cell_id}"
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(series)
        writer.writerows(zip(*series.values(), strict=True))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax, suffix, label in zip(axes, ("WSE_m", "depth_m"), ("Water surface elevation (m)", "Water depth (m)")):
        for prefix, name, colour in (("HF", "HF reference", "black"), ("GNN_SR", "GNN-SR", "#1565c0"), ("LF", "LF", "#e67e22")):
            ax.plot(series["time_hours"], series[f"{prefix}_{suffix}"], label=name, color=colour)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].axhline(terrain, color="grey", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Time since event start (hours; warm-up excluded)")
    fig.suptitle(f"{event} — cell {cell_id}")
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=180)
    plt.close(fig)
    return stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--event", required=True)
    parser.add_argument("--cell-id", required=True, type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures/full_domain_hydrographs")
    args = parser.parse_args()
    print(plot(args.run, args.event, args.cell_id, args.output_dir))
