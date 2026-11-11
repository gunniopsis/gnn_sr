"""Export the five final Burnett box-plot metrics as verified event tables."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/v2/full_domain/c2_f2/fold_1/seed_42/lsg_wet_mask_comparison"
LSG = ROOT / "outputs/v2/lsg_full_domain/burnett/fold_1/seed_42"
PLOT = ROOT / "reports/figures/full_domain_comparison/burnett_full_domain_lf_gnnsr_lsg_boxplots.json"
DEST = ROOT / "reports/tables/burnett_fold1_evaluation_tables"
MODELS = (("LF", "LF HEC-RAS"), ("GNN-SR", "GNN-SR"), ("LSG", "LSG"))
METRICS = (
    ("maximum_extent_csi", "Maximum flood extent — CSI, POD and RFA", "CSI and POD: higher is better. RFA: lower is better."),
    ("average_peak_difference_m", "Peak water depth — average peak difference (m)", "Signed mean of HF TUFLOW peak depth minus model peak depth. Positive values indicate underprediction; negative values indicate overprediction. Closer to zero is better."),
    ("maximum_depth_r2", "Peak water depth — coefficient of determination (R²)", "Higher is better; the ideal value is 1."),
    ("rmse_m", "Water depth hydrographs — AvgRMSE (m)", "Retains the box-plot label AvgRMSE and its adopted pooled event RMSE calculation. Lower is better."),
    ("fidelity_index", "Water depth hydrographs — fidelity index (FI)", "Higher is better; the ideal value is 1. Retains the adopted FI calculation."),
)


def verified_rows(directory):
    path = directory / "event_metrics.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((directory / "completion.json").read_text())
    if manifest["state"] != "complete" or manifest["artifacts"][path.name] != digest:
        raise ValueError(f"Metric verification failed: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream)), digest


def main():
    rows, digest = verified_rows(SOURCE)
    lsg_rows, lsg_digest = verified_rows(LSG)
    plot = json.loads(PLOT.read_text())
    with (ROOT / "reference_files/BurnettRV/BurnettRV_event_summary.csv").open(newline="", encoding="utf-8-sig") as stream:
        events = [r["tuflow_evt_name_old"] for r in csv.DictReader(stream) if int(r["Group"]) == 1]
    expected = {(event, model) for event in events for model, _ in MODELS}
    records = {(r["event_name"], r["model"]): r for r in rows}
    lsg = {r["event_name"]: r for r in lsg_rows if r["mask"] == "authors_fold_wet"}
    if len(events) != 18 or len(rows) != 54 or set(records) != expected or len(lsg) != 18 or set(lsg) != set(events):
        raise ValueError("Event coverage mismatch")
    if plot["source_sha256"] != digest or plot["events"] != events:
        raise ValueError("Source or ordering differs from the final box plots")
    precise = []
    lsg_csi_changes = {}
    for number, event in enumerate(events, 1):
        scores = {}
        reference_wet = None
        for model, _ in MODELS:
            row = records[event, model]
            counts = lsg[event] if model == "LSG" else row
            for candidate in (row, counts):
                if (int(candidate["group"]), int(candidate["n_target_cells"]), int(candidate["n_timesteps"])) != (1, 769709, 383) or candidate["model"] != model:
                    raise ValueError("Scoring support/model mismatch")
            scores[model] = {key: float(row[key]) for key, _, _ in METRICS}
            if model == "LSG":
                lsg_csi_changes[event] = float(counts["maximum_extent_csi"]) - scores[model]["maximum_extent_csi"]
                if f'{float(counts["maximum_extent_csi"]):.3f}' != f'{scores[model]["maximum_extent_csi"]:.3f}':
                    raise ValueError("Recomputed LSG CSI differs at displayed precision")
                scores[model]["maximum_extent_csi"] = float(counts["maximum_extent_csi"])
            tp, fp, fn, tn = (int(counts[k]) for k in ("TP", "FP", "FN", "TN"))
            if min(tp, fp, fn, tn) < 0 or tp + fp + fn + tn != 769709:
                raise ValueError("Invalid maximum-extent counts")
            if reference_wet is not None and tp + fn != reference_wet:
                raise ValueError("Reference extent differs between models")
            reference_wet = tp + fn
            if not math.isclose(tp / (tp + fp + fn), scores[model]["maximum_extent_csi"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Classification counts disagree with plotted CSI")
            scores[model].update(POD=tp / (tp + fn), RFA=fp / (tp + fp))
            if not all(math.isfinite(value) for value in scores[model].values()):
                raise ValueError("Non-finite metric")
        precise.append({"test_event_number": number, "event_name": event, "models": scores})

    title = "Burnett River fold 1 — per-event evaluation tables"
    note = ("All tables use the same 18 held-out events, numbered in the supplied event catalogue order, "
            "and compare against HF TUFLOW on the common 769,709-cell LSG fold-1 mask over hours 48–430 "
            "(383 timesteps). GNN-SR uses the validation-selected full-domain checkpoint. "
            "Values are rounded to three decimal places; full precision and event identifiers are retained in the accompanying JSON. "
            "The tables retain the final box-plot source values, with LSG maximum-extent CSI taken from the verified recomputed evaluation for consistency with POD/RFA. This leaves all displayed CSI values unchanged.")
    extent_note = ("Maximum extent uses depth ≥ 0.03 m. CSI = TP/(TP+FP+FN); POD = TP/(TP+FN); "
                   "RFA = FP/(TP+FP), the false-alarm ratio. LSG POD/RFA use the verified recomputed "
                   "authors_fold_wet evaluation. Its CSI differs from the box-plot source by approximately 0.00000238 for event 6 (no change at three decimals); other events agree to floating-point precision.")
    markdown = [f"# {title}", "", note, ""]
    markup = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>',
              '<style>body{font-family:Arial,sans-serif;color:#182536;margin:32px;max-width:1200px}h1{font-size:24px}h2{font-size:19px}p{font-size:14px;line-height:1.5}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:7px 10px;border-bottom:1px solid #d5dce4;text-align:right}th{background:#edf2f7}tbody tr:nth-child(even){background:#f7f9fb}section{margin:32px 0 48px}@media print{@page{size:A4 landscape;margin:14mm}section{break-before:page;margin:0}section:first-of-type{break-before:auto}table{font-size:11px}th,td{padding:5px 8px}}</style></head><body>',
              f'<h1>{html.escape(title)}</h1><p>{html.escape(note)}</p>']
    for index, (key, heading, interpretation) in enumerate(METRICS, 1):
        keys = (key, "POD", "RFA") if index == 1 else (key,)
        headers = ["Test event"] + [f'{"CSI" if metric == key else metric}: {label}' if index == 1 else label for metric in keys for _, label in MODELS]
        table = [[str(r["test_event_number"])] + [f'{r["models"][model][metric]:.3f}' for metric in keys for model, _ in MODELS] for r in precise]
        caption = f"Table {index}. {heading}"
        explanation = interpretation + (" " + extent_note if index == 1 else "")
        markdown += [f"## {caption}", "", "| " + " | ".join(headers) + " |", "|" + "---:|" * len(headers)]
        markdown += ["| " + " | ".join(row) + " |" for row in table]
        markdown += ["", explanation, ""]
        markup += [f'<section><h2>{html.escape(caption)}</h2><table><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in headers) + '</tr></thead><tbody>']
        markup += ['<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in table]
        markup += [f'</tbody></table><p>{html.escape(explanation)}</p></section>']
    markup += ['</body></html>']
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    DEST.with_suffix(".html").write_text("\n".join(markup), encoding="utf-8")
    provenance = {"source": str((SOURCE / "event_metrics.csv").relative_to(ROOT)), "source_sha256": digest,
                  "lsg_counts_source": str((LSG / "event_metrics.csv").relative_to(ROOT)), "lsg_source_sha256": lsg_digest,
                  "note": note, "maximum_extent_note": extent_note,
                  "lsg_csi_change_from_boxplots": lsg_csi_changes, "events": precise}
    DEST.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Created five tables, each with 18 events: {DEST.with_suffix('.html')}")


if __name__ == "__main__":
    main()
