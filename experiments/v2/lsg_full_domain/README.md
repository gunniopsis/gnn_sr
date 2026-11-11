# Saved-LSG inference supporting the Burnett report

This is baseline evaluation only. It loads the externally supplied Burnett fold-1 LSG model; it does not train, tune or adapt it. It is included because the report's LSG POD/RFA, peak-area timing and measured inference times require predictions beyond the supplied five-metric summary.

Use the separate LSG environment and all inputs in `input_manifest.json`. Run `python scripts/verify_lsg_full_domain_inputs.py`, then `python scripts/run_lsg_full_domain.py`. Use `--resume` to continue a previously started evaluation. The PBS template is `gadi/evaluate.pbs` in this directory.

The 18 original events, warm-up, full-domain ordering and fold-specific scoring mask are preserved. Generated HDF5 predictions, event metrics and provenance belong in `outputs/v2/lsg_full_domain/burnett/fold_1/seed_42/`. See [report outputs](../../../docs/report_outputs.md) for their consumers.
