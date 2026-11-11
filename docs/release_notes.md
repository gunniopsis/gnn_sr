# Public release scope and provenance

Prepared against the November 2026 final thesis, especially Appendix B (pages 35-39), Appendix C (pages 39-41), and the Burnett evaluation in the main report and Appendix D.

The source release includes 3 width runs, 41 loss runs, 20 message-passing runs, the adopted full-domain run and supporting Burnett LF/LSG evaluation. It includes no other catchment or adaptation study. The two example polygons are the original development AOI and Burnett full domain.

Scientific model/data/graph/loss/training/metric modules are copied from the working project, with text line endings normalized to LF. Core scientific calculations are unchanged. Public-only differences are:

- One canonical full-domain name, `c2_f2`, replaces the deployment-specific `c2_f2_8h` name in configurations and consumers.
- The full-domain config enforces the report's 100-epoch boundary and disables automatic epoch-cap extension. Its runner checks all remaining training settings against the AOI reference.
- The depth planner exposes only the report's 20 configurations.
- Verification uses this repository's root manifest instead of old deployment-bundle manifests and frozen historical run hashes.
- Cluster templates contain no personal allocation code. Continuation and evaluation are explicit submissions. Optimizer/scheduler/random-state resume behaviour remains unchanged.
- Analysis labels refer to the validation-selected checkpoint; original epoch numbers are documentation, not enforced outcomes for newly trained models.
- The training-figure command reuses the original plotting functions without bundling document-authoring machinery.

`SOURCE_MANIFEST.json` lists every delivered source file, its SHA-256, and its origin in the working project or a public release template. The manifest excludes its own hash to avoid recursion. The separate ZIP checksum covers the whole archive, including the manifest. No datasets, environments, output histories, checkpoints or generated plots are included.

Verify the source with `python scripts/verify_repository.py`. A deliberate later source edit requires a new verified release manifest before using workflows that enforce it. Resume new runs only with the source/environment that created them. This snapshot does not claim bit-for-bit reproduction across different GPUs or dependency versions.
