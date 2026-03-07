# Active Plan

## Current milestone

- `v1.44`: Main Real Runner + Health Gate + Canonical Paper Map + Paper Freeze

## Single goal

- Promote the benchmark suite into a real main-result execution layer with one canonical runner, gate-checked outputs, canonical paper numbering, and paper-level freeze.

## Status

- [x] `v1.44` main real/fake manifests added as the canonical entry contracts.
- [x] `scripts/run_main_real_benchmark.py` added as the single result-layer orchestrator.
- [x] Result-health gate added with explicit fail reasons and non-zero exit on gate failure.
- [x] Canonical paper map export added with stable `Table N` / `Figure N` copies.
- [x] Paper-level freeze added for canonical paper artifacts.
- [x] `paper_ready/` and `submission_pack/` now carry canonical paper assets and paper freeze.
- [x] Dry collect validated at `data/outputs/v144_main_real_dry`.
- [x] Fake smoke path validated end-to-end under `data/outputs/v144_fake_smoke`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `v1.44` milestone marked done.

## v1.45 Candidate Tasks

- Real compare producer execution lanes
- Provider cost and parse-fail health gates
- Canonical caption metadata and paper metadata pack
- CI paper-freeze verification
- Partial rerun and resume for main-result bundles

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on the benchmark, reporting, export, and provenance layers.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/run_benchmark_suite.py`
  - `scripts/report_result_health.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py`
  - `scripts/freeze_benchmark_run.py`
  - `tests/test_benchmark_suite_runner_smoke.py`
  - `tests/test_export_paper_ready_smoke.py`
- Only escalate beyond that when blocked by producer/runtime integration, and then prefer minimal script-layer overlays first.
- Do not modify, unless implementation proves it is impossible to avoid:
  - `src/pov_compiler/pipeline.py`
  - `src/pov_compiler/retrieval/retriever.py`
  - `src/pov_compiler/streaming/runner.py`
  - `scripts/eval_nlq.py`
  - `scripts/run_ab_bye_compare.py`
- Preserve existing `Output` top-level fields and current paper-ready artifact names.
- Keep `python scripts/test_fast.py` as the official local xdist-enabled test entry.
