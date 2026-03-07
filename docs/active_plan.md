# Active Plan

## Current milestone

- `v1.45`: Main Real Pilot + Result Diagnosis Layer

## Single goal

- Promote the main-result runner into a safe pilot flow that can explain why a real main-result run is empty, weak, or trustworthy through a standalone result diagnosis layer.

## Status

- [x] `v1.45` real/fake pilot manifests added with fixed query-bank reuse and smaller pilot budgets.
- [x] `scripts/run_main_real_benchmark.py --mode pilot` now chains suite, significance, health, diagnosis, freeze, paper-ready, paper freeze, and submission pack.
- [x] Standalone `result_diagnosis/` artifacts added with explicit no-data / near-zero / significance / provider-noise summaries.
- [x] `paper_ready/` now carries `result_diagnosis/` and diagnosis figures.
- [x] `submission_pack/` now carries `result_diagnosis/` and diagnosis-first guidance in `README.md`.
- [x] Fake pilot validated end to end under `data/outputs/v145_fake_pilot`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.45` milestone marked done.

## v1.46 Candidate Tasks

- Real pilot provider telemetry ingestion
- Diagnosis-driven query-bank refinement loop
- Main-result resume / partial rerun support
- Canonical caption and caption-freeze pack
- CI verification for benchmark freeze plus paper freeze

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
