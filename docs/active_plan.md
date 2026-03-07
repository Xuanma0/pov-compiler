# Active Plan

## Current milestone

- `v1.48`: Admission Calibration And Query Strength Audit

## Single goal

- Promote the pilot runner from admission-controlled triage to calibration-aware and query-aware real-pilot analysis, so small pilots can say whether the admission profile is sensible and whether weak deltas come from weak queries rather than weak algorithms.

## Status

- [x] `v1.48` real/fake pilot manifests added with `admission_calibration_enabled` and `query_strength_audit_enabled`.
- [x] `scripts/run_main_real_benchmark.py --mode pilot` now chains suite, significance, health, provider telemetry, diagnosis, delta audit, admission control, admission calibration, query strength audit, freeze, paper-ready, paper freeze, and submission pack.
- [x] Standalone `admission_calibration/` artifact layer added with threshold recommendations plus `calibration_status`.
- [x] Standalone `query_strength_audit/` artifact layer added with per-group `recommended_action` rows and `fig_query_strength_breakdown.*`.
- [x] `paper_ready/` now carries `admission_calibration/` and `query_strength_audit/`, and `report.md` includes calibration plus query-strength summaries.
- [x] `submission_pack/` now carries `admission_calibration/` and `query_strength_audit/`, and its README enforces the order admission -> calibration -> diagnosis -> delta audit -> query strength -> canonical figures.
- [x] Fake small pilot validated end to end under `data/outputs/v148_fake_pilot`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.48` milestone marked done.

## v1.49 Candidate Tasks

- Live main_real pilot on real compare roots
- Data-driven promotion and pruning of frozen query banks
- Calibration history and trend tracking across repeated pilots
- Real-provider cost and latency sidecar ingestion from external logs
- Canonical paper export without missing task placeholders

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
