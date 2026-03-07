# Active Plan

## Current milestone

- `v1.47`: Real Pilot Admission Control And Delta Audit

## Single goal

- Promote the pilot runner from telemetry-aware diagnosis to admission-controlled real-pilot triage, so weak deltas can explicitly say whether to expand the run and which layer to tune next without touching runtime core modules.

## Status

- [x] `v1.47` real/fake admission manifests added with explicit admission thresholds for sample size, no-data rate, effect size, and provider noise.
- [x] `scripts/run_main_real_benchmark.py --mode pilot` now chains suite, significance, health, provider telemetry, diagnosis, delta audit, admission control, freeze, paper-ready, paper freeze, and submission pack.
- [x] Standalone `delta_audit/` artifact layer added with per-budget `recommended_action` rows and `fig_delta_audit_breakdown.*`.
- [x] Standalone `admission_control/` artifact layer added with `admission_status`, fail reasons, and admission metrics.
- [x] `paper_ready/` now carries `admission_control/` and `delta_audit/`, and `report.md` includes admission plus delta-audit summaries.
- [x] `submission_pack/` now carries `admission_control/` and `delta_audit/`, and its README enforces the order admission -> diagnosis -> delta audit -> canonical figures.
- [x] Fake admission pilot validated end to end under `data/outputs/v147_fake_admission`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.47` milestone marked done.

## v1.48 Candidate Tasks

- Live main_real pilot on real compare roots
- Per-query query-bank weakness attribution
- Admission-aware sample expansion policy
- Real-provider telemetry ingestion from external sidecars
- CI verification for admission plus delta-audit artifacts

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
