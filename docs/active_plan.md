# Active Plan

## Current milestone

- `v1.49`: Real Small Pilot + Provider Normalization + Query Promotion Pack

## Single goal

- Promote the fake small-pilot stack into a real-provider small pilot that can normalize provider telemetry, preserve explicit missing-field semantics, and export a promotion-ready query pack without touching runtime core modules.

## Status

- [x] `v1.49` real/fake pilot manifests added with `provider_normalization_enabled`, `query_promotion_enabled`, and `require_real_calls`.
- [x] `scripts/run_main_real_benchmark.py --mode pilot` now chains suite, significance, result health, provider telemetry, provider normalization, diagnosis, delta audit, admission control, admission calibration, query strength audit, query promotion pack, freeze, paper-ready, paper freeze, and submission pack.
- [x] Standalone `provider_normalization/` artifact layer added with unified schema, explicit missing semantics, and `normalization_status`.
- [x] Standalone `query_promotion_pack/` artifact layer added with `promoted_queries.yaml`, `analysis_only_queries.yaml`, and provenance-bearing `promotion_summary.json`.
- [x] `result_diagnosis` now prefers normalized telemetry and records fallback state when normalization is absent.
- [x] `paper_ready/` now carries `provider_normalization/` and `query_promotion_pack/`, and `report.md` includes normalization plus promotion summaries.
- [x] `submission_pack/` now carries normalized telemetry and promotion packs, and its README enforces the read order normalized telemetry -> query promotion -> main-figure interpretation.
- [x] Real small pilot validated end to end under `data/outputs/v149_main_real_pilot`, with explicit `gate_status=partial`, `admission_status=fail`, `calibration_status=weak`, and `normalization_status=partial` when real calls are unavailable.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.49` milestone marked done.

## v1.50 Candidate Tasks

- Live provider call observation and retry-safe real pilot admission
- Result-health no-data hardening without placeholder fallbacks
- Promotion-pack merge workflow for frozen query banks
- Cross-provider normalization history and drift tracking
- Canonical paper export without missing-task placeholders

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on the benchmark, reporting, export, and provenance layers.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/run_main_real_benchmark.py`
  - `scripts/report_result_diagnosis.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py`
  - `src/pov_compiler/bench/reporting/provider_telemetry.py`
  - `src/pov_compiler/bench/reporting/query_strength_audit.py`
  - `tests/test_benchmark_suite_runner_smoke.py`
  - `tests/test_export_paper_ready_smoke.py`
- Only escalate beyond that when blocked by result-layer no-data handling, and still prefer script-layer fallbacks before producer/runtime edits.
- Do not modify, unless implementation proves it is impossible to avoid:
  - `src/pov_compiler/pipeline.py`
  - `src/pov_compiler/retrieval/retriever.py`
  - `src/pov_compiler/streaming/runner.py`
  - `scripts/eval_nlq.py`
  - `scripts/run_ab_bye_compare.py`
- Preserve existing `Output` top-level fields and current paper-ready artifact names.
- Keep `python scripts/test_fast.py` as the official local xdist-enabled test entry.
