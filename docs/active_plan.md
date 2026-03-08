# Active Plan

## Current milestone

- `v1.51`: Repeatability Audit + Sample-Size Recommendation + Query Uplift Candidates

## Single goal

- Explain weak real-pilot outcomes with repeated-run evidence, produce conservative next-run sample-size guidance, and export query uplift candidates without promoting weak queries into the main bank.

## Status

- [x] `v1.51` real/fake repeat manifests added with `repeat_enabled`, `repeat_count`, `repeat_seed_strategy`, and `repeat_profile`.
- [x] `scripts/run_main_real_benchmark.py --mode pilot` now materializes repeated sidecar roots and chains repeatability audit, sample-size recommendation, and query uplift candidate export after the existing result-layer pipeline.
- [x] `provider_telemetry/summary.json` now carries repeated-run latency and availability aggregates for repeatability interpretation.
- [x] Standalone `repeatability_audit/` added with per-metric variance, coefficient of variation, `stability_flag`, and provider-noise interpretation.
- [x] Standalone `sample_size_recommendation/` added with conservative `ok|range_only|weak` status and UID/pair recommendations derived from repeatability and effect-size evidence.
- [x] Standalone `query_uplift_candidates/` added to separate “may improve with more signal/sample” from direct promotion.
- [x] `result_diagnosis` now consumes repeatability results and explicitly explains provider instability vs small-sample weakness vs stable no-effect.
- [x] `paper_ready/` now carries `repeatability_audit/`, `sample_size_recommendation/`, and `query_uplift_candidates/`, and `report.md` summarizes stability and recommended next sample size.
- [x] `submission_pack/` now carries the new audit layers and updates README reading order to proof -> repeatability -> sample size -> diagnosis/delta -> uplift/promotion -> figures.
- [x] Fake repeat pilot validated under `data/outputs/v151_fake_repeat` with `gate_status=ok`, `admission_status=partial`, `calibration_status=ok`, `repeatability_status=partial`, and `sample_size_recommendation_status=ok`.
- [x] Real repeat pilot validated under `data/outputs/v151_main_real_repeat` with live-call proof preserved, `admission_status=partial`, `calibration_status=partial`, `repeatability_status=partial`, and `sample_size_recommendation_status=range_only`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.51` milestone marked done.

## v1.52 Candidate Tasks

- Live compare-root ingestion without fixture-backed compare tables
- Repeatability-aware admission threshold tuning
- Query uplift to revised bank draft with human-review checkpoints
- Cost-aware repeated-pilot budget planner
- Canonical paper export with zero `missing_tasks`

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
