# Active Plan

## Current milestone

- `v1.46`: Real Pilot Provider Telemetry Diagnosis

## Single goal

- Promote the pilot runner from fake-only diagnosis to real-pilot-ready provider telemetry diagnosis, so weak deltas can be explained in terms of usage/cost/latency/parse-fail/fallback noise without touching runtime core modules.

## Status

- [x] `v1.46` real/fake pilot manifests added with telemetry contract flags and provider metadata.
- [x] `scripts/run_main_real_benchmark.py --mode pilot` now chains suite, significance, health, provider telemetry, diagnosis, freeze, paper-ready, paper freeze, and submission pack.
- [x] Standalone `provider_telemetry/` sidecar added with `summary.json` and `by_variant.csv`.
- [x] `result_diagnosis/` now consumes provider telemetry and emits per-variant telemetry rows plus non-`unavailable` `provider_noise_summary`.
- [x] `paper_ready/` now carries `provider_telemetry/` and a provider-noise section in `report.md`.
- [x] `submission_pack/` now carries `provider_telemetry/` and provider-noise-first guidance in `README.md`.
- [x] Fake pilot validated end to end under `data/outputs/v146_fake_pilot`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.46` milestone marked done.

## v1.47 Candidate Tasks

- Real main_real pilot on live compare roots
- Provider telemetry from real model-stack compare artifacts
- Diagnosis-driven query-bank refinement loop
- Partial rerun and resume for main-result runner
- CI verification for paper-ready plus paper-freeze artifacts

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
