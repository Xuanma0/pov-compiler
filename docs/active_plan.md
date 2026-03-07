# Active Plan

## Current milestone

- `v1.43`: Frozen Query Bank + Result Health + Benchmark Freeze

## Single goal

- Converge the benchmark suite into a real main-result production line with fixed query inputs, explicit health reporting, artifact freeze, and one-click paper-ready/submission-pack export.

## Status

- [x] `v1.43` query banks frozen as versioned YAML configs with stable hashes.
- [x] `v1.43` real/fake manifests added and wired into `run_benchmark_suite.py`.
- [x] Result-health report added with explicit no-data / zero-delta accounting.
- [x] Benchmark freeze added with SHA256 artifact ledger and freeze manifest.
- [x] `paper_ready/` and `submission_pack/` extended to include health/freeze/query-bank provenance.
- [x] Fake smoke path validated end-to-end under `data/outputs/v143_smoke`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `v1.43` milestone marked done.

## v1.44 Candidate Tasks

- Query-bank-aware producer execution lanes
- Per-UID paired significance contract
- CI benchmark freeze verification
- Default submission-pack export path
- Real main-result run/resume orchestration

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on the benchmark, reporting, export, and provenance layers.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/run_benchmark_suite.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py`
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
