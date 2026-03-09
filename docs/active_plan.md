# Active Plan

## Current milestone

- `v1.58`: Large-Sample Persistent Memory Mainline Admission

## Single goal

- Decide whether `persistent_object_memory_v2` is strong enough, on a large-sample real main experiment, to replace the current mainline memory route without widening scope into retrieval or decision logic.

## Status

- [x] Added aligned fake/real large-sample baseline manifests under `configs/benchmarks/v1.58_main_*_baseline.yaml`.
- [x] Added aligned fake/real large-sample persistent manifests under `configs/benchmarks/v1.58_main_*_persistent.yaml`.
- [x] `scripts/run_main_real_benchmark.py` now records `object_memory_logic_variant`, `query_bank_id/hash`, provider label/model route, `uid_set_id`, and `run_signature_hash` for paired mainline compare.
- [x] `scripts/compare_persistent_memory_main.py` now fails fast on contract mismatch and writes mainline compare tables, significance, figures, summary, and snapshot.
- [x] `scripts/report_persistent_memory_main_decision.py` now converts the aligned compare into a mainline admission decision with `promote_persistent_memory_to_mainline` support.
- [x] Large-sample real baseline run validated under `data/outputs/v158_main_real_baseline/`.
- [x] Large-sample real persistent run validated under `data/outputs/v158_main_real_persistent/`.
- [x] Persistent-memory main compare validated under `data/outputs/v158_persistent_memory_main_compare/compare/` with `alignment_ok=true` and `persistent_memory_main_status=improved`.
- [x] Mainline decision validated under `data/outputs/v158_persistent_memory_main_compare/promotion_decision/` with `promotion_decision=promote_persistent_memory_to_mainline`.
- [x] `paper_ready/` can now carry `persistent_memory_main_compare/` and `persistent_memory_main_decision/`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.58` milestone marked done.

## v1.59 Candidate Tasks

- Retrieval-side grounding from persistent object-memory evidence
- Decision-side use of reappearance-backed lost-object memory
- Large-sample persistent-memory repeatability and cost audit
- Lost-object query-family hardening after mainline promotion
- Appendix split for YOLO-only vs YOLO+SAM3 vs persistent-memory-v2 evidence

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on benchmark, reporting, export, and provenance; do not widen runtime scope unless absolutely necessary.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/run_main_real_benchmark.py`
  - `scripts/compare_persistent_memory_main.py`
  - `scripts/report_persistent_memory_main_decision.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py`
  - `src/pov_compiler/bench/reporting/persistent_memory_main.py`
  - `tests/test_export_paper_ready_smoke.py`
- Do not modify, unless implementation proves it is impossible to avoid:
  - `src/pov_compiler/perception/backends.py`
  - `src/pov_compiler/perception/runner.py`
  - `src/pov_compiler/retrieval/retriever.py`
  - `src/pov_compiler/streaming/runner.py`
  - `scripts/eval_nlq.py`
  - `scripts/run_ab_bye_compare.py`
  - `scripts/run_model_stack_compare.py`
  - `src/pov_compiler/l3_decisions/model_compiler.py`
  - `src/pov_compiler/repository/*`
  - `src/pov_compiler/pipeline.py`
  - `src/pov_compiler/bench/suite_runner.py`
  - `src/pov_compiler/bench/manifest.py`
- Preserve existing top-level `Output` fields, compare artifact names, and paper-ready provenance contracts.
