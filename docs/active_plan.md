# Active Plan

## Current milestone

- `v1.55`: SAM3 Object-Persistence Uplift

## Single goal

- Decide whether local `SAM3`-style persistence improves object memory, lost-object support, and chain object grounding enough to justify a larger main-real follow-up beyond `YOLO26n only`.

## Status

- [x] Added `configs/perception/yolo26n_sam3_local.yaml` for the smallest `YOLO26n + SAM3` local perception contract.
- [x] Added fake and real object-persistence pilot manifests under `configs/benchmarks/v1.55_signal_uplift_*.yaml`.
- [x] `src/pov_compiler/perception/backends.py` and `src/pov_compiler/perception/runner.py` now distinguish `YOLO26n only` from `YOLO26n + SAM3` in metadata and cache keys.
- [x] Added `scripts/run_object_persistence_pilot.py` and `scripts/report_object_persistence_uplift.py`.
- [x] Fake object-persistence pilot validated under `data/outputs/v155_signal_uplift_fake/` with `object_persistence_status=improved`.
- [x] Real object-persistence pilot validated under `data/outputs/v155_signal_uplift_real/` with `object_persistence_status=improved`.
- [x] Object-persistence report validated under `data/outputs/v155_signal_uplift_real/object_persistence_uplift/` with `next_action_recommendation=need_more_object_memory_logic`.
- [x] `paper_ready/` can now carry `object_persistence_uplift/`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] Provider dry-run remained optional and was skipped because `PARATERA_*` env vars were absent.
- [x] `v1.55` milestone marked done.

## v1.56 Candidate Tasks

- Object-memory logic hardening for persistent tracks
- Larger real SAM3 follow-up with aligned sample
- Retrieval-side chain grounding after persistence uplift
- Formal SAM3 admission gate before medium-scale main_real
- Appendix split for YOLO-only vs YOLO+SAM3 evidence

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on benchmark, reporting, export, and provenance; do not widen runtime scope unless absolutely necessary.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/run_main_real_benchmark.py`
  - `scripts/compare_query_banks.py`
  - `scripts/report_query_bank_promotion_decision.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py` (only if pack needs to surface new result-layer panels)
  - `src/pov_compiler/bench/reporting/query_bank_promotion.py`
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
