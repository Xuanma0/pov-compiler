# Active Plan

## Current milestone

- `v1.57`: Persistent Object Memory v2

## Single goal

- Decide whether a minimal persistent object-memory v2 can turn the already-proven `YOLO26n + SAM3` persistence signal into stronger lost-object support, reappearance recall, and chain object grounding without widening scope into retrieval or decision logic.

## Status

- [x] Added fake and real persistent-object-memory manifests under `configs/benchmarks/v1.57_object_memory_*.yaml`.
- [x] Added `configs/queries/persistent_object_memory_core_v1.yaml` to stress lost-object, reappearance, chain grounding, persistent recall, and last-tracked/last-interacted behavior.
- [x] `src/pov_compiler/perception/object_memory_v0.py` now supports the smallest persistent memory v2 fields (`memory_tier`, short/long-term scores, reappearance, persistence confidence) without changing the top-level output schema.
- [x] `scripts/run_persistent_object_memory_pilot.py` and `scripts/report_persistent_object_memory_uplift.py` now produce compare/report artifacts for `persistence_v1` vs `persistent_v2`.
- [x] Fake persistent-object-memory pilot validated under `data/outputs/v157_object_memory_fake/` with `persistent_object_memory_status=improved`.
- [x] Real persistent-object-memory pilot validated under `data/outputs/v157_object_memory_real/` with `persistent_object_memory_status=improved`.
- [x] Persistent-object-memory uplift report validated under `data/outputs/v157_object_memory_real/persistent_object_memory_uplift/` with `next_action_recommendation=promote_persistent_object_memory`.
- [x] `paper_ready/` can now carry `persistent_object_memory_uplift/`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] Provider dry-run remained optional and was skipped because `PARATERA_*` env vars were absent.
- [x] `v1.57` milestone marked done.

## v1.58 Candidate Tasks

- Medium-scale persistent-memory admission gate
- Retrieval-side consumption of persistent object-memory evidence
- Decision-side use of reappearance-backed lost-object memory
- Lost-object query-family hardening after persistent memory promotion
- Appendix split for YOLO-only vs YOLO+SAM3 vs persistent-memory-v2 evidence

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on benchmark, reporting, export, and provenance; do not widen runtime scope unless absolutely necessary.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/run_persistent_object_memory_pilot.py`
  - `scripts/report_persistent_object_memory_uplift.py`
  - `scripts/export_paper_ready.py`
  - `src/pov_compiler/bench/reporting/persistent_object_memory_uplift.py`
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
