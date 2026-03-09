# Active Plan

## Current milestone

- `v1.56`: Object-Memory Logic Uplift

## Single goal

- Decide whether a minimal `object_memory_v0` uplift can turn the already-proven `YOLO26n + SAM3` persistence signal into stronger lost-object support and chain object grounding without widening scope into retrieval or decision logic.

## Status

- [x] Added fake and real object-memory uplift manifests under `configs/benchmarks/v1.56_object_memory_*.yaml`.
- [x] Added `configs/queries/object_memory_logic_core_v1.yaml` to stress lost-object, chain-support, persistence support, and object-memory recall.
- [x] `src/pov_compiler/perception/object_memory_v0.py` now supports the smallest persistence-aware merge/alias/last-seen uplift without changing the top-level output schema.
- [x] `scripts/run_object_memory_uplift_pilot.py` and `scripts/report_object_memory_uplift.py` now produce compare/report artifacts for current vs `persistence_v1` object-memory logic.
- [x] Fake object-memory uplift pilot validated under `data/outputs/v156_object_memory_fake/` with `object_memory_logic_status=improved`.
- [x] Real object-memory uplift pilot validated under `data/outputs/v156_object_memory_real/` with `object_memory_logic_status=improved`.
- [x] Object-memory uplift report validated under `data/outputs/v156_object_memory_real/object_memory_uplift/` with `next_action_recommendation=need_more_object_memory_logic`.
- [x] `paper_ready/` can now carry `object_memory_uplift/`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] Provider dry-run remained optional and was skipped because `PARATERA_*` env vars were absent.
- [x] `v1.56` milestone marked done.

## v1.57 Candidate Tasks

- Retrieval-side chain grounding after object-memory uplift
- Decision-side consumption of persistent object-memory evidence
- Medium-scale SAM3 + object-memory admission gate
- Lost-object query-family hardening after persistence-backed memory
- Appendix split for YOLO-only vs YOLO+SAM3+memory evidence

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on benchmark, reporting, export, and provenance; do not widen runtime scope unless absolutely necessary.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/run_object_memory_uplift_pilot.py`
  - `scripts/report_object_memory_uplift.py`
  - `scripts/export_paper_ready.py`
  - `src/pov_compiler/bench/reporting/object_memory_uplift.py`
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
