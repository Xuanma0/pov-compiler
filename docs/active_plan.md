# Active Plan

## Current milestone

- `v1.52`: Local YOLO26n Signal Uplift Pilot

## Single goal

- Raise weak real-pilot signal coverage with the smallest possible local-perception runtime hook, then prove whether object coverage, object memory, lost-object support, and query strength materially improve under local `YOLO26n`.

## Status

- [x] `configs/perception/yolo26n_local.yaml` added with the inventoried local checkpoint `D:\BYES\models\yolo26\yolo26n.pt` and explicit perception/cache settings.
- [x] `src/pov_compiler/perception/backends.py` now exposes stable backend/model metadata for local real perception runs.
- [x] `src/pov_compiler/perception/runner.py` cache keys now include the actual model path/name and hand-task path, preventing stub/YOLO cache collisions.
- [x] `scripts/run_signal_uplift_pilot.py` added to compare baseline vs local-`YOLO26n` perception without copying evaluator logic.
- [x] `src/pov_compiler/bench/reporting/signal_uplift.py` and `scripts/report_signal_uplift.py` added to summarize object coverage, object memory, lost-object support, and query-strength gains.
- [x] Fake uplift pilot validated under `data/outputs/v152_signal_uplift_fake/` with `signal_uplift_status=improved` and `next_action_recommendation=keep_yolo26n_and_scale_real`.
- [x] Real uplift pilot validated under `data/outputs/v152_signal_uplift_real/` with local `YOLO26n`, `signal_uplift_status=improved`, and `next_action_recommendation=query_bank_still_too_weak`.
- [x] `paper_ready/` now carries `signal_uplift/` and exports the summary plus compare-side uplift figures.
- [x] `submission_pack/` now preserves `signal_uplift/` and updates README reading order to include the perception-signal interpretation step.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] Provider health side-check intentionally skipped because `PARATERA_*` env vars were absent; it did not block the milestone.
- [x] `v1.52` milestone marked done.

## v1.53 Candidate Tasks

- SAM3 segmentation uplift pilot on the same real small sample
- YOLO26s scale-up and cost/latency tradeoff against YOLO26n
- Signal-aware query bank refresh for lost-object and chain-heavy groups
- Local-perception canonical export with zero `missing_tasks`
- Geometry-track DA3 feasibility memo without runtime integration

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on benchmark, reporting, export, provenance, and only the minimal perception-runtime surface required for local model integration.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `src/pov_compiler/perception/backends.py`
  - `src/pov_compiler/perception/runner.py`
  - `src/pov_compiler/perception/object_memory_v0.py` (only if blocked by label fragmentation)
  - `src/pov_compiler/pipeline.py` (only for perception-config passthrough or metadata wiring)
  - `scripts/run_signal_uplift_pilot.py`
  - `scripts/report_signal_uplift.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py`
  - `src/pov_compiler/bench/reporting/signal_uplift.py`
  - `tests/test_export_paper_ready_smoke.py`
- Only escalate beyond that when blocked by signal-uplift reporting or local-perception metadata handling, and still prefer script-layer fallbacks before broader runtime edits.
- Do not modify, unless implementation proves it is impossible to avoid:
  - `src/pov_compiler/retrieval/retriever.py`
  - `src/pov_compiler/streaming/runner.py`
  - `scripts/eval_nlq.py`
  - `scripts/run_ab_bye_compare.py`
  - `scripts/run_model_stack_compare.py`
- Preserve existing `Output` top-level fields, current compare/paper-ready artifact names, and the result-layer provenance contract.
- Keep `python scripts/test_fast.py` as the official local xdist-enabled test entry.
