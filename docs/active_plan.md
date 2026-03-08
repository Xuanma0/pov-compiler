# Active Plan

## Current milestone

- `v1.53`: Signal-Aware Query Bank Rewrite

## Single goal

- Convert the verified `YOLO26n` signal uplift into a stronger candidate main-result query bank, separating `candidate`, `analysis_only`, and `drop` decisions without mutating `core_real_v1`.

## Status

- [x] `core_real_v2_candidate.yaml` added as a signal-aware stronger-bank candidate and kept separate from `core_real_v1`.
- [x] `core_real_v2_analysis_only.yaml` added to isolate weak-but-informative repo-summary queries from the stronger main-result bank.
- [x] `scripts/rewrite_query_bank.py` added to derive the v2 candidate bank from `query_strength_audit`, `query_uplift_candidates`, `signal_uplift`, `delta_audit`, `admission_calibration`, and provider telemetry context.
- [x] `scripts/report_query_bank_selection.py` added to emit explicit `candidate | analysis_only | drop` decisions with reasons.
- [x] `scripts/run_signal_uplift_pilot.py` now preserves stronger-bank provenance in `manifest/query_bank_lock.json`, `manifest/query_banks/`, and compare summaries.
- [x] Fake stronger-bank pilot validated under `data/outputs/v153_query_bank_fake/` with `signal_uplift_status=improved`.
- [x] Real stronger-bank pilot validated under `data/outputs/v153_query_bank_real/` with `query_bank_id=core_real_v2_candidate`.
- [x] `query_bank_rewrite/` validated with `candidate_count=6`, `analysis_only_count=2`, `chosen_signal_labels=["bowl","cell phone"]`, and `rewrite_confidence=medium`.
- [x] `query_bank_selection/` validated with `candidate_groups=["decision","chain","lost_object"]` and `analysis_only_groups=["repo_summary"]`.
- [x] `signal_uplift/` now compares `signal_uplift_core_v1` against `core_real_v2_candidate` and reports `next_action_recommendation=promote_v2_candidate`.
- [x] `paper_ready/` now carries `query_bank_rewrite/` and `query_bank_selection/`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.53` milestone marked done.

## v1.54 Candidate Tasks

- Candidate-bank-to-main-real promotion gate
- Lost-object query hardening with segmentation evidence
- Query-bank A/B panel for main-result budget curves
- Analysis-only appendix export for repo-summary queries
- Larger real sample validation for `core_real_v2_candidate`

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on benchmark, reporting, export, and provenance; do not widen runtime scope unless absolutely necessary.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/run_signal_uplift_pilot.py`
  - `scripts/report_signal_uplift.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py` (only if pack needs to surface new result-layer panels)
  - `src/pov_compiler/bench/reporting/signal_uplift.py`
  - `src/pov_compiler/bench/reporting/query_strength_audit.py`
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
