# Active Plan

## Current milestone

- `v1.54`: Medium-Scale v1 vs v2 Query-Bank Compare

## Single goal

- Run aligned medium-scale real experiments for `core_real_v1` and `core_real_v2_candidate`, then produce a formal promotion decision instead of inferring promotion from small-pilot uplift alone.

## Status

- [x] Added aligned medium-scale manifests for `core_real_v1` and `core_real_v2_candidate` under both fake and real profiles.
- [x] `scripts/run_main_real_benchmark.py` now annotates compare summaries with `query_bank_id`, `query_bank_hash`, `compare_pair_id`, and `run_signature_hash`.
- [x] `scripts/compare_query_banks.py` added with fail-fast alignment checks for UID set, budgets, provider signature, and perception signature.
- [x] `scripts/report_query_bank_promotion_decision.py` added to convert aligned compare outputs into `promote_v2 | keep_v1 | expand_sample_first | consider_sam3_next`.
- [x] Medium-scale real `v1` run validated under `data/outputs/v154_main_real_v1/`.
- [x] Medium-scale real `v2` run validated under `data/outputs/v154_main_real_v2/`.
- [x] Aligned compare validated under `data/outputs/v154_query_bank_compare/compare/` with `alignment_ok=true`, `selected_uids_count=6`, and matched provider/perception signatures.
- [x] Promotion decision validated under `data/outputs/v154_query_bank_compare/promotion_decision/` with `promotion_decision=keep_v1`.
- [x] `paper_ready/` and `submission_pack/` can now carry `query_bank_compare/` and `query_bank_promotion_decision/`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.54` milestone marked done.

## v1.55 Candidate Tasks

- Larger aligned sample for `core_real_v2_candidate`
- Strict-metric-focused v2 query hardening
- SAM3 decision gate after larger aligned v2 run
- Main-table promotion freeze for `v1` vs `v2`
- Appendix export split for candidate vs analysis-only query banks

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
