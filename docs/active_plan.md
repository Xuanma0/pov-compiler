# Active Plan

## Current milestone

- `v1.59`: Mainline Admission Cleanup And Sample Contract Hardening

## Single goal

- Make the `persistent_object_memory_v2` mainline conclusion publication-safe by explicitly separating promotion from admission, hardening sample/coverage/freeze evidence, and exporting the required mainline panels into `paper_ready/` and `submission_pack/`.

## Status

- [x] Added `configs/benchmarks/v1.59_mainline_cleanup_fake.yaml` and `configs/benchmarks/v1.59_mainline_cleanup_real.yaml` to freeze cleanup/sample-contract reporting expectations.
- [x] Added `scripts/report_mainline_admission_cleanup.py` and `scripts/report_sample_contract.py` on top of shared reporting in `src/pov_compiler/bench/reporting/mainline_admission_cleanup.py`.
- [x] `mainline_admission_cleanup` now explains why `promotion_decision=promote_persistent_memory_to_mainline` can coexist with `admission_status=partial`.
- [x] `sample_contract` now writes explicit `sample_contract_status` and `large_sample_claim_status` instead of leaving wording implicit.
- [x] Real cleanup report validated under `data/outputs/v159_mainline_cleanup/` with `mainline_admission_cleanup_status=explained`.
- [x] Real sample-contract report validated under `data/outputs/v159_sample_contract/` with `sample_contract_status=borderline` and `large_sample_claim_status=supported_with_caveat`.
- [x] `paper_ready/` export now explicitly carries `persistent_memory_main_compare/`, `persistent_memory_main_decision/`, `mainline_admission_cleanup/`, and `sample_contract/`.
- [x] `submission_pack/` export now explicitly carries the same mainline panels and documents the fixed reading order.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.59` milestone marked done.

## v1.60 Candidate Tasks

- Retrieval-side grounding from persistent-memory mainline evidence
- Decision-side use of reappearance-backed lost-object memory
- Large-sample repeatability and cost audit for persistent mainline
- Mainline wording split between large-sample and large-scale claims
- Appendix package for baseline vs persistent vs cleanup evidence

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on benchmark, reporting, export, and provenance; do not widen runtime scope unless absolutely necessary.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/report_mainline_admission_cleanup.py`
  - `scripts/report_sample_contract.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py`
  - `src/pov_compiler/bench/reporting/mainline_admission_cleanup.py`
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
