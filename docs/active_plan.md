# Active Plan

## Current milestone

- `v1.60`: Harder Mainline Admission Closure For Persistent Memory

## Single goal

- Keep `persistent_object_memory_v2` on the promoted mainline, harden the paired real sample contract to `adequate`, separate provider semantics noise from real admission evidence, and make the explicit mainline panels first-class in `paper_ready/` and `submission_pack/`.

## Status

- [x] Added `configs/benchmarks/v1.60_mainline_harder_*_{baseline,persistent}.yaml` to freeze the harder paired baseline/persistent contracts for fake and real mainline runs.
- [x] Refreshed the v160 fixture evidence so the paired contract reaches `selected_uids_count=8`, `paired_sample_count=16`, and adequate coverage thresholds.
- [x] `scripts/compare_persistent_memory_main.py` now refreshes compare provenance with `sample_contract_status`, `large_sample_claim_status`, `promotion_ready`, `mainline_admission_ready`, and provider semantics cleanup fields.
- [x] `scripts/report_persistent_memory_main_decision.py` now separates `promotion_ready` from `mainline_admission_ready`.
- [x] Added `scripts/report_harder_sample_contract.py` and `scripts/report_mainline_admission_closure.py` on top of shared reporting in `src/pov_compiler/bench/reporting/mainline_admission_closure.py`.
- [x] Harder sample-contract evidence now reaches `sample_contract_status=adequate` and `large_sample_claim_status=supported`.
- [x] Mainline cleanup now improves from `explained` to `improved`, while explicitly isolating provider semantics noise from real run cleanliness.
- [x] Final closure now lands at `partial_but_harder`, with `promotion_ready=true`, `mainline_admission_ready=false`, and `recommended_next_step=need_provider_semantics_cleanup`.
- [x] `paper_ready/` explicitly carries `persistent_memory_main_compare/`, `persistent_memory_main_decision/`, `mainline_admission_cleanup/`, `harder_sample_contract/`, and `mainline_admission_closure/`.
- [x] `submission_pack/` explicitly carries the same mainline panels and no longer leaves obvious `None` placeholders in its self-description.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.60` milestone marked done.

## v1.61 Candidate Tasks

- Provider semantics cleanup for admission noise-only partial runs
- Final wording split between `supported` and `supported_with_caveat` mainline claims
- Persistent-memory appendix pack for camera-ready evidence bundles
- Retrieval-side follow-up only if strict gain regresses after provider cleanup
- Mainline compare dashboard over baseline, persistent, cleanup, and closure panels

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Keep work on benchmark, reporting, export, and provenance; do not widen runtime scope unless absolutely necessary.
- Prefer modifying only these existing non-doc files for the next milestone:
  - `scripts/compare_persistent_memory_main.py`
  - `scripts/report_persistent_memory_main_decision.py`
  - `scripts/report_mainline_admission_cleanup.py`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py`
  - `src/pov_compiler/bench/reporting/mainline_admission_cleanup.py`
  - `src/pov_compiler/bench/reporting/mainline_admission_closure.py`
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
