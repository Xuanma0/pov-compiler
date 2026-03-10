# Active Plan

## Current milestone

- `v1.61`: Documentation Consolidation And Bilingual Hardening

## Single goal

- Consolidate repository docs around the v1.60 evidence state, add canonical bilingual entry points, classify legacy docs, and make `paper_ready/` and `submission_pack/` explicitly self-describing.

## Status

- [x] Added canonical bilingual doc entry points at `docs/README.en.md` and `docs/README.zh-CN.md`.
- [x] Added paired canonical English/Chinese docs for overview, repo map, experiment history, current mainline status, development workflow, and glossary.
- [x] Added `docs/doc_inventory.md`, `docs/doc_migration_map.md`, and `docs/doc_style_guide.md` to classify documentation and migration state.
- [x] Added `docs/archive/README.md` to explain archive policy instead of deleting historical milestone evidence.
- [x] Root `README.md` now points to canonical docs and current v1.60 evidence rather than acting as a milestone log.
- [x] Added `scripts/check_docs_encoding.py` and `scripts/check_docs_integrity.py`.
- [x] Added docs-focused smoke coverage for encoding, integrity, bilingual pairs, and reading order.
- [x] `paper_ready` and `submission_pack` exports now include repository docs entry sections and explicit reading order.
- [x] `python scripts/check_docs_encoding.py` returned `status=ok`.
- [x] `python scripts/check_docs_integrity.py` returned `status=ok`.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `python scripts/security_scan_secrets.py` passed with `found_count=0`.
- [x] `v1.61` milestone marked done.

## v1.62 Candidate Tasks

- Provider semantics cleanup for admission-noise-only partial runs
- Camera-ready evidence index across compare, decision, cleanup, and closure panels
- Canonical changelog refresh aligned with v1.42-v1.61 milestones
- Historical docs archival move from soft redirect to physical archive tree
- Mainline evidence dashboard over compare, decision, cleanup, and closure panels

## Frozen constraints

- Prefer new files over editing runtime files.
- Keep follow-up work on docs, reporting, export, and provenance; do not widen runtime scope unless absolutely necessary.
- Prefer modifying only these existing files for the next milestone:
  - `README.md`
  - `CHANGELOG.md`
  - `docs/**/*.md`
  - `scripts/export_paper_ready.py`
  - `scripts/export_submission_pack.py`
  - `tests/test_export_paper_ready_smoke.py`
- Do not modify, unless implementation proves it is impossible to avoid:
  - `src/pov_compiler/retrieval/retriever.py`
  - `src/pov_compiler/streaming/runner.py`
  - `src/pov_compiler/perception/**`
  - `scripts/eval_nlq.py`
  - `scripts/run_ab_bye_compare.py`
  - `scripts/run_model_stack_compare.py`
  - `src/pov_compiler/l3_decisions/model_compiler.py`
  - `src/pov_compiler/repository/*`
  - `src/pov_compiler/models/*`
  - `src/pov_compiler/pipeline.py`
  - `src/pov_compiler/bench/suite_runner.py`
  - `src/pov_compiler/bench/manifest.py`
- Preserve existing top-level `Output` fields, compare artifact names, and paper-ready provenance contracts.
