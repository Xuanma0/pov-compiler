# Active Plan

## Current milestone

- `v1.42`: Benchmark Suite + Statistical Significance + Prompt Registry

## Single goal

- Build a manifest-driven result production line that can reproducibly generate paper main results, significance tables, prompt provenance, and a submission-ready package without refactoring the core offline/retrieval/runtime stack.

## Status

- [x] `v1.42` design boundary frozen in docs.
- [x] Experiment Manifest schema and loader added.
- [x] Benchmark Suite Runner added with collect-only smoke path.
- [x] Statistical Significance pipeline added with degrade-safe outputs.
- [x] File-backed Prompt Registry added with validation and prompt-lock export.
- [x] `scripts/export_paper_ready.py` extended for suite/significance provenance.
- [x] xdist-enabled pytest path verified with `python -m pytest -q -n auto`.
- [x] `v1.42` milestone marked done.

## v1.43 Candidate Tasks

- Make `submission_pack` emission part of the default smoke/export path.
- Add real run/resume execution lanes to the benchmark suite manifest.
- Wire prompt registry provenance into runtime prompt selection with minimal runtime touch.
- Expand significance inputs to richer paired per-uid tables and stronger failure attribution.
- Add CI job(s) for `v1.42` smoke outputs and artifact contract validation.

## Frozen constraints

- Prefer new files over editing existing runtime files.
- Initial implementation should modify at most these existing non-doc files:
  - `scripts/export_paper_ready.py`
  - `README.md`
  - `tests/test_export_paper_ready_smoke.py`
- Only escalate beyond that when blocked by prompt runtime integration, and then prefer:
  - `src/pov_compiler/l3_decisions/model_compiler.py`
  - `src/pov_compiler/retrieval/model_planner.py`
  - `src/pov_compiler/repository/summarizer.py`
- Do not modify, unless implementation proves it is impossible to avoid:
  - `src/pov_compiler/pipeline.py`
  - `src/pov_compiler/retrieval/retriever.py`
  - `src/pov_compiler/streaming/runner.py`
  - `scripts/eval_nlq.py`
- Preserve existing `Output` top-level fields and current paper-ready artifact names.
- Keep `python scripts/test_fast.py` as the official local xdist-enabled test entry.
