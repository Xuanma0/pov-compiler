# Decisions

## 2026-03-07: v1.42 is an additive benchmark-production layer

- Context
  The repo already has many compare/sweep scripts and stable artifact contracts, but the orchestration surface is fragmented. The `v1.42` goal is to turn that surface into a repeatable result-production line rather than reworking the core runtime.
- Decision
  Add a manifest-driven suite layer, significance pipeline, prompt registry, and submission-pack export as new modules and scripts. Treat existing scripts as stable command-level producers wherever possible.
- Consequences
  This keeps scope bounded, preserves current retrieval/runtime behavior, and lets `v1.42` focus on reproducibility and paper output quality instead of core algorithm refactors.

## 2026-03-07: Avoid oversized orchestrators in v1.42

- Context
  The highest-risk files are already oversized and highly coupled: `src/pov_compiler/pipeline.py`, `src/pov_compiler/retrieval/retriever.py`, `src/pov_compiler/streaming/runner.py`, and `scripts/eval_nlq.py`.
- Decision
  Do not use `v1.42` to push more logic into those files. New benchmark/suite/significance/prompt logic must live in new files, with only narrow adapter edits to existing exporters and docs.
- Consequences
  Lower regression risk against `Output` contracts, retrieval behavior, and existing smoke-tested orchestration. The cost is some short-term duplication at the suite/export layer, which is acceptable for this milestone.

## 2026-03-07: Prompt Registry starts as a reproducibility contract first

- Context
  Prompted model paths already exist for decisions, planner, and repo summary, but rewiring all runtime prompt construction immediately would widen the blast radius of `v1.42`.
- Decision
  Introduce a file-backed Prompt Registry and prompt lock/export flow first. In `v1.42`, the registry must be loadable, validatable, versioned, and exported with benchmark outputs. Runtime wiring into `model_compiler.py`, `model_planner.py`, and `repository/summarizer.py` is conditional, not mandatory for phase-one implementation.
- Consequences
  `v1.42` gains prompt provenance and submission reproducibility immediately, while avoiding unnecessary edits to working model paths. If runtime prompt selection becomes necessary, the escalation path is explicit and limited.

## 2026-03-07: Submission packaging extends paper-ready export

- Context
  `scripts/export_paper_ready.py` is already the repo's aggregation point for tables, figures, copied compare panels, and `snapshot.json`.
- Decision
  Reuse that exporter as the only existing aggregation script modified in `v1.42`. Extend it to accept benchmark-suite outputs, significance outputs, and prompt-registry assets, then emit `submission_pack/` under the paper-ready output root.
- Consequences
  Existing paper-ready flows remain backward-compatible. New publication-oriented outputs are additive instead of creating another top-level exporter with overlapping responsibility.

## 2026-03-07: xdist fast test entry is standardized, not reinvented

- Context
  The repo already has `scripts/test_fast.py`, which prefers `pytest -q -n auto` and falls back cleanly when `pytest-xdist` is unavailable.
- Decision
  Keep `scripts/test_fast.py` as the official local fast-test entry for `v1.42`. Document it in README and use it in the design/test plan instead of introducing another wrapper.
- Consequences
  No extra test-runner churn is introduced. `v1.42` can advertise a fast validation path immediately with near-zero implementation risk.
