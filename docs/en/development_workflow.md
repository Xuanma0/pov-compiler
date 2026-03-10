[中文对照](../zh-CN/development_workflow.md)

# Development Workflow

- Purpose: Describe how the repo is worked on with the user and Codex.
- Version scope: `v1.60+`
- Reading order: project overview -> repo map -> this document.
- Related docs: [project_overview.md](./project_overview.md), [repo_map.md](./repo_map.md), [glossary.md](./glossary.md)

## Collaboration Model

- The user provides milestone intent and constraints.
- Codex first reads the relevant docs and outputs.
- Work is usually split into two phases:
  1. design review
  2. implementation + tests + closeout

## Typical Milestone Flow

1. Read `working_memory.md`, `repo_review.md`, `active_plan.md`, and the latest task note.
2. Inspect the relevant output directories.
3. Produce a design review with scope, risks, frozen files, and minimal test plan.
4. Implement within the allowed file set.
5. Run the exact smoke commands named in the milestone prompt.
6. Update the task note and `docs/active_plan.md`.
7. Commit only after the required validation passes.

## Validation Style

- Default full validation: `python -m pytest -q -n auto`
- Then run the milestone-specific smoke commands.
- Then run `python scripts/security_scan_secrets.py`

## Scope Discipline

- Prefer result-layer, reporting-layer, and export-layer changes.
- Do not widen scope into retrieval, streaming, or runtime unless the milestone explicitly allows it.
- Prefer new files over editing core orchestrators.
