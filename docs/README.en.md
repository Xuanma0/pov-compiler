[中文版本](./README.zh-CN.md)

# Documentation Entry

- Purpose: Canonical English entry for the repository docs at the v1.60 state.
- Version scope: `v1.60+` documentation cleanup baseline.
- Reading order:
  1. [current_mainline_status.md](./en/current_mainline_status.md)
  2. [experiment_history.md](./en/experiment_history.md)
  3. [project_overview.md](./en/project_overview.md)
  4. [repo_map.md](./en/repo_map.md)
  5. [development_workflow.md](./en/development_workflow.md)
  6. [glossary.md](./en/glossary.md)
  7. [doc_inventory.md](./doc_inventory.md)
  8. [doc_migration_map.md](./doc_migration_map.md)
- Related docs:
  [doc_style_guide.md](./doc_style_guide.md),
  [archive/README.md](./archive/README.md),
  [local_inventory.md](./local_inventory.md),
  [provider_probe.md](./provider_probe.md)

## Canonical Current Docs

- Project overview: [docs/en/project_overview.md](./en/project_overview.md)
- Repository map: [docs/en/repo_map.md](./en/repo_map.md)
- Experiment history: [docs/en/experiment_history.md](./en/experiment_history.md)
- Current mainline status: [docs/en/current_mainline_status.md](./en/current_mainline_status.md)
- Development workflow: [docs/en/development_workflow.md](./en/development_workflow.md)
- Glossary: [docs/en/glossary.md](./en/glossary.md)

## Generated Evidence Reading Order

When reading exported evidence, keep this order fixed:

1. `persistent_memory_main_compare/`
2. `persistent_memory_main_decision/`
3. `mainline_admission_cleanup/`
4. `harder_sample_contract/`
5. `mainline_admission_closure/`

Then use the canonical paper-ready tables and figures.

## Notes

- `docs/task_notes/` is historical design evidence, not the main entry.
- `docs/ARCHITECTURE.md`, `docs/CLI.md`, `docs/OUTPUTS.md`, and `docs/REPRO.md` are legacy/reference docs unless the current canonical docs explicitly point back to them.
