# Documentation Inventory

This inventory tracks the major documentation surfaces in the repository after the v1.61 cleanup.

| path | language | status | version_scope | action | canonical_target | notes |
|---|---|---|---|---|---|---|
| `README.md` | en | current | `v1.60+` | update | `docs/README.en.md`, `docs/README.zh-CN.md` | Short root entry only. |
| `CHANGELOG.md` | en | archive | `v0.2.0` | archive | `docs/en/experiment_history.md` | Historical but outdated. |
| `docs/README.en.md` | en | canonical | `v1.60+` | keep | self | English canonical docs entry. |
| `docs/README.zh-CN.md` | zh-CN | canonical | `v1.60+` | keep | self | Chinese canonical docs entry. |
| `docs/en/project_overview.md` | en | canonical | `v1.60+` | keep | self | Current high-level project description. |
| `docs/zh-CN/project_overview.md` | zh-CN | canonical | `v1.60+` | keep | self | Chinese overview counterpart. |
| `docs/en/repo_map.md` | en | canonical | `v1.60+` | keep | self | Current repository map. |
| `docs/zh-CN/repo_map.md` | zh-CN | canonical | `v1.60+` | keep | self | Chinese repo map counterpart. |
| `docs/en/experiment_history.md` | en | canonical | `v1.54-v1.60` | keep | self | Main experiment history narrative. |
| `docs/zh-CN/experiment_history.md` | zh-CN | canonical | `v1.54-v1.60` | keep | self | Chinese experiment history counterpart. |
| `docs/en/current_mainline_status.md` | en | canonical | `v1.60` | keep | self | Authoritative mainline state. |
| `docs/zh-CN/current_mainline_status.md` | zh-CN | canonical | `v1.60` | keep | self | Chinese mainline state counterpart. |
| `docs/en/development_workflow.md` | en | canonical | `v1.60+` | keep | self | Current collaboration workflow. |
| `docs/zh-CN/development_workflow.md` | zh-CN | canonical | `v1.60+` | keep | self | Chinese workflow counterpart. |
| `docs/en/glossary.md` | en | canonical | `v1.60+` | keep | self | Canonical terminology map. |
| `docs/zh-CN/glossary.md` | zh-CN | canonical | `v1.60+` | keep | self | Chinese terminology counterpart. |
| `docs/doc_inventory.md` | en | canonical | `v1.61` | keep | self | Current docs inventory. |
| `docs/doc_migration_map.md` | en | canonical | `v1.61` | keep | self | Migration and archive map. |
| `docs/doc_style_guide.md` | en | canonical | `v1.61+` | keep | self | Docs standards and terminology rules. |
| `docs/archive/README.md` | en | canonical | `v1.61+` | keep | self | Archive rules. |
| `docs/ARCHITECTURE.md` | en | archive | `v0.2.0-era` | update | `docs/en/project_overview.md` | Legacy architecture reference. |
| `docs/CLI.md` | en | archive | `early runtime` | update | `docs/en/development_workflow.md` | Legacy script reference. |
| `docs/OUTPUTS.md` | en | archive | `early output contract` | update | `docs/en/current_mainline_status.md` | Legacy output guide. |
| `docs/REPRO.md` | en | archive | `early repro flow` | update | `docs/en/development_workflow.md` | Legacy repro path. |
| `docs/FAQ.md` | en | current | `general reference` | keep | `docs/README.en.md` | Still useful, but not the canonical entry. |
| `docs/BYE_INJECTION.md` | en | current | `BYE-specific` | keep | `docs/en/repo_map.md` | Specialized reference. |
| `docs/decisions.md` | en | archive | `through v1.42` | archive | `docs/en/experiment_history.md` | Historical ADR subset only. |
| `docs/local_inventory.md` | en | current | `2026-03-10 local host` | keep | `docs/en/repo_map.md` | Local machine reference. |
| `docs/provider_probe.md` | en | current | `2026-03-10 provider probe` | keep | `docs/en/current_mainline_status.md` | Provider compatibility reference. |
| `docs/repo_review.md` | en | current | `repo review baseline` | keep | `docs/en/project_overview.md` | Deep internal review. |
| `docs/working_memory.md` | en | current | `operator note` | keep | `docs/en/development_workflow.md` | Short internal entry for future sessions. |
| `docs/active_plan.md` | en | current | `active milestone` | update | `docs/en/current_mainline_status.md` | Active work tracker, not canonical status. |
| `docs/task_notes/local_probe.md` | zh-CN | archive | `local probe` | keep | `docs/archive/README.md` | Historical task note kept in place. |
| `docs/task_notes/v1.42_design.md` to `docs/task_notes/v1.60_design.md` | mixed | archive | `v1.42-v1.60` | keep | `docs/archive/README.md` | Historical milestone evidence kept in place. |
| `data/outputs/v160_mainline_cleanup/paper_ready/*` | generated | current | `v1.60 evidence` | keep | `docs/en/current_mainline_status.md` | Generated paper-ready evidence panels. |
| `data/outputs/v160_mainline_cleanup/submission_pack/*` | generated | current | `v1.60 evidence` | keep | `docs/en/current_mainline_status.md` | Generated submission-ready evidence pack. |
