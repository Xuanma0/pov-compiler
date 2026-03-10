# Documentation Migration Map

This map records how legacy or duplicated docs should be interpreted after the v1.61 cleanup.

| original_path | new_path_or_target | reason | migration_state |
|---|---|---|---|
| `README.md` | `docs/README.en.md`, `docs/README.zh-CN.md` | Root README should stay short and point to canonical docs. | updated |
| `docs/ARCHITECTURE.md` | `docs/en/project_overview.md`, `docs/zh-CN/project_overview.md` | Original architecture narrative is old and should not be the first-read source. | soft_redirect |
| `docs/CLI.md` | `docs/en/development_workflow.md`, `docs/zh-CN/development_workflow.md` | Script usage has expanded beyond the old CLI overview. | soft_redirect |
| `docs/OUTPUTS.md` | `docs/en/current_mainline_status.md`, generated `paper_ready/` panels | Output interpretation now depends on the current mainline evidence chain. | soft_redirect |
| `docs/REPRO.md` | `docs/en/development_workflow.md`, `docs/zh-CN/development_workflow.md` | Repro flow is now milestone- and output-driven. | soft_redirect |
| `docs/FAQ.md` | `docs/README.en.md`, `docs/README.zh-CN.md` | Keep as supporting reference, not the main entry. | keep_in_place |
| `docs/decisions.md` | `docs/en/experiment_history.md`, `docs/zh-CN/experiment_history.md` | Current file only captures early decisions and conflicts with newer milestones. | archive_in_place |
| `docs/repo_review.md` | `docs/en/project_overview.md` | Keep as deep internal review; canonical status lives elsewhere. | keep_in_place |
| `docs/working_memory.md` | `docs/en/development_workflow.md` | Keep as operator quick memory, not public canonical overview. | keep_in_place |
| `docs/active_plan.md` | `docs/en/current_mainline_status.md` | Keep as active tracker; do not treat as authoritative repo state. | keep_in_place |
| `docs/local_inventory.md` | `docs/en/repo_map.md` | Keep as local host reference. | keep_in_place |
| `docs/provider_probe.md` | `docs/en/current_mainline_status.md` | Keep as provider reference with residual caveat context. | keep_in_place |
| `docs/task_notes/local_probe.md` | `docs/archive/README.md` | Historical task record. | archive_in_place |
| `docs/task_notes/v1.42_design.md` to `docs/task_notes/v1.60_design.md` | `docs/archive/README.md` | Milestone history should remain available without competing with canonical docs. | archive_in_place |
| `CHANGELOG.md` | `docs/en/experiment_history.md`, `docs/zh-CN/experiment_history.md` | Current changelog is incomplete relative to the milestone history. | archive_in_place |

## Encoding Recovery Notes

- The repository-wide Markdown and text scan did not expose a broad mojibake pattern in `README.md`, `CHANGELOG.md`, or `docs/**`.
- No canonical Markdown file needed destructive encoding recovery during v1.61.
- If later scans find ambiguous damage in historical docs, prefer conservative repair plus a note here rather than silent rewriting.
