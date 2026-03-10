# Archive Notes

## Purpose

Document how historical repository docs are preserved without conflicting with the current canonical entry points.

## Archive Rules

- Prefer archive over delete.
- Keep historical design notes as version evidence even when they are no longer first-read docs.
- If an old file still has value but its conclusion is outdated, keep it and redirect readers to the canonical docs.
- Use `docs/doc_migration_map.md` to record path mapping, status, and migration reason.

## In-Place Archive Families

- `docs/task_notes/*.md`: milestone-by-milestone design and implementation records.
- `docs/decisions.md`: early ADR-style decisions; currently incomplete relative to the v1.60 state.
- `docs/ARCHITECTURE.md`, `docs/CLI.md`, `docs/OUTPUTS.md`, `docs/REPRO.md`: legacy/reference docs that predate the current canonical bilingual set.

## Current Canonical Entry

- English: [../README.en.md](../README.en.md)
- 中文: [../README.zh-CN.md](../README.zh-CN.md)
