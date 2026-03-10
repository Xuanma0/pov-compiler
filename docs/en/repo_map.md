[中文对照](../zh-CN/repo_map.md)

# Repository Map

- Purpose: Provide a stable top-level map of source, scripts, docs, and outputs.
- Version scope: `v1.60+`
- Reading order: project overview -> this document -> development workflow.
- Related docs: [project_overview.md](./project_overview.md), [development_workflow.md](./development_workflow.md), [current_mainline_status.md](./current_mainline_status.md)

## Source Areas

- `src/pov_compiler/perception/`: perception backends, runner, object-memory logic
- `src/pov_compiler/retrieval/`: retrieval and query planning
- `src/pov_compiler/streaming/`: streaming and online policies
- `src/pov_compiler/bench/reporting/`: compare, cleanup, decision, and export reporting helpers
- `src/pov_compiler/models/`: provider-facing client layer

## Scripts

- `scripts/run_offline.py`: single-video offline compilation
- `scripts/run_main_real_benchmark.py`: main benchmark runner
- `scripts/export_paper_ready.py`: paper-ready panel export
- `scripts/export_submission_pack.py`: submission-ready archive export
- `scripts/report_*`: result-layer reports
- `scripts/check_docs_encoding.py`, `scripts/check_docs_integrity.py`: documentation checks

## Docs

- `README.md`: root short entry
- `docs/README.en.md`, `docs/README.zh-CN.md`: canonical docs entry
- `docs/en/`, `docs/zh-CN/`: canonical bilingual docs
- `docs/task_notes/`: historical design and implementation records
- `docs/archive/README.md`: archive rules

## Generated Outputs

- `data/outputs/v158_main_real_baseline/`
- `data/outputs/v158_main_real_persistent/`
- `data/outputs/v158_persistent_memory_main_compare/`
- `data/outputs/v159_mainline_cleanup/`
- `data/outputs/v160_mainline_cleanup/`

These output roots contain the current mainline evidence chain:

1. compare
2. decision
3. cleanup
4. harder sample contract
5. closure
