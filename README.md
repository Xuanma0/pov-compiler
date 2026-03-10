# POV Compiler

POV Compiler turns long first-person videos into layered artifacts for retrieval, NLQ-style evaluation, budget-aware analysis, and paper-ready reporting.

## Current Status

- Mainline memory route: `persistent_object_memory_v2` is promoted to mainline.
- Mainline admission closure: `partial_but_harder`.
- Sample contract: `adequate`.
- Large-sample claim: `supported`.
- Residual caveat: provider semantics gap remains, so promotion and admission must be read as different layers.

## Start Here

- English docs entry: [docs/README.en.md](docs/README.en.md)
- 中文文档入口: [docs/README.zh-CN.md](docs/README.zh-CN.md)
- Current mainline status:
  [docs/en/current_mainline_status.md](docs/en/current_mainline_status.md)
  /
  [docs/zh-CN/current_mainline_status.md](docs/zh-CN/current_mainline_status.md)
- Repository map:
  [docs/en/repo_map.md](docs/en/repo_map.md)
  /
  [docs/zh-CN/repo_map.md](docs/zh-CN/repo_map.md)
- Experiment history:
  [docs/en/experiment_history.md](docs/en/experiment_history.md)
  /
  [docs/zh-CN/experiment_history.md](docs/zh-CN/experiment_history.md)
- Development workflow:
  [docs/en/development_workflow.md](docs/en/development_workflow.md)
  /
  [docs/zh-CN/development_workflow.md](docs/zh-CN/development_workflow.md)
- Documentation inventory and migration map:
  [docs/doc_inventory.md](docs/doc_inventory.md)
  /
  [docs/doc_migration_map.md](docs/doc_migration_map.md)

## Primary Generated Evidence

- Mainline compare output:
  [compare_summary.json](data/outputs/v160_persistent_memory_main_compare/compare/compare_summary.json)
- Promotion decision:
  [snapshot.json](data/outputs/v160_persistent_memory_main_compare/promotion_decision/snapshot.json)
- Admission cleanup:
  [snapshot.json](data/outputs/v160_mainline_cleanup/snapshot.json)
- Harder sample contract:
  [snapshot.json](data/outputs/v160_harder_sample_contract/snapshot.json)
- Admission closure:
  [snapshot.json](data/outputs/v160_mainline_admission_closure/snapshot.json)
- Paper-ready package:
  [paper_ready](data/outputs/v160_mainline_cleanup/paper_ready)
- Submission pack:
  [submission_pack](data/outputs/v160_mainline_cleanup/submission_pack)

## Documentation Rules

- Canonical current docs live under `docs/en/` and `docs/zh-CN/`.
- Historical design notes stay under `docs/task_notes/` and are treated as archive evidence, not first-read entry points.
- Root-level `README.md` is the short entry page, not the full historical log.
- All text docs should be UTF-8 and follow [docs/doc_style_guide.md](docs/doc_style_guide.md).
