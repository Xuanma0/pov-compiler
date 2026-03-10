[English Version](./README.en.md)

# 文档入口

- 文档目的：作为仓库在 `v1.60` 状态下的中文 canonical 入口。
- 适用范围：`v1.60+` 文档整备基线。
- 推荐阅读顺序：
  1. [current_mainline_status.md](./zh-CN/current_mainline_status.md)
  2. [experiment_history.md](./zh-CN/experiment_history.md)
  3. [project_overview.md](./zh-CN/project_overview.md)
  4. [repo_map.md](./zh-CN/repo_map.md)
  5. [development_workflow.md](./zh-CN/development_workflow.md)
  6. [glossary.md](./zh-CN/glossary.md)
  7. [doc_inventory.md](./doc_inventory.md)
  8. [doc_migration_map.md](./doc_migration_map.md)
- 相关文档：
  [doc_style_guide.md](./doc_style_guide.md),
  [archive/README.md](./archive/README.md),
  [local_inventory.md](./local_inventory.md),
  [provider_probe.md](./provider_probe.md)

## 当前 canonical 文档

- 项目概览: [docs/zh-CN/project_overview.md](./zh-CN/project_overview.md)
- 仓库地图: [docs/zh-CN/repo_map.md](./zh-CN/repo_map.md)
- 实验演进史: [docs/zh-CN/experiment_history.md](./zh-CN/experiment_history.md)
- 当前主线状态: [docs/zh-CN/current_mainline_status.md](./zh-CN/current_mainline_status.md)
- 开发协作流程: [docs/zh-CN/development_workflow.md](./zh-CN/development_workflow.md)
- 术语表: [docs/zh-CN/glossary.md](./zh-CN/glossary.md)

## 生成产物阅读顺序

阅读导出证据时，固定按下面顺序：

1. `persistent_memory_main_compare/`
2. `persistent_memory_main_decision/`
3. `mainline_admission_cleanup/`
4. `harder_sample_contract/`
5. `mainline_admission_closure/`

最后再引用 canonical 的 paper-ready 表和图。

## 说明

- `docs/task_notes/` 是历史设计证据，不是主入口。
- `docs/ARCHITECTURE.md`、`docs/CLI.md`、`docs/OUTPUTS.md`、`docs/REPRO.md` 现在主要是 legacy/reference 文档，当前状态以 canonical 文档为准。
