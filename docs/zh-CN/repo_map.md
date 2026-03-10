[English Version](../en/repo_map.md)

# 仓库地图

- 文档目的：给出源码、脚本、文档、输出产物的稳定顶层结构图。
- 适用范围：`v1.60+`
- 推荐阅读顺序：项目概览 -> 本文 -> 开发协作流程。
- 相关文档：[project_overview.md](./project_overview.md)、[development_workflow.md](./development_workflow.md)、[current_mainline_status.md](./current_mainline_status.md)

## 源码区域

- `src/pov_compiler/perception/`：感知后端、runner、object-memory logic
- `src/pov_compiler/retrieval/`：检索与 query planning
- `src/pov_compiler/streaming/`：streaming 与在线策略
- `src/pov_compiler/bench/reporting/`：compare、cleanup、decision、export 的 reporting helper
- `src/pov_compiler/models/`：provider-facing client 层

## 脚本

- `scripts/run_offline.py`：单视频离线编译
- `scripts/run_main_real_benchmark.py`：主 benchmark runner
- `scripts/export_paper_ready.py`：paper-ready panel 导出
- `scripts/export_submission_pack.py`：submission-ready archive 导出
- `scripts/report_*`：结果层报告脚本
- `scripts/check_docs_encoding.py`、`scripts/check_docs_integrity.py`：文档检查脚本

## 文档

- `README.md`：根目录短入口
- `docs/README.en.md`、`docs/README.zh-CN.md`：canonical docs 入口
- `docs/en/`、`docs/zh-CN/`：中英双语 canonical 文档
- `docs/task_notes/`：历史设计与实现记录
- `docs/archive/README.md`：归档规则

## 生成产物

- `data/outputs/v158_main_real_baseline/`
- `data/outputs/v158_main_real_persistent/`
- `data/outputs/v158_persistent_memory_main_compare/`
- `data/outputs/v159_mainline_cleanup/`
- `data/outputs/v160_mainline_cleanup/`

当前主线证据链主要是：

1. compare
2. decision
3. cleanup
4. harder sample contract
5. closure
