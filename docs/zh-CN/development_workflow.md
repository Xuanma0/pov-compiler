[English Version](../en/development_workflow.md)

# 开发协作流程

- 文档目的：说明你、我、Codex 在这个仓库里的典型协作方式。
- 适用范围：`v1.60+`
- 推荐阅读顺序：项目概览 -> 仓库地图 -> 本文。
- 相关文档：[project_overview.md](./project_overview.md)、[repo_map.md](./repo_map.md)、[glossary.md](./glossary.md)

## 协作模型

- 用户给出 milestone 目标与约束。
- Codex 先阅读相关 docs 和 outputs。
- 工作通常分成两个阶段：
  1. 设计审查
  2. 实现 + 测试 + 收口

## 典型里程碑流程

1. 先读 `working_memory.md`、`repo_review.md`、`active_plan.md` 和最新 task note。
2. 检查相关输出目录。
3. 先做 design review，明确 scope、风险、冻结文件、最小验证计划。
4. 只在允许的文件集合内实现。
5. 按 milestone prompt 中指定的命令做 smoke。
6. 更新 task note 与 `docs/active_plan.md`。
7. 只有在要求的验证全部通过后，才提交。

## 验证风格

- 默认全量验证：`python -m pytest -q -n auto`
- 然后执行 milestone 指定的 smoke 命令。
- 最后执行 `python scripts/security_scan_secrets.py`

## Scope Discipline

- 优先在结果层、报告层、导出层做改动。
- 除非 milestone 明确允许，不要把范围扩大到 retrieval、streaming、runtime 核心。
- 优先新增文件，而不是改核心 orchestrator。
