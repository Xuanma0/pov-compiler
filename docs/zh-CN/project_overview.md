[English Version](../en/project_overview.md)

# 项目概览

- 文档目的：说明 POV Compiler 是什么、解决什么问题。
- 适用范围：`v1.60+`
- 推荐阅读顺序：当前主线状态 -> 实验演进史 -> 本文 -> 仓库地图。
- 相关文档：[current_mainline_status.md](./current_mainline_status.md)、[repo_map.md](./repo_map.md)、[experiment_history.md](./experiment_history.md)

## POV Compiler 做什么

POV Compiler 会把长时第一视角视频编译成可检索、可评测、可比较、可导出的分层产物，当前支持：

- 单视频离线编译
- 检索与 NLQ 风格评测
- 预算驱动与 compare 驱动的结果分析
- paper-ready 与 submission-ready 证据导出

## 当前技术主线

到 `v1.60` 为止，主线技术叙事是：

1. `YOLO26n` 提升了基础 object signal。
2. `YOLO26n + SAM3` 提升了 object persistence 与 chain grounding。
3. object-memory logic uplift 与 persistent object memory v2 进一步把这些 signal 变成更强的 memory 证据。
4. large-sample real main compare 证明 persistent memory 值得 promote 到 mainline。
5. mainline admission 仍未完全闭合，残余 caveat 主要来自 provider semantics。

## 这个仓库不是什么

- 它不是以 SDK 为核心的 polished package。
- 它不是单个评测脚本集合。
- 它不是通用视频感知框架。

更准确地说，它是一个强调 provenance 和 reporting 的 research / experiment workbench。
