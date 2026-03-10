[English Version](../en/experiment_history.md)

# 实验演进史

- 文档目的：总结与当前主线叙事最相关的版本转折点。
- 适用范围：`v1.54` 到 `v1.60`，更早版本只做必要背景说明。
- 推荐阅读顺序：当前主线状态 -> 本文 -> 开发协作流程。
- 相关文档：[current_mainline_status.md](./current_mainline_status.md)、[project_overview.md](./project_overview.md)

## 关键里程碑

### v1.54

- `core_real_v2_candidate` 降低了 weak query groups。
- 但 medium-scale real run 上 strict main gain 仍未明显抬升。
- 结论：只改 query bank 不够。

### v1.55

- `YOLO26n + SAM3` 提升了 object persistence、chain support、query strength。
- 但 real uplift 仍指向 object-memory logic 是下一个瓶颈。

### v1.57

- persistent object memory v2 提升了 reappearance support、lost-object support、chain grounding、query strength。
- 推荐动作升级为 `promote_persistent_object_memory`。

### v1.58

- large-sample real main compare 证明 persistent memory 在 paired contract 下仍有稳定收益。
- promotion decision 升级为 `promote_persistent_memory_to_mainline`。

### v1.59

- promotion 与 admission 的概念被明确拆开。
- cleanup 解释了为什么 promotion 可以成立，但 admission 仍是 partial。

### v1.60

- harder sample contract 达到 `adequate`。
- large-sample wording 达到 `supported`。
- admission closure 推进到 `partial_but_harder`。
- 残余 caveat 仍然是 provider semantics gap。
