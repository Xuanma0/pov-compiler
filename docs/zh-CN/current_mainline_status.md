[English Version](../en/current_mainline_status.md)

# 当前主线状态

- 文档目的：记录 v1.60 证据链之后的当前 authoritative 项目状态。
- 适用范围：`v1.60`
- 推荐阅读顺序：本文 -> 实验演进史 -> 生成产物面板。
- 相关文档：[experiment_history.md](./experiment_history.md)、[project_overview.md](./project_overview.md)

## 主线状态

- persistent memory mainline：`promoted`
- mainline admission closure：`partial_but_harder`
- sample contract status：`adequate`
- large-sample claim status：`supported`
- residual caveat：`provider_semantics_gap`

## 证据链

按以下顺序阅读生成产物：

1. `data/outputs/v160_persistent_memory_main_compare/compare/`
2. `data/outputs/v160_persistent_memory_main_compare/promotion_decision/`
3. `data/outputs/v160_mainline_cleanup/`
4. `data/outputs/v160_harder_sample_contract/`
5. `data/outputs/v160_mainline_admission_closure/`

## 已经确定的结论

- 在当前 large-sample real main experiment 上，persistent memory 明显优于 paired baseline。
- promotion to mainline 已成立。
- “large-sample” 这个表述已经可以使用。

## 仍未完全闭合的部分

- `mainline_admission_ready` 仍然是 `false`。
- 剩余 gap 不在 memory signal 本身，而在 provider 侧证据的 cleanliness / semantics。
