[English Version](../en/glossary.md)

# 术语表

- 文档目的：统一项目核心术语的中英对应关系。
- 适用范围：`v1.60+`
- 推荐阅读顺序：当前主线状态 -> 需要时查本文。
- 相关文档：[current_mainline_status.md](./current_mainline_status.md)、[development_workflow.md](./development_workflow.md)

| 英文术语 | 中文 | 含义 |
|---|---|---|
| persistent memory | 持久记忆 | 当前主线采用的 persistent object-memory 路线。 |
| mainline admission | 主线准入 | 证据是否足够干净、足够完整，从而可以把某条路线视为当前主线。 |
| sample contract | 样本契约 | 冻结的 paired-sample 证据，包括 UID 集、budget、query bank、provider/perception 对齐。 |
| coverage contract | 覆盖率契约 | 用于支撑样本强度和 wording 的 coverage 阈值。 |
| provider semantics gap | provider 语义缺口 | provider 侧 availability semantics 残余歧义；它不推翻 promotion，但会影响 admission cleanliness。 |
| paper_ready | 论文导出版 | 供论文撰写使用的统一 report/table/figure 导出。 |
| submission_pack | 投稿归档包 | 用于评审或交接的自描述证据包。 |
| promotion | 提升/晋升结论 | 相对比较意义上的“应替代 baseline”。 |
| admission | 准入结论 | 证据完整度意义上的“是否足够干净地成为主线”。 |
| partial_but_harder | 更硬但仍未完全闭合 | 证据比之前更强，但还没有完全 clean enough 的状态。 |
