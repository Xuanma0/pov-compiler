[中文对照](../zh-CN/experiment_history.md)

# Experiment History

- Purpose: Summarize the key milestone transitions that matter for the current mainline story.
- Version scope: `v1.54` to `v1.60`, with earlier context summarized only where needed.
- Reading order: current mainline status -> this document -> development workflow.
- Related docs: [current_mainline_status.md](./current_mainline_status.md), [project_overview.md](./project_overview.md)

## Milestones

### v1.54

- `core_real_v2_candidate` reduced weak query groups.
- But strict main gain still did not move enough on medium-scale real runs.
- Conclusion: query-bank-only changes were not sufficient.

### v1.55

- `YOLO26n + SAM3` improved object persistence, chain support, and query strength.
- Real uplift still pointed to object-memory logic as the next bottleneck.

### v1.57

- persistent object memory v2 improved reappearance support, lost-object support, chain grounding, and query strength.
- Recommendation became `promote_persistent_object_memory`.

### v1.58

- large-sample real main compare showed persistent memory improvements remained aligned under a paired contract.
- Promotion decision became `promote_persistent_memory_to_mainline`.

### v1.59

- Promotion and admission were separated conceptually.
- Cleanup explained why promotion could be valid while admission remained partial.

### v1.60

- Harder sample contract reached `adequate`.
- Large-sample wording reached `supported`.
- Admission closure improved to `partial_but_harder`.
- Residual caveat remained: provider semantics gap.
