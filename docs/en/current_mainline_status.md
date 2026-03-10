[中文对照](../zh-CN/current_mainline_status.md)

# Current Mainline Status

- Purpose: Record the current authoritative project state after the v1.60 evidence chain.
- Version scope: `v1.60`
- Reading order: this document -> experiment history -> generated evidence panels.
- Related docs: [experiment_history.md](./experiment_history.md), [project_overview.md](./project_overview.md)

## Mainline Status

- persistent memory mainline: `promoted`
- mainline admission closure: `partial_but_harder`
- sample contract status: `adequate`
- large-sample claim status: `supported`
- residual caveat: `provider_semantics_gap`

## Evidence Chain

Use the following generated artifacts in order:

1. `data/outputs/v160_persistent_memory_main_compare/compare/`
2. `data/outputs/v160_persistent_memory_main_compare/promotion_decision/`
3. `data/outputs/v160_mainline_cleanup/`
4. `data/outputs/v160_harder_sample_contract/`
5. `data/outputs/v160_mainline_admission_closure/`

## What Is Settled

- Persistent memory is stronger than the paired baseline on the current large-sample real main experiment.
- Promotion to mainline is valid.
- Large-sample wording is usable.

## What Is Not Fully Closed

- `mainline_admission_ready` is still `false`.
- The remaining gap is not the main memory signal itself; it is the cleanliness and semantics of provider-side evidence.
