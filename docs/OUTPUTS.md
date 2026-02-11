# Outputs and Schemas

This document summarizes key output files and high-value fields.

## 1) Main Offline JSON

Default output path example:

```text
data/outputs/<run>/json/<video_uid>_v03_decisions.json
```

Top-level keys:

- `video_id`
- `meta` (`fps`, `duration_s`, `sample_fps`, ...)
- `events` (core event segmentation)
- `events_v0` (perception-aware event layer)
- `highlights` (decision-centric windows)
- `stats` (`kept_duration_s`, `compression_ratio`, ...)
- `token_codec` (`version`, `vocab`, `tokens`)
- `decision_points` (S-A-C-O + alternatives)
- `perception` (when enabled)
- `debug.signals`

### `events[]`

Each event contains:

- `id`, `t0`, `t1`
- `scores` (e.g., `boundary_conf`)
- `anchors[]` (`turn_head`, `stop_look`, `interaction_stub`)

### `highlights[]`

Each key clip contains:

- `id`, `t0`, `t1`
- `source_event`
- `anchor_type`, `anchor_t`
- `conf`, `meta`

### `token_codec.tokens[]`

Token fields:

- `id`, `t0`, `t1`, `type`, `conf`
- `source_event`
- `source`, `meta`

### `decision_points[]`

Decision fields:

- `id`, `t`, `t0`, `t1`
- `source_event`, `source_highlight`
- `trigger`
- `state` (motion before/after, nearby tokens/signals)
- `action`
- `constraints[]`
- `outcome`
- `alternatives[]`
- `conf`, `meta`

### `perception`

When perception is enabled:

- `meta` (backend, processed frames, throughput)
- `frames[]` (objects/hands/contact per sampled frame)
- `signals` (`time`, `visual_change`, `contact_score`)
- `summary` (`backend`, `deps_ok`, `fallback_used`, `frames_processed`, `contact_events_count`, `objects_topk`, ...)

## 2) Index Files

```text
data/cache/<video_uid>.index.npz
data/cache/<video_uid>.index_meta.json
```

- `index.npz`: vector matrix + ids
- `index_meta.json`: segment metadata (`kind`, `id`, `t0`, `t1`, `source_event`, anchor/token hints)

## 3) Cross Eval Outputs

```text
data/outputs/<run>/eval/<video_uid>/results_overall.csv
data/outputs/<run>/eval/<video_uid>/results_by_query_type.csv
data/outputs/<run>/eval/<video_uid>/results_per_query.csv
data/outputs/<run>/eval/<video_uid>/report.md
```

Required columns in `results_overall.csv`:

- `video_id`, `variant`
- `budget_max_total_s`, `budget_max_tokens`, `budget_max_decisions`
- `compression_ratio`, `coverage_ratio`
- `hit_at_k`, `mrr`
- `tokens_total`, `decisions_total`, `highlights_total`

## 4) NLQ Eval Outputs

```text
data/outputs/<run>/nlq/<video_uid>/nlq_results.csv
data/outputs/<run>/nlq/<video_uid>/nlq_summary.csv
data/outputs/<run>/nlq/<video_uid>/nlq_report.md
data/outputs/<run>/nlq_summary_all.csv
```

Key metrics:

- Standard: `hit_at_k`, `mrr`
- Strict: `hit_at_k_strict`, `hit_at_1_strict`
- Distractor-aware: `top1_in_distractor_rate` (or `fp_rate`)
- Top1 kind diagnostics: `top1_kind_highlight_rate`, `top1_kind_token_rate`, `top1_kind_decision_rate`, `top1_kind_event_rate`
- Grouping dims: `variant`, `query_type`, `duration_bucket`, budgets

## 5) Batch Smoke Root

```text
data/outputs/<run>/
  manifest.jsonl
  summary.csv
  summary.md
  nlq_summary_all.csv
  json/
  cache/
  eval/
  nlq/
  perception/
  event/
```

`manifest.jsonl` per-entry includes source path, uid, sampled/probed metadata, and stage status fields.

## 6) Paper Figure Outputs

```text
data/outputs/<fig_run>/figures/
data/outputs/<fig_run>/tables/
data/outputs/<fig_run>/snapshot.json
```

- Figures include budget curves, ablation, strict/fp, duration-bucket, optional compare deltas.
- Tables include `table_main`, `table_ablation`, and compare table when compare mode is enabled.
