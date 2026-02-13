# POV -> BYE Offline Injection

Purpose: export existing POV Compiler output JSON into BYE-compatible `events/events_v1.jsonl` without depending on BYE repo code.

## CLI

```bash
python scripts/export_bye_events.py \
  --json data/outputs/<run>/json/<uid>_v03_decisions.json \
  --out_dir data/outputs/bye_inject_demo \
  --include events_v1,highlights,tokens,decisions
```

Optional flags:
- `--video_id <id>`: override detected video id.
- `--no-sort`: disable deterministic sorting.

## Output

- `events/events_v1.jsonl`: one JSON per line.
- `snapshot.json`: reproducibility metadata.

Each event line includes:
- `tsMs` (int)
- `category` (`scenario`)
- `name` (`pov.event` / `pov.highlight` / `pov.token` / `pov.decision`)
- `payload` with at least:
  - `video_id`
  - `t0_ms`
  - `t1_ms`
  - `source_kind`
  - `source_event`
  - `conf` (nullable)

Determinism:
- sorted by `(tsMs, name_priority, t0_ms, t1_ms, stable_index)` by default.
- JSONL lines are written with stable key ordering.

## v0 Minimal Closed Loop (with BYE repo)

If you have a local BYE repo, run:

```bash
python scripts/bye_regression_smoke.py \
  --pov_json data/outputs/<run>/json/<uid>_v03_decisions.json \
  --out_dir data/outputs/bye_regression_smoke/<uid> \
  --bye_root <path_to_bye_repo> \
  --bye-report <optional_override_report_script.py>
```

Optional:
- `--video <mp4>`: include video in run package.
- `--bye-lint / --bye-report / --bye-regression`: explicit entrypoint overrides.
- `--skip_regression`: run export + lint + report only.
- `--strict`: fail non-zero when BYE root/entrypoints/steps are missing or failing.

Output layout:
- `events/events_v1.jsonl`
- `run_package/` (minimal BYE package with events + manifest/metadata)
- `logs/` (stdout/stderr per BYE step attempt)
- `report/` (copied report artifacts when available)
- `bye_metrics.json` + `bye_metrics.csv` (normalized BYE-side metrics for aggregation)
- `snapshot.json` (full audit trail: args, BYE root, resolved entrypoints, return codes, artifact paths)

When BYE repo is missing:
- non-strict mode prints warning and still succeeds with export artifacts.
- strict mode exits non-zero.

## Batch Hook in ego4d_smoke

Enable per-video BYE integration in batch runs:

```bash
python scripts/ego4d_smoke.py \
  --root "<YOUR_EGO4D_ROOT>" \
  --out_dir data/outputs/ego4d_smoke_bye \
  --n 1 --no-proxy --run-bye \
  --bye-root <path_to_bye_repo> \
  --bye-video-mode none
```

Per UID output goes to `out_dir/bye/<uid>/...`, and `summary.csv` includes `bye_status`, `bye_report_rc`, `bye_regression_rc`, `bye_metrics_path`, and `bye_numeric_*` columns.

## Budget Sweep + A/B Compare

Quality-vs-budget curve for one run (strict UID matching by default):

```bash
python scripts/sweep_bye_budgets.py \
  --pov-json-dir data/outputs/<run>/json \
  --uids-file data/outputs/uids.txt \
  --out-dir data/outputs/<run>/bye_budget \
  --bye-root <path_to_bye_repo_or_fake_bye> \
  --budgets "20/50/4,40/100/8,60/200/12" \
  --primary-metric qualityScore
```

Compare two sweeps (e.g., stub vs real):

```bash
python scripts/compare_bye_budget_sweeps.py \
  --a_dir data/outputs/ab/run_stub/compare/bye_budget/stub \
  --b_dir data/outputs/ab/run_real/compare/bye_budget/real \
  --a_label stub --b_label real \
  --primary-metric qualityScore \
  --out_dir data/outputs/ab/compare/bye_budget/compare
```

Main compare artifacts:
- `tables/table_budget_compare.csv`
- `tables/table_budget_compare.md`
- `figures/fig_bye_primary_vs_budget_seconds_compare.(png/pdf)`
- `figures/fig_bye_primary_delta_vs_budget_seconds.(png/pdf)`
- `compare_summary.json`
