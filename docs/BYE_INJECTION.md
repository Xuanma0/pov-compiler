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
  --bye_root <path_to_bye_repo>
```

Optional:
- `--video <mp4>`: include video in run package.
- `--skip_regression`: run export + lint + report only.
- `--strict`: fail non-zero when BYE root/entrypoints/steps are missing or failing.

Output layout:
- `events/events_v1.jsonl`
- `run_package/` (minimal BYE package with events + manifest/metadata)
- `logs/` (stdout/stderr per BYE step attempt)
- `report/` (copied report artifacts when available)
- `snapshot.json` (full audit trail: args, BYE root, resolved entrypoints, return codes, artifact paths)

When BYE repo is missing:
- non-strict mode prints warning and still succeeds with export artifacts.
- strict mode exits non-zero.
