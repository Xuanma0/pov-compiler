# Working Memory

## Core goal

Turn long first-person videos into compact layered artifacts for retrieval, NLQ evaluation, streaming-budget analysis, and BYE/paper-ready export. The canonical offline producer is `scripts/run_offline.py` -> `src/pov_compiler/pipeline.py:OfflinePipeline.run()`.

## Key entry points

- compile:
  `scripts/run_offline.py`
- index/retrieve/context:
  `scripts/build_index.py`, `scripts/retrieve.py`, `scripts/build_context.py`
- query/eval:
  `scripts/gen_queries.py`, `scripts/eval_cross.py`, `scripts/eval_nlq.py`
- batch/compare:
  `scripts/ego4d_smoke.py`, `scripts/run_ab_bye_compare.py`, `scripts/run_model_stack_compare.py`
- core contracts:
  `src/pov_compiler/schemas.py`, `src/pov_compiler/repository/schema.py`, `src/pov_compiler/repository/summary_schema.py`

## Do not break

- top-level `Output` fields:
  `events`, `events_v0`, `events_v1`, `highlights`, `token_codec`, `decision_points`, `decisions_model_v1`, `perception`, `object_memory_v0`, `repository`
- retrieval field conventions used across modules:
  place/object/chain hints, `repo_chunks`, `repo_trace`, strict metric columns
- output directory conventions documented in `docs/OUTPUTS.md`
- env-only secret handling in `src/pov_compiler/models/client.py`
- fake-provider execution path; it is part of normal testable behavior, not disposable scaffolding

## Common commands

```powershell
python scripts\run_offline.py --video data\raw_videos\demo.mp4 --out data\outputs\demo_v03_decisions.json
python scripts\build_index.py --video data\raw_videos\demo.mp4 --json data\outputs\demo_v03_decisions.json --out_prefix data\cache\demo
python scripts\retrieve.py --json data\outputs\demo_v03_decisions.json --index data\cache\demo --query "anchor=turn_head top_k=6 mode=highlights max_tokens=160" --out data\outputs\context_q.json
python scripts\build_context.py --json data\outputs\demo_v03_decisions.json --out data\outputs\context.json --mode highlights --max-highlights 8 --max-tokens 160
python scripts\gen_queries.py --json data\outputs\demo_v03_decisions.json --out data\outputs\queries.jsonl
python scripts\eval_cross.py --json data\outputs\demo_v03_decisions.json --queries data\outputs\queries.jsonl --out_dir data\outputs\eval_v05 --sweep
python scripts\eval_nlq.py --json data\outputs\demo_v03_decisions.json --index data\cache\demo --out_dir data\outputs\nlq_demo --mode hard_pseudo_nlq --seed 0 --top-k 6 --sweep
python scripts\test_fast.py
```

## Output artifacts

- main JSON:
  `data/outputs/<run>/json/<video_uid>_v03_decisions.json`
- vector index:
  `data/cache/<video_uid>.index.npz`
  `data/cache/<video_uid>.index_meta.json`
- eval:
  `data/outputs/<run>/eval/<video_uid>/...`
- NLQ:
  `data/outputs/<run>/nlq/<video_uid>/...`
- paper/export:
  compare directories plus `snapshot.json`

## Current biggest technical debt

- oversized orchestrators:
  `pipeline.py`, `retrieval/retriever.py`, `streaming/runner.py`, `scripts/eval_nlq.py`
- duplicated helper code:
  repeated `_load_yaml` and `_as_output` patterns across scripts
- implicit contracts:
  retrieval, repo, `events_v1`, object memory, and evaluation share field assumptions without one compatibility layer
- hygiene drift:
  version mismatch (`README`/docs say `0.2.0`; package says `0.1.0`), mojibake in `retrieval/query_planner.py`, checked-in `__pycache__`

## First-read reminder for future tasks

Before changing code, read `docs/repo_review.md` and identify:

- which artifact contract the task touches
- whether the task changes retrieval, repo, streaming, or evaluation behavior
- the smallest validation command that covers that path
