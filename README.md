# pov-compiler v0.1 (offline)

Offline POV event pipeline for local MP4 videos:
- Stable event segmentation (`events[]`)
- Anchor mining (`turn_head`, `stop_look`, `interaction_stub`)
- Anchor suppression for `stop_look` (merge/filter/top-k per event)
- Decision-centric sampling (`highlights[]` + `stats`)
- Semantic Token Codec (`token_codec.version=0.2`)
- DecisionPoint compiler (`decision_points[]` with S-A-C-O + alternatives)
- Minimal Context Builder (budget-controlled `context.json`)
- Vector Index + Retrieve (query-driven selection for RAG/tool calls)
- Debug signal export (`time`, `motion_energy`, `embed_dist`, `boundary_score`)

## Install

```bash
pip install -r requirements.txt
```

Optional config file support:
- install `pyyaml` (if not already installed)

Optional CLIP plugin:
- install `torch`, `open_clip_torch`, and dependencies (e.g. `Pillow`)

## Run Offline Pipeline

```bash
python scripts/run_offline.py --video data/raw_videos/demo.mp4 --out data/outputs/demo.json
```

Common overrides:

```bash
python scripts/run_offline.py \
  --video data/raw_videos/demo.mp4 \
  --out data/outputs/demo.json \
  --sample-fps 4 \
  --thresh 0.65 \
  --min-event-s 3.0
```

Enable CLIP embedding (if optional deps are installed):

```bash
python scripts/run_offline.py --video data/raw_videos/demo.mp4 --out data/outputs/demo.json --use-clip
```

If CLIP dependencies are missing, pipeline automatically falls back to histogram embedding.

## Export Event Clips

```bash
python scripts/export_clips.py --video data/raw_videos/demo.mp4 --json data/outputs/demo.json --out_dir data/outputs/clips
```

Export highlights only:

```bash
python scripts/export_clips.py --video data/raw_videos/demo.mp4 --json data/outputs/demo.json --out_dir data/outputs/highlights --mode highlights
```

Export both events and highlights:

```bash
python scripts/export_clips.py --video data/raw_videos/demo.mp4 --json data/outputs/demo.json --out_dir data/outputs/clips --mode both
```

## Build Context JSON

```bash
python scripts/build_context.py \
  --json data/outputs/demo.json \
  --out data/outputs/context.json \
  --mode highlights \
  --max-highlights 8 \
  --max-tokens 160
```

`context.json` is the compact downstream input for Be-Your-Eyes / VLM / LLM stages.

Query-aware context (retrieve + budget in one step):

```bash
python scripts/build_context.py \
  --json data/outputs/demo_v02_token.json \
  --out data/outputs/context_q.json \
  --index data/cache/demo \
  --query "time=110-140 top_k=6 mode=highlights max_tokens=160"
```

Decision-focused context:

```bash
python scripts/build_context.py \
  --json data/outputs/demo_v02_token.json \
  --out data/outputs/context_decisions.json \
  --mode decisions \
  --max-decisions 10
```

## Build Vector Index

```bash
python scripts/build_index.py \
  --video data/raw_videos/demo.mp4 \
  --json data/outputs/demo_v02_token.json \
  --out_prefix data/cache/demo
```

Output files:
- `data/cache/demo.index.npz`
- `data/cache/demo.index_meta.json`

## Retrieve

```bash
python scripts/retrieve.py \
  --json data/outputs/demo_v02_token.json \
  --index data/cache/demo \
  --query "anchor=turn_head top_k=6 mode=highlights max_tokens=160" \
  --out data/outputs/context_q.json
```

Query syntax examples:
- `time=120-150 top_k=6`
- `token=HIGHLIGHT,ATTENTION_STOP_LOOK top_k=8`
- `anchor=stop_look,turn_head`
- `event=event_0003`
- `decision=ATTENTION_TURN_HEAD top_k=6`
- `text=door top_k=6` (requires CLIP image index + open_clip text encoder)

## Ego4D Local Smoke Test

If Ego4D is already unpacked at `D:\Ego4D_Dataset`, run:

```bash
python scripts/ego4d_smoke.py --root "D:\Ego4D_Dataset" --out_dir "data/outputs/ego4d_smoke" --n 5 --seed 0 --proxy --run-eval --run-nlq --nlq-mode pseudo_nlq --sweep --jobs 1
```

Behavior:
- recursively scans `*.mp4` under `--root` (default min size: 10MB)
- samples `N` videos with fixed seed
- if `ffmpeg` exists and proxy is enabled, generates `540p` `h264` no-audio proxy files
- supports parallel execution with `--jobs`
- supports resume with `--resume` (default true), skipping completed stages
- runs per video:
  - `run_offline.py`
  - `build_index.py`
  - `gen_queries.py`
  - `eval_cross.py`
  - `eval_nlq.py`
- writes:
  - `manifest.jsonl`
  - `json/`, `cache/`, `eval/`, `nlq/` per-video outputs
  - aggregate `summary.csv` and `summary.md`
  - aggregate `nlq_summary_all.csv`

Notes:
- if `ffmpeg` is unavailable, proxy is automatically skipped and source mp4 is used
- source files are never deleted unless you pass `--delete-original`
- NLQ args:
  - `--run-nlq/--no-run-nlq`
  - `--nlq-mode {mock,pseudo_nlq,hard_pseudo_nlq,ego4d}`
  - `--nlq-ann <path>` (required for `ego4d` mode)
  - `--nlq-sweep/--no-nlq-sweep`
  - `--nlq-n --nlq-seed --nlq-topk`

Hard pseudo-NLQ (no label leakage, with distractor negatives):

```bash
python scripts/eval_nlq.py \
  --json data/outputs/demo_v03_decisions.json \
  --index data/cache/demo \
  --out_dir data/outputs/nlq_hard \
  --mode hard_pseudo_nlq \
  --seed 0 \
  --sweep
```

Notes:
- `hard_pseudo_nlq` queries are natural-language templates without explicit token/decision labels.
- hard mode defaults to `allow_gt_fallback=false` (no GT-time fallback).
- summary rows include `duration_bucket` (`short/medium/long/very_long`) for bucketed analysis.

Inspect unpacked directory tree and MP4 distribution:

```bash
python scripts/inspect_ego4d_dir.py --root "D:\Ego4D_Dataset" --depth 3 --out data/outputs/ego4d_dir_report.md
```

Build metadata catalog (`duration/fps/resolution`) for faster filtering:

```bash
python scripts/catalog_videos.py --root "D:\Ego4D_Dataset" --out data/outputs/ego4d_catalog.csv --limit 0 --jobs 2
```

Short-video-first smoke example:

```bash
python scripts/ego4d_smoke.py \
  --root "D:\Ego4D_Dataset" \
  --out_dir "data/outputs/ego4d_smoke_short" \
  --n 5 \
  --seed 0 \
  --no-proxy \
  --run-eval \
  --run-nlq \
  --nlq-mode pseudo_nlq \
  --prefer-short \
  --max-duration-s 120 \
  --probe-candidates 50 \
  --jobs 1
```

Optional sampling controls in smoke:
- `--catalog <csv|jsonl>`: reuse prebuilt catalog instead of probing durations at runtime
- `--include <regex>` / `--exclude <regex>` (repeatable)
- `--stratified --duration-bins "30,60,180,600,1800"`
- `--prefer-short` / `--prefer-long`

## Eval (Fixed Queries + Cross-Variant)

Generate a fixed query set from `full` output first:

```bash
python scripts/gen_queries.py \
  --json data/outputs/demo_v03_decisions.json \
  --out data/outputs/queries.jsonl \
  --seed 0 \
  --n_time 10 \
  --n_hard_time 10 \
  --n_token 10 \
  --n_decision 10
```

Cross-variant evaluation on the same query set:

```bash
python scripts/eval_cross.py \
  --json data/outputs/demo_v03_decisions.json \
  --queries data/outputs/queries.jsonl \
  --out_dir data/outputs/eval_v05 \
  --sweep
```

Outputs:
- `data/outputs/eval_v05/results_overall.csv`
- `data/outputs/eval_v05/results_by_query_type.csv`
- `data/outputs/eval_v05/results_per_query.csv`
- `data/outputs/eval_v05/report.md`

Query types in fixed set:
- `time`
- `anchor`
- `token`
- `decision`
- `hard_time` (low overlap with highlights, intended to stress structural retrieval)

Batch suite (directory mode) now defaults to cross-variant fixed-query evaluation:

```bash
python scripts/eval_suite.py --json_dir data/outputs --pattern "*_v03_decisions.json" --out data/outputs/results_overall.csv --sweep
```

Suite outputs:
- `results_overall.csv`
- `results_by_query_type.csv`
- `results_per_query.csv`
- `report.md`

`results_overall.csv` core columns:
- `video_id`
- `variant`
- `budget_max_total_s`
- `budget_max_tokens`
- `budget_max_decisions`
- `compression_ratio`
- `coverage_ratio`
- `hit_at_k`
- `mrr`
- `tokens_total`
- `decisions_total`
- `highlights_total`

`results_by_query_type.csv` adds:
- `query_type`
- per-target retrieval metrics (`hit_at_k_event`, `hit_at_k_decision`, `hit_at_k_token`)

`report.md` includes:
- overall variant comparison
- by-query-type comparison table
- explicit deltas on `token`, `decision`, and `hard_time` query slices

## Paper Figures (Matplotlib)

Generate paper-ready figures/tables from smoke outputs:

```bash
python scripts/make_paper_figures.py \
  --cross_dir data/outputs/ego4d_smoke_real/eval \
  --nlq_csv data/outputs/ego4d_smoke_real/nlq_summary_all.csv \
  --out_dir data/outputs/paper_figs_v01 \
  --budget "max_total_s=40,max_tokens=100,max_decisions=8" \
  --macro_avg \
  --formats "png,pdf"
```

Outputs:
- `figures/fig_budget_hitk_vs_seconds.(png/pdf)`
- `figures/fig_budget_mrr_vs_seconds.(png/pdf)`
- `figures/fig_ablation_overall.(png/pdf)`
- `figures/fig_ablation_by_query_type.(png/pdf)`
- `figures/fig_coverage_compression.(png/pdf)`
- `figures/fig_by_duration_bucket_hitk.(png/pdf)`
- `tables/table_main.tex`, `tables/table_ablation.tex`
- `tables/table_main.md`, `tables/table_ablation.md`
- `snapshot.json` (inputs, commit hash, time, params)

Recommended experiment protocol for paper plots:
- short-first subset: `--prefer-short --max-duration-s 120 --n 5`
- medium subset: `--min-duration-s 180 --max-duration-s 600 --n 5`
- long subset: `--min-duration-s 1800 --n 2`
- run figure generation per subset, then merge macro tables for the final main result

## Output JSON Shape

```json
{
  "video_id": "demo",
  "meta": {"fps": 30.0, "duration_s": 120.0, "sample_fps": 4.0},
  "events": [
    {
      "id": "event_0001",
      "t0": 0.0,
      "t1": 8.0,
      "scores": {"boundary_conf": 0.77},
      "anchors": [
        {"type": "turn_head", "t": 2.0, "conf": 0.85, "meta": {}}
      ]
    }
  ],
  "highlights": [
    {
      "id": "hl_0001",
      "t0": 10.0,
      "t1": 14.0,
      "source_event": "event_0002",
      "anchor_type": "turn_head",
      "anchor_t": 12.0,
      "conf": 0.81,
      "meta": {"anchor_types": ["turn_head"], "merged_count": 1}
    }
  ],
  "stats": {
    "original_duration_s": 120.0,
    "kept_duration_s": 42.0,
    "compression_ratio": 2.8571,
    "num_highlights": 9
  },
  "token_codec": {
    "version": "0.2",
    "vocab": ["EVENT_START", "EVENT_END", "MOTION_MOVING", "..."],
    "tokens": [
      {
        "id": "tok_000001",
        "t0": 0.0,
        "t1": 0.0,
        "type": "EVENT_START",
        "conf": 0.77,
        "source_event": "event_0001",
        "source": {"event_id": "event_0001"},
        "meta": {}
      }
    ]
  },
  "decision_points": [
    {
      "id": "dp_000001",
      "t": 12.0,
      "t0": 10.0,
      "t1": 14.0,
      "source_event": "event_0002",
      "source_highlight": "hl_0003",
      "trigger": {"anchor_type": "turn_head", "conf": 0.82},
      "state": {"motion_state_before": "STILL", "motion_state_after": "MOVING"},
      "action": {"type": "ATTENTION_TURN_HEAD", "conf": 0.79},
      "constraints": [{"type": "STABILITY_CONSTRAINT", "score": 0.71}],
      "outcome": {"type": "SCENE_CHANGED", "conf": 0.76},
      "alternatives": [
        {"action_type": "LOOK_FORWARD_ONLY", "risk_delta": 0.15},
        {"action_type": "TURN_HEAD_OPPOSITE", "risk_delta": 0.05}
      ],
      "conf": 0.78
    }
  ],
  "debug": {
    "signals": {
      "time": [],
      "motion_energy": [],
      "embed_dist": [],
      "boundary_score": []
    }
  }
}
```

## Notes

- Sampling is done with `VideoReader.iter_samples(sample_fps=4)` by default.
- `embed_dist`:
  - Default: lightweight histogram embedding (no torch).
  - Optional: CLIP embedding plugin via `--use-clip`.
- `motion_energy`:
  - Preferred: Farneback optical flow mean magnitude.
  - Fallback: mean absolute grayscale frame difference.
- `boundary_score = 0.6 * embed_dist_norm + 0.4 * motion_change_norm`.
- stop_look suppression defaults:
  - `max_stop_look_per_event = 3`
  - `min_gap_s = 2.0`
  - `stop_look_min_conf = 0.6`
- highlights defaults:
  - `pre_s = 2.0`, `post_s = 2.0`
  - `max_total_s = 60.0`
  - priority: `interaction > turn_head > stop_look`
- token codec defaults:
  - `attention_pre_s = 0.5`, `attention_post_s = 0.5`
  - `motion_min_run_s = 0.8`
  - `motion_max_runs_per_event = 4`
  - `scene_change_top_k = 3`
- index defaults:
  - `max_frames_per_segment = 24`
  - backend: `numpy` brute-force cosine, optional `faiss` acceleration if installed
- retrieval defaults:
  - `default_top_k = 8`
  - `prefer = highlight`
  - text query requires `torch + open_clip_torch` and CLIP image embeddings
- eval defaults:
  - `num_time_queries = 10`
  - `time_window_s = 8`
  - `default_top_k = 6`
  - fixed queries:
    - `seed = 0`
    - `n_time = 10`
    - `n_anchor = 6`
    - `n_token = 10`
    - `n_decision = 10`
    - `n_hard_time = 10`
    - `hard_overlap_thresh = 0.05`
  - budgets:
    - `max_total_s = [20, 40, 60]`
    - `max_tokens = [50, 100, 200]`
    - `max_decisions = [4, 8, 12]`
- decision defaults:
  - `pre_s = 2.0`, `post_s = 2.0`
  - `merge_iou = 0.7`
  - `min_gap_s = 0.0`
- context defaults:
  - mode: `highlights`
  - `max_events = 8`
  - `max_highlights = 10`
  - `max_decisions = 12`
  - `max_tokens = 200`

## Test

```bash
pytest -q
```
