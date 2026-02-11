# CLI Reference

All commands below are copy-paste ready for Windows CMD.

## 1) Single Video Pipeline

```text
cd /d D:\pov-compiler
python scripts\run_offline.py --video data\raw_videos\demo.mp4 --out data\outputs\demo_v03_decisions.json
```

With perception (stub):

```text
python scripts\run_offline.py --video data\raw_videos\demo.mp4 --out data\outputs\demo_v03_decisions.json --run-perception --perception-backend stub --perception-fps 5 --perception-max-frames 300
```

With perception (real strict):

```text
python scripts\run_offline.py --video data\raw_videos\demo.mp4 --out data\outputs\demo_v03_decisions.json --run-perception --perception-backend real --perception-strict --perception-fps 5 --perception-max-frames 300
```

## 2) Index + Retrieve + Context

```text
python scripts\build_index.py --video data\raw_videos\demo.mp4 --json data\outputs\demo_v03_decisions.json --out_prefix data\cache\demo
python scripts\retrieve.py --json data\outputs\demo_v03_decisions.json --index data\cache\demo --query "anchor=turn_head top_k=6 mode=highlights max_tokens=160" --out data\outputs\context_q.json
python scripts\build_context.py --json data\outputs\demo_v03_decisions.json --out data\outputs\context.json --mode highlights --max-highlights 8 --max-tokens 160
```

## 3) Fixed Queries + Cross Eval

```text
python scripts\gen_queries.py --json data\outputs\demo_v03_decisions.json --out data\outputs\queries.jsonl --seed 0 --n-time 10 --n-hard-time 10 --n-token 10 --n-decision 10
python scripts\eval_cross.py --json data\outputs\demo_v03_decisions.json --queries data\outputs\queries.jsonl --out_dir data\outputs\eval_v05 --sweep
```

## 4) NLQ Eval

```text
python scripts\eval_nlq.py --json data\outputs\demo_v03_decisions.json --index data\cache\demo --out_dir data\outputs\nlq_demo --mode hard_pseudo_nlq --seed 0 --top-k 6 --sweep
```

## 5) Ego4D Smoke (short-first)

```text
python scripts\ego4d_smoke.py --root "<YOUR_EGO4D_ROOT>" --out_dir data\outputs\ego4d_smoke_short --n 5 --seed 0 --no-proxy --prefer-short --max-duration-s 120 --probe-candidates 50 --run-perception --perception-backend stub --perception-fps 5 --perception-max-frames 300 --run-eval --run-nlq --nlq-mode hard_pseudo_nlq --no-sweep --jobs 1
```

Stratified duration bins:

```text
python scripts\ego4d_smoke.py --root "<YOUR_EGO4D_ROOT>" --out_dir data\outputs\ego4d_smoke_stratified --n 6 --seed 0 --no-proxy --stratified --duration-bins "30,60,180,600,1800" --prefer-short --probe-candidates 200 --run-perception --perception-backend real --perception-strict --perception-fps 5 --perception-max-frames 300 --run-eval --run-nlq --nlq-mode hard_pseudo_nlq --no-sweep --jobs 1
```

## 6) Dataset Inspection/Catalog

```text
python scripts\inspect_ego4d_dir.py --root "<YOUR_EGO4D_ROOT>" --depth 3 --out data\outputs\ego4d_dir_report.md
python scripts\catalog_videos.py --root "<YOUR_EGO4D_ROOT>" --out data\outputs\ego4d_catalog.csv --jobs 2
```

## 7) Perception Smoke

```text
python scripts\perception_smoke.py --video "<YOUR_VIDEO_MP4>" --out_dir data\outputs\perception_smoke_demo --run-perception --perception-backend real --perception-fps 5 --perception-max-frames 200 --perception-strict
```

## 8) Paper Figures

Single run:

```text
python scripts\make_paper_figures.py --cross_dir data\outputs\ego4d_smoke_short\eval --nlq_csv data\outputs\ego4d_smoke_short\nlq_summary_all.csv --out_dir data\outputs\paper_figs_short --macro_avg
```

Compare run (real vs stub):

```text
python scripts\make_paper_figures.py --cross_dir data\outputs\ego4d_ab_real_n6\eval --nlq_csv data\outputs\ego4d_ab_real_n6\nlq_summary_all.csv --out_dir data\outputs\paper_figs_ab_real_vs_stub_n6 --macro-avg --label real --compare_dir data\outputs\paper_figs_stub_n6 --compare_label stub
```

## 9) Reranker Sweep + Debug

```text
python scripts\sweep_reranker.py --json_dir data\outputs\ego4d_smoke_real\json --index_dir data\outputs\ego4d_smoke_real\cache --mode hard_pseudo_nlq --search random --trials 50 --metric objective_combo --seed 0 --out_dir data\outputs\rerank_sweep_v1
python scripts\debug_rerank.py --json data\outputs\demo_v03_decisions.json --index data\cache\demo --mode hard_pseudo_nlq --n 20 --seed 0 --out data\outputs\rerank_debug\debug.csv
```
