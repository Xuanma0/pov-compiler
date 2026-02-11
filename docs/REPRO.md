# Reproducibility Guide

This guide gives two reproducible tracks:

1. Stub pipeline (fast baseline)
2. Real perception pipeline (YOLO26 + MediaPipe Tasks)

Then compares both in paper figures.

## 0) Environment

```text
cd /d D:\pov-compiler
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -U ultralytics mediapipe
```

## 1) Prepare Local Models (Only If Missing)

```text
if not exist weights mkdir weights
if not exist weights\yolo26n.pt python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
if not exist weights\yolo26s.pt python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"
if not exist assets\mediapipe\hand_landmarker.task python -c "import urllib.request, pathlib; p=pathlib.Path('assets/mediapipe/hand_landmarker.task'); p.parent.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', str(p)); print(p)"
```

## 2) Run Stub Baseline (n=6)

```text
python scripts\ego4d_smoke.py --root "<YOUR_EGO4D_ROOT>" --out_dir data\outputs\ego4d_ab_stub_n6 --n 6 --seed 0 --no-proxy --stratified --duration-bins "30,60,180,600,1800" --prefer-short --probe-candidates 200 --run-perception --perception-backend stub --perception-fps 5 --perception-max-frames 300 --run-eval --run-nlq --nlq-mode hard_pseudo_nlq --no-sweep --jobs 1
```

## 3) Run Real Strict (same sampling policy)

```text
python scripts\ego4d_smoke.py --root "<YOUR_EGO4D_ROOT>" --out_dir data\outputs\ego4d_ab_real_n6 --n 6 --seed 0 --no-proxy --stratified --duration-bins "30,60,180,600,1800" --prefer-short --probe-candidates 200 --run-perception --perception-backend real --perception-fps 5 --perception-max-frames 300 --perception-strict --run-eval --run-nlq --nlq-mode hard_pseudo_nlq --no-sweep --jobs 1
```

## 4) Generate Per-Run Figures

```text
python scripts\make_paper_figures.py --cross_dir data\outputs\ego4d_ab_stub_n6\eval --nlq_csv data\outputs\ego4d_ab_stub_n6\nlq_summary_all.csv --out_dir data\outputs\paper_figs_stub_n6 --macro_avg
python scripts\make_paper_figures.py --cross_dir data\outputs\ego4d_ab_real_n6\eval --nlq_csv data\outputs\ego4d_ab_real_n6\nlq_summary_all.csv --out_dir data\outputs\paper_figs_real_n6 --macro_avg
```

## 5) Generate Compare Figures (real vs stub)

```text
python scripts\make_paper_figures.py --cross_dir data\outputs\ego4d_ab_real_n6\eval --nlq_csv data\outputs\ego4d_ab_real_n6\nlq_summary_all.csv --out_dir data\outputs\paper_figs_ab_real_vs_stub_n6 --macro-avg --label real --compare_dir data\outputs\paper_figs_stub_n6 --compare_label stub
```

Main compare artifacts:

- `tables/table_compare.md`
- `tables/table_compare.tex`
- `figures/fig_compare_delta_strict.png/.pdf`
- `figures/fig_compare_by_query_type.png/.pdf`
- `figures/fig_compare_by_duration_bucket.png/.pdf`

## 6) Validation

```text
python -m pytest -q
```

## 7) Repro Notes

- Keep `--seed` fixed for sampling determinism.
- Keep budget controls fixed when comparing runs.
- Prefer using `compare_dir` from a prior figure run to reuse aligned metadata.
- Do not commit `data/outputs/` artifacts.
