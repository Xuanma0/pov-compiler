# FAQ

## Q1) `perception-backend=real` fails on MediaPipe init.

- Ensure `mediapipe` is installed.
- Ensure `assets/mediapipe/hand_landmarker.task` exists.
- Run strict check:

```text
python scripts\perception_smoke.py --video "<YOUR_VIDEO_MP4>" --out_dir data\outputs\perception_check --run-perception --perception-backend real --perception-fps 5 --perception-max-frames 200 --perception-strict
```

## Q2) Why do `real` and `stub` sometimes show identical NLQ metrics?

Possible causes:
- retrieval/reranker path dominates and does not consume changed signals strongly enough
- compare set has limited sensitivity for selected query types
- metrics are budget-saturated (same selected subset)

Check:
- `top1_kind_*_rate`
- strict metrics (`hit_at_k_strict`, `top1_in_distractor_rate`)
- query-type slices (`hard_pseudo_anchor`, `hard_pseudo_token`, `hard_pseudo_decision`)

## Q3) Why does figure generation print budget fallback warning?

`make_paper_figures.py` uses nearest available budget when requested point is absent. This is expected behavior:

- requested: `40/100/8`
- fallback: nearest existing point, often `60/200/12`

The selected budget is written into `snapshot.json`.

## Q4) How to avoid re-running expensive stages?

Use `ego4d_smoke.py` default resume behavior (`--resume` is enabled by default). Completed stages with valid outputs are skipped.

## Q5) ffmpeg is missing. Can smoke still run?

Yes. Run with `--no-proxy` or rely on auto-skip behavior when ffmpeg is unavailable.

## Q6) Which flag is correct for macro average in figure script?

Both are supported:
- `--macro_avg`
- `--macro-avg`

## Q7) How to force strict real perception without fallback?

Use:

```text
--perception-backend real --perception-strict
```

This implies fail-fast behavior if dependencies or frame inference fail.
