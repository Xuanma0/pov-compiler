from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - exercised only when dependency is missing.
    pd = None


_BUDGET_COL_MAP = {
    "max_total_s": "budget_max_total_s",
    "max_tokens": "budget_max_tokens",
    "max_decisions": "budget_max_decisions",
}


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should be present in test/runtime env.
        raise ImportError("pandas is required for reporting. Install with `pip install pandas`.")
    return pd


def _group_reduce(df: Any, group_cols: list[str], metric_cols: list[str], agg: str) -> Any:
    lib = _require_pandas()
    work = df.copy()
    marker = "__all__"
    used_cols = list(group_cols)
    if not used_cols:
        work[marker] = "all"
        used_cols = [marker]
    grouped = work.groupby(used_cols, dropna=False)[metric_cols]
    agg_name = str(agg).lower()
    if agg_name == "sum":
        out = grouped.sum().reset_index()
    elif agg_name == "median":
        out = grouped.median().reset_index()
    elif agg_name == "max":
        out = grouped.max().reset_index()
    elif agg_name == "min":
        out = grouped.min().reset_index()
    else:
        out = grouped.mean().reset_index()
    if marker in out.columns:
        out = out.drop(columns=[marker])
    return out


def _group_nunique(df: Any, group_cols: list[str], value_col: str, out_col: str) -> Any:
    lib = _require_pandas()
    work = df.copy()
    marker = "__all__"
    used_cols = list(group_cols)
    if not used_cols:
        work[marker] = "all"
        used_cols = [marker]
    out = work.groupby(used_cols, dropna=False)[value_col].nunique().reset_index(name=out_col)
    if marker in out.columns:
        out = out.drop(columns=[marker])
    return out


def _to_numeric(df: Any, cols: list[str]) -> Any:
    lib = _require_pandas()
    work = df.copy()
    for col in cols:
        if col in work.columns:
            work[col] = lib.to_numeric(work[col], errors="coerce")
    return work


def load_csvs(paths: list[Path]) -> Any:
    """Load multiple CSV files into a single DataFrame.

    Missing/unreadable files are skipped.
    """

    lib = _require_pandas()
    frames: list[Any] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            frame = lib.read_csv(path)
        except Exception:
            continue
        if "video_uid" not in frame.columns:
            if "video_id" in frame.columns:
                frame["video_uid"] = frame["video_id"]
            else:
                frame["video_uid"] = path.parent.name
        frame["__source_path"] = str(path)
        frames.append(frame)

    if not frames:
        return lib.DataFrame()
    return lib.concat(frames, ignore_index=True, sort=False)


def aggregate_over_videos(
    df: Any,
    group_cols: list[str],
    metric_cols: list[str],
    agg: str = "mean",
) -> Any:
    """Macro-average metrics over videos (equal video weight)."""

    lib = _require_pandas()
    if df is None or len(df) == 0:
        return lib.DataFrame(columns=[*group_cols, *metric_cols, "num_videos"])

    work = df.copy()
    if "video_uid" not in work.columns:
        if "video_id" in work.columns:
            work["video_uid"] = work["video_id"]
        else:
            work["video_uid"] = "video_0000"

    use_metrics = [col for col in metric_cols if col in work.columns]
    if not use_metrics:
        return lib.DataFrame(columns=[*group_cols, "num_videos"])
    work = _to_numeric(work, use_metrics)

    per_video = _group_reduce(work, [*group_cols, "video_uid"], use_metrics, agg)
    if "video_uid" in group_cols:
        return per_video

    macro = _group_reduce(per_video, group_cols, use_metrics, agg)
    count_df = _group_nunique(per_video, group_cols, "video_uid", "num_videos")
    if group_cols:
        macro = macro.merge(count_df, on=group_cols, how="left")
    else:
        num_videos = int(count_df["num_videos"].iloc[0]) if len(count_df) > 0 else 0
        macro["num_videos"] = num_videos
    return macro


def pick_budget_slice(df: Any, budget: dict[str, Any]) -> Any:
    """Filter rows at a given budget point with numeric tolerance."""

    lib = _require_pandas()
    if df is None or len(df) == 0 or not budget:
        return lib.DataFrame() if df is None else df.copy()

    out = df.copy()
    mask = lib.Series([True] * len(out))

    for key, raw_target in budget.items():
        col = key if key in out.columns else _BUDGET_COL_MAP.get(key, key)
        if col not in out.columns:
            continue
        series = lib.to_numeric(out[col], errors="coerce")
        target = lib.to_numeric(lib.Series([raw_target]), errors="coerce").iloc[0]
        if lib.isna(target):
            mask = mask & (out[col].astype(str) == str(raw_target))
            continue
        target_val = float(target)
        tol = max(1e-6, abs(target_val) * 1e-6)
        mask = mask & ((series - target_val).abs() <= tol)

    return out.loc[mask].reset_index(drop=True)


def compute_deltas(
    df: Any,
    baseline_variant: str = "highlights_only",
    target_variants: list[str] | None = None,
) -> Any:
    """Compute variant deltas against a baseline for hit@k and mrr."""

    lib = _require_pandas()
    targets = target_variants or ["highlights_plus_tokens", "highlights_plus_decisions", "full"]
    if df is None or len(df) == 0:
        return lib.DataFrame(
            columns=[
                "variant",
                "baseline_variant",
                "delta_hit_at_k",
                "delta_hit@k",
                "delta_mrr",
            ]
        )

    work = df.copy()
    for col in ("hit_at_k", "mrr"):
        if col not in work.columns:
            work[col] = 0.0
    work = _to_numeric(work, ["hit_at_k", "mrr"])
    if "variant" not in work.columns:
        return lib.DataFrame()

    key_cols = [
        col
        for col in (
            "video_uid",
            "video_id",
            "query_type",
            "budget_max_total_s",
            "budget_max_tokens",
            "budget_max_decisions",
        )
        if col in work.columns
    ]

    baseline = work.loc[work["variant"] == str(baseline_variant), [*key_cols, "hit_at_k", "mrr"]]
    if baseline.empty:
        return lib.DataFrame()
    baseline = baseline.groupby(key_cols, dropna=False)[["hit_at_k", "mrr"]].mean().reset_index()
    baseline = baseline.rename(columns={"hit_at_k": "baseline_hit_at_k", "mrr": "baseline_mrr"})

    target = work.loc[work["variant"].isin([str(v) for v in targets]), [*key_cols, "variant", "hit_at_k", "mrr"]]
    if target.empty:
        return lib.DataFrame()
    target = target.groupby([*key_cols, "variant"], dropna=False)[["hit_at_k", "mrr"]].mean().reset_index()
    target = target.rename(columns={"hit_at_k": "target_hit_at_k", "mrr": "target_mrr"})

    merged = target.merge(baseline, on=key_cols, how="left")
    merged["baseline_variant"] = str(baseline_variant)
    merged["delta_hit_at_k"] = merged["target_hit_at_k"] - merged["baseline_hit_at_k"]
    merged["delta_hit@k"] = merged["delta_hit_at_k"]
    merged["delta_mrr"] = merged["target_mrr"] - merged["baseline_mrr"]

    sort_cols = [col for col in [*key_cols, "variant"] if col in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged

