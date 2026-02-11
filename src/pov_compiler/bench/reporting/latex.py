from __future__ import annotations

from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - exercised only when dependency is missing.
    pd = None


_VARIANT_ORDER = {
    "raw_events_only": 0,
    "highlights_only": 1,
    "highlights_plus_tokens": 2,
    "highlights_plus_decisions": 3,
    "full": 4,
}

_QUERY_ORDER = {
    "pseudo_time": 0,
    "pseudo_anchor": 1,
    "pseudo_hard_time": 2,
    "pseudo_token": 3,
    "pseudo_decision": 4,
    "time": 5,
    "anchor": 6,
    "hard_time": 7,
    "token": 8,
    "decision": 9,
}


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should be present in test/runtime env.
        raise ImportError("pandas is required for reporting. Install with `pip install pandas`.")
    return pd


def build_main_table(df: Any) -> Any:
    """Return a compact overall table for paper reporting."""

    lib = _require_pandas()
    if df is None or len(df) == 0:
        return lib.DataFrame(columns=["variant", "hit_at_k", "mrr", "coverage_ratio", "compression_ratio"])

    cols = [col for col in ("variant", "hit_at_k", "mrr", "coverage_ratio", "compression_ratio", "num_videos") if col in df.columns]
    out = df.loc[:, cols].copy()
    if "variant" in out.columns:
        out["__variant_order"] = out["variant"].map(_VARIANT_ORDER).fillna(999)
        out = out.sort_values(["__variant_order", "variant"]).drop(columns=["__variant_order"])
    return out.reset_index(drop=True)


def build_ablation_table(df: Any) -> Any:
    """Return query-type x variant hit@k table."""

    lib = _require_pandas()
    if df is None or len(df) == 0:
        return lib.DataFrame(columns=["query_type"])
    if "query_type" not in df.columns or "variant" not in df.columns or "hit_at_k" not in df.columns:
        return lib.DataFrame(columns=["query_type"])

    pivot = (
        df.pivot_table(index="query_type", columns="variant", values="hit_at_k", aggfunc="mean")
        .reset_index()
        .copy()
    )
    ordered_variants = [v for v in _VARIANT_ORDER.keys() if v in pivot.columns]
    other_cols = [c for c in pivot.columns if c not in {"query_type", *ordered_variants}]
    pivot = pivot[["query_type", *ordered_variants, *other_cols]]
    pivot["__query_order"] = pivot["query_type"].map(_QUERY_ORDER).fillna(999)
    pivot = pivot.sort_values(["__query_order", "query_type"]).drop(columns=["__query_order"])
    return pivot.reset_index(drop=True)


def _fmt_cell(value: Any, float_format: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        try:
            return float_format % value
        except Exception:
            return str(value)
    return str(value)


def df_to_markdown_table(df: Any, float_format: str = "%.4f") -> str:
    """Convert a DataFrame to a markdown table without extra dependencies."""

    lib = _require_pandas()
    if df is None or len(df) == 0:
        return "| empty |\n|---|\n| n/a |"

    columns = [str(col) for col in df.columns]
    lines: list[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in df.iterrows():
        values = [_fmt_cell(row[col], float_format=float_format) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def df_to_latex_table(
    df: Any,
    caption: str,
    label: str,
    float_format: str = "%.4f",
) -> str:
    """Convert DataFrame into a table environment with tabular body."""

    lib = _require_pandas()
    if df is None:
        df = lib.DataFrame()

    def _escape(text: str) -> str:
        escaped = str(text)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for src, dst in replacements.items():
            escaped = escaped.replace(src, dst)
        return escaped

    columns = [str(col) for col in df.columns]
    align = "l" + ("r" * max(0, len(columns) - 1))
    tab_lines: list[str] = []
    tab_lines.append(rf"\begin{{tabular}}{{{align}}}")
    tab_lines.append(r"\hline")
    tab_lines.append(" & ".join(_escape(col) for col in columns) + r" \\")
    tab_lines.append(r"\hline")
    for _, row in df.iterrows():
        values = [_fmt_cell(row[col], float_format=float_format) for col in df.columns]
        tab_lines.append(" & ".join(_escape(value) for value in values) + r" \\")
    tab_lines.append(r"\hline")
    tab_lines.append(r"\end{tabular}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        *tab_lines,
        rf"\caption{{{_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"
