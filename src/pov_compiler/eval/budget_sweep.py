from __future__ import annotations

from itertools import product
from typing import Any

from pov_compiler.eval.ablation import apply_variant
from pov_compiler.eval.metrics import evaluate_output, interval_duration, merge_intervals
from pov_compiler.schemas import DecisionPoint, KeyClip, Output, Token


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _token_priority(token: Token, highlights: list[KeyClip], decisions: list[DecisionPoint]) -> tuple[int, float]:
    if token.type == "HIGHLIGHT":
        base = 1000
    elif token.type.startswith("ATTENTION_"):
        base = 900
    elif token.type in ("EVENT_START", "EVENT_END"):
        base = 800
    elif token.type.startswith("MOTION_"):
        base = 400
    elif token.type == "SCENE_CHANGE":
        base = 300
    else:
        base = 200

    if any(_overlap(token.t0, token.t1, hl.t0, hl.t1) for hl in highlights):
        base += 80
    if any(_overlap(token.t0, token.t1, dp.t0, dp.t1) for dp in decisions):
        base += 60
    return base, float(token.conf)


def _select_highlights_by_budget(highlights: list[KeyClip], max_total_s: float) -> list[KeyClip]:
    if max_total_s <= 0:
        return []
    ranked = sorted(highlights, key=lambda hl: (float(hl.conf), float(hl.t1 - hl.t0)), reverse=True)
    selected: list[KeyClip] = []
    merged_intervals: list[tuple[float, float]] = []
    current = 0.0
    for hl in ranked:
        candidate = selected + [hl]
        merged = merge_intervals([(x.t0, x.t1) for x in candidate])
        duration = interval_duration(merged)
        if duration <= max_total_s + 1e-9:
            selected.append(hl)
            merged_intervals = merged
            current = duration
        elif current <= 0 and (hl.t1 - hl.t0) > 0:
            # keep at least one highlight if budget is tiny
            selected.append(hl)
            merged_intervals = merge_intervals([(hl.t0, min(hl.t1, hl.t0 + max_total_s))])
            break
    selected.sort(key=lambda x: (x.t0, x.t1))
    return selected


def apply_budget(output: Output, budget: dict[str, Any]) -> Output:
    cloned = output.model_copy(deep=True)
    duration_s = float(cloned.meta.get("duration_s", 0.0))

    max_total_s = float(budget.get("max_total_s", budget.get("budget_max_total_s", 1e12)))
    max_tokens = int(budget.get("max_tokens", budget.get("budget_max_tokens", 10**9)))
    max_decisions = int(budget.get("max_decisions", budget.get("budget_max_decisions", 10**9)))

    if cloned.highlights:
        cloned.highlights = _select_highlights_by_budget(cloned.highlights, max_total_s=max_total_s)

    highlight_ids = {hl.id for hl in cloned.highlights}
    highlight_intervals = [(hl.t0, hl.t1) for hl in cloned.highlights]

    if cloned.decision_points:
        filtered_decisions = [
            dp
            for dp in cloned.decision_points
            if (
                (dp.source_highlight is not None and dp.source_highlight in highlight_ids)
                or dp.source_highlight is None
                or any(_overlap(dp.t0, dp.t1, t0, t1) for t0, t1 in highlight_intervals)
            )
        ]
        filtered_decisions.sort(key=lambda dp: (1 if dp.source_highlight else 0, float(dp.conf)), reverse=True)
        cloned.decision_points = filtered_decisions[: max(0, max_decisions)]
        cloned.decision_points.sort(key=lambda dp: (dp.t, dp.t0, dp.t1))

    if cloned.token_codec.tokens:
        ranked = sorted(
            cloned.token_codec.tokens,
            key=lambda tok: (*_token_priority(tok, cloned.highlights, cloned.decision_points), float(tok.t0)),
            reverse=True,
        )
        cloned.token_codec.tokens = ranked[: max(0, max_tokens)]
        cloned.token_codec.tokens.sort(key=lambda tok: (tok.t0, tok.t1, tok.type))

    merged = merge_intervals([(hl.t0, hl.t1) for hl in cloned.highlights])
    kept_duration = interval_duration(merged)
    compression = float(duration_s / kept_duration) if kept_duration > 1e-9 else 0.0
    cloned.stats = dict(cloned.stats)
    cloned.stats["original_duration_s"] = duration_s
    cloned.stats["kept_duration_s"] = kept_duration
    cloned.stats["compression_ratio"] = compression
    cloned.stats["num_highlights"] = len(cloned.highlights)
    return cloned


def _grid_from_budgets(budgets: dict[str, Any]) -> list[dict[str, Any]]:
    max_total_s_list = [float(x) for x in budgets.get("max_total_s", [20, 40, 60])]
    max_tokens_list = [int(x) for x in budgets.get("max_tokens", [50, 100, 200])]
    max_decisions_list = [int(x) for x in budgets.get("max_decisions", [4, 8, 12])]

    grid: list[dict[str, Any]] = []
    for max_total_s, max_tokens, max_decisions in product(max_total_s_list, max_tokens_list, max_decisions_list):
        grid.append(
            {
                "max_total_s": float(max_total_s),
                "max_tokens": int(max_tokens),
                "max_decisions": int(max_decisions),
            }
        )
    grid.sort(key=lambda x: (x["max_total_s"], x["max_tokens"], x["max_decisions"]))
    return grid


def run_budget_sweep(
    output: Output,
    variant: str,
    budgets: dict[str, Any],
    eval_config: dict[str, Any] | None = None,
    retriever_config: dict[str, Any] | None = None,
    index_prefix: str | None = None,
) -> list[dict[str, Any]]:
    base = apply_variant(output, variant=variant)
    rows: list[dict[str, Any]] = []
    for budget in _grid_from_budgets(budgets):
        budgeted = apply_budget(base, budget=budget)
        metrics = evaluate_output(
            output=budgeted,
            eval_config=eval_config,
            retriever_config=retriever_config,
            index_prefix=index_prefix,
        )
        row: dict[str, Any] = {
            "video_id": output.video_id,
            "variant": variant,
            "budget_max_total_s": float(budget["max_total_s"]),
            "budget_max_tokens": int(budget["max_tokens"]),
            "budget_max_decisions": int(budget["max_decisions"]),
        }
        row.update(metrics)
        rows.append(row)
    return rows
