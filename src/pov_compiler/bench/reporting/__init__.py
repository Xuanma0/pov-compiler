from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "aggregate_over_videos": ("pov_compiler.bench.reporting.aggregate", "aggregate_over_videos"),
    "compute_deltas": ("pov_compiler.bench.reporting.aggregate", "compute_deltas"),
    "load_csvs": ("pov_compiler.bench.reporting.aggregate", "load_csvs"),
    "pick_budget_slice": ("pov_compiler.bench.reporting.aggregate", "pick_budget_slice"),
    "build_ablation_table": ("pov_compiler.bench.reporting.latex", "build_ablation_table"),
    "build_main_table": ("pov_compiler.bench.reporting.latex", "build_main_table"),
    "df_to_latex_table": ("pov_compiler.bench.reporting.latex", "df_to_latex_table"),
    "df_to_markdown_table": ("pov_compiler.bench.reporting.latex", "df_to_markdown_table"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
