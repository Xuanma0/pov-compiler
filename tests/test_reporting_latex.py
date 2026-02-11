from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.latex import df_to_latex_table


def test_df_to_latex_table_contains_expected_sections() -> None:
    df = pd.DataFrame(
        [
            {"variant": "highlights_only", "hit_at_k": 0.42, "mrr": 0.30},
            {"variant": "full", "hit_at_k": 0.63, "mrr": 0.50},
        ]
    )
    latex = df_to_latex_table(
        df,
        caption="Main comparison table.",
        label="tab:main",
        float_format="%.3f",
    )
    assert "\\begin{table}" in latex
    assert "\\caption{Main comparison table.}" in latex
    assert "\\label{tab:main}" in latex
    assert "variant" in latex
    assert "hit\\_at\\_k" in latex
