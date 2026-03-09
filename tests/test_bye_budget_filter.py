from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.integrations.bye.budget_filter import Budget, apply_budget


def _evt(name: str, t0_ms: int, t1_ms: int, ts_ms: int | None = None) -> dict[str, object]:
    return {
        "tsMs": int(ts_ms if ts_ms is not None else t0_ms),
        "category": "scenario",
        "name": name,
        "payload": {
            "video_id": "uid",
            "t0_ms": int(t0_ms),
            "t1_ms": int(t1_ms),
            "source_kind": "x",
            "source_event": "",
            "conf": None,
        },
    }


def test_apply_budget_keeps_base_events_and_filters_context() -> None:
    rows = [
        _evt("pov.event", 0, 10000),
        _evt("pov.highlight", 0, 4000),
        _evt("pov.token", 1000, 2000),
        _evt("pov.token", 3000, 3500),
        _evt("pov.decision", 5000, 6000),
        _evt("pov.decision", 7000, 8000),
    ]
    kept, stats = apply_budget(rows, Budget(max_total_s=10.0, max_tokens=1, max_decisions=1))
    names = [str(x.get("name", "")) for x in kept]
    assert names.count("pov.event") == 1
    assert names.count("pov.token") == 1
    assert names.count("pov.decision") == 1
    assert names.count("pov.highlight") == 1
    assert int(stats["before_total"]) == 6
    assert int(stats["after_total"]) == 4
    assert stats["after_counts"]["pov.event"] == 1


def test_apply_budget_respects_max_total_s_union() -> None:
    rows = [
        _evt("pov.event", 0, 20000),
        _evt("pov.highlight", 0, 3000),
        _evt("pov.token", 3000, 6000),
        _evt("pov.decision", 6000, 9000),
    ]
    kept, stats = apply_budget(rows, Budget(max_total_s=5.0, max_tokens=10, max_decisions=10))
    names = [str(x.get("name", "")) for x in kept]
    # First highlight is kept, subsequent context windows exceed 5s union budget.
    assert names.count("pov.highlight") == 1
    assert names.count("pov.token") == 0
    assert names.count("pov.decision") == 0
    assert float(stats["kept_duration_s"]) <= 5.0 + 1e-9
    assert float(stats["compression_ratio"]) >= 1.0

