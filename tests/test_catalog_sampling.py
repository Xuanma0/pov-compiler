from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ego4d_smoke import choose_sample_entries, filter_entries_by_patterns, parse_duration_bins


def _entry(uid: str, duration_s: float, rel: str) -> dict:
    return {
        "video_uid": uid,
        "duration_s": duration_s,
        "relative_path": rel,
        "src_path": f"D:/Ego4D_Dataset/{rel}",
    }


def test_prefer_short_and_long_sampling() -> None:
    entries = [
        _entry("a", 12.0, "train/a.mp4"),
        _entry("b", 40.0, "train/b.mp4"),
        _entry("c", 75.0, "train/c.mp4"),
        _entry("d", 120.0, "train/d.mp4"),
    ]
    short = choose_sample_entries(entries, n=2, seed=0, prefer_short=True)
    assert [x["video_uid"] for x in short] == ["a", "b"]

    long = choose_sample_entries(entries, n=2, seed=0, prefer_long=True)
    assert [x["video_uid"] for x in long] == ["d", "c"]


def test_stratified_sampling_covers_multiple_bins() -> None:
    entries = [
        _entry("a", 10.0, "x/a.mp4"),   # <30
        _entry("b", 45.0, "x/b.mp4"),   # 30-60
        _entry("c", 100.0, "x/c.mp4"),  # 60-180
        _entry("d", 300.0, "x/d.mp4"),  # 180-600
        _entry("e", 1200.0, "x/e.mp4"), # 600-1800
    ]
    bins = parse_duration_bins("30,60,180,600,1800")
    picked = choose_sample_entries(entries, n=4, seed=0, stratified=True, duration_bins=bins)
    picked_durations = [float(x["duration_s"]) for x in picked]
    assert len(picked) == 4
    assert min(picked_durations) < 30
    assert max(picked_durations) > 180


def test_include_exclude_filtering() -> None:
    entries = [
        _entry("a", 10.0, "v2_packed/full_scale/train/a.mp4"),
        _entry("b", 20.0, "v2_packed/full_scale/test/b.mp4"),
        _entry("c", 30.0, "other/c.mp4"),
    ]
    filtered = filter_entries_by_patterns(
        entries,
        include_patterns=[r"full_scale"],
        exclude_patterns=[r"/test/"],
    )
    assert [x["video_uid"] for x in filtered] == ["a"]
