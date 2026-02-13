from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.utils.fs import choose_indices, list_mp4_files, make_video_uid, to_posix_relative, write_jsonl
from pov_compiler.utils.media import probe_video_metadata
from pov_compiler.utils.subprocesses import ffprobe_readable, has_command, run_command


MIN_SIZE_DEFAULT = 10 * 1024 * 1024
DEFAULT_SUMMARY_BUDGET = {"max_total_s": 40, "max_tokens": 100, "max_decisions": 8}


def scan_and_plan(
    root: Path,
    n: int,
    seed: int,
    min_size_bytes: int = MIN_SIZE_DEFAULT,
) -> list[dict[str, Any]]:
    root = Path(root)
    files = list_mp4_files(root=root, min_size_bytes=min_size_bytes)
    entries: list[dict[str, Any]] = []
    uid_counts: dict[str, int] = {}

    for path in files:
        uid = make_video_uid(path=path, root=root)
        uid_counts[uid] = uid_counts.get(uid, 0) + 1
        if uid_counts[uid] > 1:
            uid = f"{uid}_{uid_counts[uid]}"
        rel = to_posix_relative(path, root)
        entries.append(
            {
                "video_uid": uid,
                "src_path": str(path.resolve()),
                "relative_path": rel,
                "folder_group": str(Path(rel).parent.as_posix()),
                "size_bytes": int(path.stat().st_size),
                "duration_s": None,
                "fps": None,
                "width": None,
                "height": None,
                "probed": False,
                "candidate": True,
                "proxy_path": None,
                "chosen": False,
                "pipeline_json_path": None,
                "index_prefix": None,
                "cross_eval_dir": None,
                "nlq_eval_dir": None,
                "perception_dir": None,
                "event_dir": None,
                "status_stage": "scanned",
            }
        )

    chosen_idxs = choose_indices(total=len(entries), n=n, seed=seed)
    for idx, entry in enumerate(entries):
        entry["chosen"] = idx in chosen_idxs
    return entries


def _parse_bool_auto_args(parser: argparse.ArgumentParser, name: str, default: bool | None, help_text: str) -> None:
    underscore = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", f"--{underscore}", dest=underscore, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        f"--no_{underscore}",
        dest=underscore,
        action="store_false",
        help=f"Disable {name}",
    )
    parser.set_defaults(**{underscore: default})


def _compile_patterns(patterns: list[str] | None) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns or []:
        pattern = str(pattern).strip()
        if not pattern:
            continue
        compiled.append(re.compile(pattern, flags=re.IGNORECASE))
    return compiled


def filter_entries_by_patterns(
    entries: list[dict[str, Any]],
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
) -> list[dict[str, Any]]:
    includes = _compile_patterns(include_patterns)
    excludes = _compile_patterns(exclude_patterns)
    out: list[dict[str, Any]] = []
    for entry in entries:
        target = f"{entry.get('relative_path', '')} {entry.get('src_path', '')}"
        if includes and not any(p.search(target) for p in includes):
            entry["candidate"] = False
            continue
        if excludes and any(p.search(target) for p in excludes):
            entry["candidate"] = False
            continue
        entry["candidate"] = True
        out.append(entry)
    return out


def parse_duration_bins(raw: str | None) -> list[float]:
    if not raw:
        return [30.0, 60.0, 180.0, 600.0, 1800.0]
    out: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except Exception:
            continue
        if value > 0:
            out.append(value)
    out = sorted(set(out))
    return out if out else [30.0, 60.0, 180.0, 600.0, 1800.0]


def _duration_bucket(duration: float | None, bins: list[float]) -> int:
    if duration is None:
        return len(bins) + 1
    for i, bound in enumerate(bins):
        if duration < bound:
            return i
    return len(bins)


def choose_sample_entries(
    candidates: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
    prefer_short: bool = False,
    prefer_long: bool = False,
    stratified: bool = False,
    duration_bins: list[float] | None = None,
    min_duration_s: float | None = None,
    max_duration_s: float | None = None,
) -> list[dict[str, Any]]:
    n = max(0, int(n))
    if n == 0 or not candidates:
        return []

    filtered: list[dict[str, Any]] = []
    for entry in candidates:
        duration = entry.get("duration_s")
        if duration is not None:
            try:
                duration = float(duration)
            except Exception:
                duration = None
        if min_duration_s is not None or max_duration_s is not None:
            if duration is None:
                continue
            if min_duration_s is not None and duration < float(min_duration_s):
                continue
            if max_duration_s is not None and duration > float(max_duration_s):
                continue
        filtered.append(entry)

    if not filtered:
        return []

    # Deterministic tie-break.
    def _stable_key(entry: dict[str, Any]) -> str:
        raw = f"{entry.get('video_uid','')}|{int(seed)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    ordered = sorted(filtered, key=_stable_key)

    if stratified:
        bins = duration_bins or [30.0, 60.0, 180.0, 600.0, 1800.0]
        buckets: dict[int, list[dict[str, Any]]] = {}
        for entry in ordered:
            duration = entry.get("duration_s")
            duration_value = float(duration) if isinstance(duration, (int, float)) else None
            bid = _duration_bucket(duration_value, bins)
            buckets.setdefault(bid, []).append(entry)

        bucket_ids = sorted(buckets.keys())
        if prefer_long:
            bucket_ids = sorted(bucket_ids, reverse=True)
        selected: list[dict[str, Any]] = []
        cursor: dict[int, int] = {bid: 0 for bid in bucket_ids}
        while len(selected) < n:
            progressed = False
            for bid in bucket_ids:
                bucket = buckets.get(bid, [])
                i = cursor.get(bid, 0)
                if i >= len(bucket):
                    continue
                selected.append(bucket[i])
                cursor[bid] = i + 1
                progressed = True
                if len(selected) >= n:
                    break
            if not progressed:
                break
        return selected

    if prefer_short or prefer_long:
        def _sort_key(entry: dict[str, Any]) -> tuple[int, float, str]:
            duration = entry.get("duration_s")
            if isinstance(duration, (int, float)):
                return 0, float(duration), str(entry.get("video_uid", ""))
            return 1, float("inf"), str(entry.get("video_uid", ""))

        sorted_entries = sorted(ordered, key=_sort_key)
        if prefer_long:
            known = [e for e in sorted_entries if isinstance(e.get("duration_s"), (int, float))]
            unknown = [e for e in sorted_entries if not isinstance(e.get("duration_s"), (int, float))]
            sorted_entries = list(reversed(known)) + unknown
        return sorted_entries[:n]

    picks = choose_indices(total=len(filtered), n=min(n, len(filtered)), seed=int(seed))
    return [entry for i, entry in enumerate(filtered) if i in picks]


def _to_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number:
        return None
    return number


def _to_int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return catalog

    def _norm(value: str) -> str:
        try:
            return str(Path(value).resolve()).lower()
        except Exception:
            return str(value).strip().lower()

    def _put(row: dict[str, Any]) -> None:
        p = str(row.get("path", row.get("src_path", ""))).strip()
        if p:
            catalog[_norm(p)] = row
        rel = str(row.get("relative_path", "")).strip().replace("\\", "/")
        if rel:
            catalog[rel.lower()] = row

    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                _put(payload)
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                _put(dict(row))
    return catalog


def enrich_candidates_with_metadata(
    candidates: list[dict[str, Any]],
    *,
    catalog_map: dict[str, dict[str, Any]] | None,
    probe_candidates: int,
    seed: int,
    jobs: int,
) -> int:
    catalog_map = catalog_map or {}

    def _match_catalog(entry: dict[str, Any]) -> dict[str, Any] | None:
        src_key = str(Path(str(entry["src_path"])).resolve()).lower()
        rel_key = str(entry.get("relative_path", "")).lower()
        return catalog_map.get(src_key) or catalog_map.get(rel_key)

    pending: list[dict[str, Any]] = []
    for entry in candidates:
        hit = _match_catalog(entry)
        if hit is not None:
            entry["duration_s"] = _to_float_or_none(hit.get("duration_s"))
            entry["fps"] = _to_float_or_none(hit.get("fps"))
            entry["width"] = _to_int_or_none(hit.get("width"))
            entry["height"] = _to_int_or_none(hit.get("height"))
            entry["probed"] = False
        else:
            pending.append(entry)

    if not pending:
        return 0
    k = int(probe_candidates)
    if k <= 0 or k >= len(pending):
        probe_set = pending
    else:
        picks = choose_indices(total=len(pending), n=k, seed=int(seed))
        probe_set = [entry for i, entry in enumerate(pending) if i in picks]

    if not probe_set:
        return 0

    probed_count = 0
    max_workers = max(1, min(int(jobs), len(probe_set)))
    if max_workers <= 1:
        for entry in probe_set:
            meta = probe_video_metadata(Path(str(entry["src_path"])))
            entry["duration_s"] = _to_float_or_none(meta.get("duration_s"))
            entry["fps"] = _to_float_or_none(meta.get("fps"))
            entry["width"] = _to_int_or_none(meta.get("width"))
            entry["height"] = _to_int_or_none(meta.get("height"))
            entry["probed"] = True
            probed_count += 1
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(probe_video_metadata, Path(str(entry["src_path"]))): entry
                for entry in probe_set
            }
            for fut in as_completed(future_map):
                entry = future_map[fut]
                try:
                    meta = fut.result()
                except Exception:
                    meta = {}
                entry["duration_s"] = _to_float_or_none(meta.get("duration_s"))
                entry["fps"] = _to_float_or_none(meta.get("fps"))
                entry["width"] = _to_int_or_none(meta.get("width"))
                entry["height"] = _to_int_or_none(meta.get("height"))
                entry["probed"] = True
                probed_count += 1
    return probed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ego4D local smoke test runner for POV Compiler")
    parser.add_argument("--root", default=r"D:\Ego4D_Dataset", help="Ego4D root directory")
    parser.add_argument("--out_dir", default="data/outputs/ego4d_smoke", help="Output directory")
    parser.add_argument("--n", type=int, default=5, help="Number of sampled videos")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (default 1)")
    _parse_bool_auto_args(parser, "resume", default=True, help_text="Skip completed stages if outputs are valid")

    _parse_bool_auto_args(parser, "proxy", default=None, help_text="Enable proxy transcoding if ffmpeg exists")
    parser.add_argument("--proxy_height", type=int, default=540, help="Proxy height")
    parser.add_argument("--crf", type=int, default=28, help="H264 CRF for proxy")
    parser.add_argument("--preset", default="veryfast", help="H264 preset for proxy")
    parser.add_argument(
        "--delete-original",
        dest="delete_original",
        action="store_true",
        help="Delete source only after proxy+ffprobe success",
    )

    _parse_bool_auto_args(parser, "run-eval", default=True, help_text="Run eval (gen_queries + eval_cross)")
    _parse_bool_auto_args(parser, "sweep", default=True, help_text="Run budget sweep in eval_cross")

    _parse_bool_auto_args(parser, "run-nlq", default=True, help_text="Run NLQ evaluation")
    parser.add_argument("--nlq-mode", choices=["mock", "pseudo_nlq", "hard_pseudo_nlq", "ego4d"], default="pseudo_nlq")
    parser.add_argument("--nlq-ann", default=None, help="Annotation path for NLQ mode=ego4d")
    _parse_bool_auto_args(parser, "nlq-sweep", default=True, help_text="Run budget sweep in NLQ eval")
    parser.add_argument("--nlq-n", type=int, default=10, help="NLQ query count knob")
    parser.add_argument("--nlq-seed", type=int, default=0, help="NLQ query seed")
    parser.add_argument("--nlq-topk", type=int, default=6, help="NLQ query top_k")
    _parse_bool_auto_args(parser, "run-perception", default=False, help_text="Run Perception v0 stage")
    parser.add_argument("--perception-fps", type=float, default=10.0, help="Perception sample fps")
    parser.add_argument("--perception-max-frames", type=int, default=300, help="Perception max frames per video")
    parser.add_argument("--perception-backend", choices=["stub", "real"], default="stub", help="Perception backend")
    _parse_bool_auto_args(
        parser,
        "perception-fallback-stub",
        default=True,
        help_text="Allow fallback from real backend to stub backend",
    )
    parser.add_argument(
        "--perception-strict",
        action="store_true",
        help="Strict perception mode: no fallback, fail on missing deps/frame errors",
    )

    parser.add_argument("--min-size-bytes", "--min_size_bytes", dest="min_size_bytes", type=int, default=MIN_SIZE_DEFAULT)
    parser.add_argument("--min-duration-s", type=float, default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--prefer-short", action="store_true")
    parser.add_argument("--prefer-long", action="store_true")
    parser.add_argument("--duration-bins", default="30,60,180,600,1800")
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--probe-candidates", type=int, default=50)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--summary-max-total-s", type=float, default=float(DEFAULT_SUMMARY_BUDGET["max_total_s"]))
    parser.add_argument("--summary-max-tokens", type=int, default=int(DEFAULT_SUMMARY_BUDGET["max_tokens"]))
    parser.add_argument("--summary-max-decisions", type=int, default=int(DEFAULT_SUMMARY_BUDGET["max_decisions"]))
    return parser.parse_args()


def _run_python_script(script_path: Path, args: list[str], cwd: Path) -> tuple[bool, str]:
    cmd = [sys.executable, str(script_path), *[str(x) for x in args]]
    result = run_command(cmd, cwd=cwd, check=False)
    output = (result.stdout or "") + (("\n" + result.stderr.strip()) if result.stderr.strip() else "")
    return result.returncode == 0, output.strip()


def _build_proxy(
    src_path: Path,
    proxy_path: Path,
    proxy_height: int,
    crf: int,
    preset: str,
) -> tuple[bool, str]:
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vf",
        f"scale=-2:{int(proxy_height)}",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(proxy_path),
    ]
    result = run_command(cmd, check=False)
    ok = result.returncode == 0 and proxy_path.exists() and proxy_path.stat().st_size > 0
    msg = (result.stdout or "") + (("\n" + result.stderr.strip()) if result.stderr.strip() else "")
    return ok, msg.strip()


def _extract_markdown_section(md_text: str, heading: str) -> str:
    lines = md_text.splitlines()
    collected: list[str] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if stripped == heading:
            capture = True
            collected.append(line)
            continue
        if capture and stripped.startswith("## "):
            break
        if capture:
            collected.append(line)
    return "\n".join(collected).strip()


def _select_default_budget_row(
    rows: list[dict[str, str]],
    budget_total_s: float,
    budget_tokens: int,
    budget_decisions: int,
) -> dict[str, str] | None:
    target: dict[str, str] | None = None
    for row in rows:
        if str(row.get("variant", "")) != "full":
            continue
        try:
            if (
                float(row.get("budget_max_total_s", "nan")) == float(budget_total_s)
                and int(float(row.get("budget_max_tokens", "nan"))) == int(budget_tokens)
                and int(float(row.get("budget_max_decisions", "nan"))) == int(budget_decisions)
            ):
                return row
        except Exception:
            pass
        if target is None:
            target = row
    return target


def _is_pipeline_json_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    for key in ("video_id", "events", "highlights", "decision_points"):
        if key not in payload:
            return False
    has_tokens = isinstance(payload.get("tokens"), list)
    if not has_tokens:
        token_codec = payload.get("token_codec", {})
        has_tokens = isinstance(token_codec, dict) and isinstance(token_codec.get("tokens"), list)
    return has_tokens


def _is_index_complete(index_prefix: Path) -> bool:
    return Path(f"{index_prefix}.index.npz").exists() and Path(f"{index_prefix}.index_meta.json").exists()


def _is_queries_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return False
    return len(lines) > 0


def _is_cross_eval_complete(eval_dir: Path) -> bool:
    return (eval_dir / "report.md").exists()


def _is_nlq_complete(nlq_dir: Path) -> bool:
    return (nlq_dir / "nlq_report.md").exists()


def _is_perception_complete(perception_dir: Path) -> bool:
    return (
        (perception_dir / "perception.json").exists()
        and (perception_dir / "events_v0.json").exists()
        and (perception_dir / "report.md").exists()
    )


def plan_stage_actions(
    *,
    json_path: Path,
    index_prefix: Path,
    queries_path: Path,
    eval_dir: Path,
    nlq_dir: Path,
    run_eval: bool,
    run_nlq: bool,
    resume: bool,
    perception_dir: Path | None = None,
    run_perception: bool = False,
) -> dict[str, bool]:
    actions = {
        "run_offline": True,
        "build_index": True,
        "gen_queries": bool(run_eval),
        "eval_cross": bool(run_eval),
        "eval_nlq": bool(run_nlq),
    }
    if bool(run_perception):
        actions["run_perception"] = True
    if not resume:
        return actions

    if _is_pipeline_json_complete(json_path):
        actions["run_offline"] = False
    if bool(run_perception) and perception_dir is not None and _is_perception_complete(perception_dir):
        actions["run_perception"] = False
    if _is_index_complete(index_prefix):
        actions["build_index"] = False
    if run_eval:
        if _is_cross_eval_complete(eval_dir):
            actions["gen_queries"] = False
            actions["eval_cross"] = False
        else:
            if _is_queries_complete(queries_path):
                actions["gen_queries"] = False
    if run_nlq and _is_nlq_complete(nlq_dir):
        actions["eval_nlq"] = False

    # Dependency propagation.
    if actions["run_offline"]:
        if bool(run_perception) and "run_perception" in actions:
            actions["run_perception"] = True
        actions["build_index"] = True
        if run_eval:
            actions["gen_queries"] = True
            actions["eval_cross"] = True
        if run_nlq:
            actions["eval_nlq"] = True
    if bool(run_perception) and actions.get("run_perception", False) and actions.get("run_offline", False):
        actions["run_perception"] = True
    elif actions["build_index"]:
        if run_nlq:
            actions["eval_nlq"] = True
    if run_eval and actions["eval_cross"] and not _is_queries_complete(queries_path):
        actions["gen_queries"] = True
    return actions


def _proxy_one(
    entry: dict[str, Any],
    *,
    proxy_dir: Path,
    proxy_height: int,
    crf: int,
    preset: str,
    delete_original: bool,
) -> dict[str, Any]:
    src_path = Path(str(entry["src_path"]))
    uid = str(entry["video_uid"])
    proxy_path = proxy_dir / f"{uid}.mp4"
    ok, msg = _build_proxy(
        src_path=src_path,
        proxy_path=proxy_path,
        proxy_height=proxy_height,
        crf=crf,
        preset=preset,
    )
    out = dict(entry)
    if ok:
        out["proxy_path"] = str(proxy_path.resolve())
        out["status_stage"] = "proxy"
        if delete_original:
            if ffprobe_readable(proxy_path):
                try:
                    src_path.unlink()
                    out["deleted_original"] = True
                except OSError:
                    out["deleted_original"] = False
            else:
                out["deleted_original"] = False
    else:
        out["proxy_path"] = None
        out["proxy_error"] = msg[-500:] if msg else "ffmpeg_failed"
    return out


def _process_video(
    entry: dict[str, Any],
    *,
    json_dir: Path,
    cache_dir: Path,
    eval_root_dir: Path,
    nlq_root_dir: Path,
    perception_root_dir: Path,
    event_root_dir: Path,
    seed: int,
    run_perception: bool,
    perception_fps: float,
    perception_max_frames: int,
    perception_backend: str,
    perception_fallback_stub: bool,
    perception_strict: bool,
    perception_cache_root: Path,
    run_eval: bool,
    sweep: bool,
    run_nlq: bool,
    nlq_mode: str,
    nlq_ann: str | None,
    nlq_sweep: bool,
    nlq_n: int,
    nlq_seed: int,
    nlq_topk: int,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scripts_dir = ROOT / "scripts"
    run_offline_script = scripts_dir / "run_offline.py"
    build_index_script = scripts_dir / "build_index.py"
    gen_queries_script = scripts_dir / "gen_queries.py"
    eval_cross_script = scripts_dir / "eval_cross.py"
    eval_nlq_script = scripts_dir / "eval_nlq.py"
    perception_smoke_script = scripts_dir / "perception_smoke.py"

    uid = str(entry["video_uid"])
    src_path = Path(str(entry["src_path"]))
    proxy_path = Path(str(entry["proxy_path"])) if entry.get("proxy_path") else None
    input_path = proxy_path if proxy_path is not None else src_path
    json_path = json_dir / f"{uid}_v03_decisions.json"
    index_prefix = cache_dir / uid
    eval_dir = eval_root_dir / uid
    nlq_dir = nlq_root_dir / uid
    perception_dir = perception_root_dir / uid
    event_dir = event_root_dir / uid
    perception_cache_dir = perception_cache_root / uid
    eval_dir.mkdir(parents=True, exist_ok=True)
    nlq_dir.mkdir(parents=True, exist_ok=True)
    perception_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    queries_path = eval_dir / "queries.jsonl"

    actions = plan_stage_actions(
        json_path=json_path,
        index_prefix=index_prefix,
        queries_path=queries_path,
        eval_dir=eval_dir,
        nlq_dir=nlq_dir,
        perception_dir=perception_dir,
        run_perception=run_perception,
        run_eval=run_eval,
        run_nlq=run_nlq,
        resume=resume,
    )

    stage_results = {stage: ("run" if do else "skip") for stage, do in actions.items()}
    status = "ok"
    status_stage = "planned"
    last_error = ""

    def _fail(stage_name: str, log: str) -> None:
        nonlocal status, status_stage, last_error
        status = f"failed_{stage_name}"
        status_stage = stage_name
        last_error = log

    if actions["run_offline"]:
        ok, log = _run_python_script(
            run_offline_script,
            ["--video", str(input_path), "--out", str(json_path)],
            cwd=ROOT,
        )
        if not ok:
            _fail("run_offline", log)
        else:
            stage_results["run_offline"] = "done"
            status_stage = "run_offline"
    else:
        stage_results["run_offline"] = "skipped"

    if status == "ok" and actions["build_index"]:
        ok, log = _run_python_script(
            build_index_script,
            ["--video", str(input_path), "--json", str(json_path), "--out_prefix", str(index_prefix)],
            cwd=ROOT,
        )
        if not ok:
            _fail("build_index", log)
        else:
            stage_results["build_index"] = "done"
            status_stage = "build_index"
    elif status == "ok":
        stage_results["build_index"] = "skipped"

    perception_json_path = perception_dir / "perception.json"
    perception_report_path = perception_dir / "report.md"
    event_json_path = perception_dir / "events_v0.json"
    if status == "ok" and run_perception and actions.get("run_perception", False):
        effective_fallback = bool(perception_fallback_stub) and not bool(perception_strict)
        perception_args = [
            "--video",
            str(input_path),
            "--out_dir",
            str(perception_dir),
            "--backend",
            str(perception_backend),
            "--fps",
            str(float(perception_fps)),
            "--max-frames",
            str(int(perception_max_frames)),
            "--cache-dir",
            str(perception_cache_dir),
            "--perception-fallback-stub" if effective_fallback else "--no-perception-fallback-stub",
        ]
        if bool(perception_strict):
            perception_args.append("--perception-strict")
        ok, log = _run_python_script(
            perception_smoke_script,
            perception_args,
            cwd=ROOT,
        )
        if not ok:
            _fail("run_perception", log)
        else:
            stage_results["run_perception"] = "done"
            status_stage = "run_perception"
            # Mirror events_v0 artifact to dedicated event directory for easier browsing.
            if event_json_path.exists():
                event_dir.mkdir(parents=True, exist_ok=True)
                (event_dir / "events_v0.json").write_text(event_json_path.read_text(encoding="utf-8"), encoding="utf-8")
    elif run_perception:
        stage_results["run_perception"] = "skipped"

    if status == "ok" and run_eval and actions["gen_queries"]:
        ok, log = _run_python_script(
            gen_queries_script,
            [
                "--json",
                str(json_path),
                "--out",
                str(queries_path),
                "--seed",
                str(seed),
                "--n_time",
                "10",
                "--n_hard_time",
                "10",
                "--n_token",
                "10",
                "--n_decision",
                "10",
            ],
            cwd=ROOT,
        )
        if not ok:
            _fail("gen_queries", log)
        else:
            stage_results["gen_queries"] = "done"
            status_stage = "gen_queries"
    elif run_eval:
        stage_results["gen_queries"] = "skipped"

    results_overall_path = eval_dir / "results_overall.csv"
    report_path = eval_dir / "report.md"
    if status == "ok" and run_eval and actions["eval_cross"]:
        eval_args = [
            "--json",
            str(json_path),
            "--queries",
            str(queries_path),
            "--out_dir",
            str(eval_dir),
        ]
        if sweep:
            eval_args.append("--sweep")
        ok, log = _run_python_script(eval_cross_script, eval_args, cwd=ROOT)
        if not ok:
            _fail("eval_cross", log)
        else:
            stage_results["eval_cross"] = "done"
            status_stage = "eval_cross"
    elif run_eval:
        stage_results["eval_cross"] = "skipped"

    nlq_results_path = nlq_dir / "nlq_results.csv"
    nlq_summary_path = nlq_dir / "nlq_summary.csv"
    nlq_report_path = nlq_dir / "nlq_report.md"
    nlq_safety_path = nlq_dir / "safety_report.json"
    if status == "ok" and run_nlq and actions["eval_nlq"]:
        nlq_args = [
            "--json",
            str(json_path),
            "--index",
            str(index_prefix),
            "--out_dir",
            str(nlq_dir),
            "--mode",
            str(nlq_mode),
            "--n",
            str(int(nlq_n)),
            "--seed",
            str(int(nlq_seed)),
            "--topk",
            str(int(nlq_topk)),
        ]
        if nlq_mode == "ego4d" and nlq_ann:
            nlq_args.extend(["--ann", str(nlq_ann)])
        if nlq_sweep:
            nlq_args.append("--sweep")
        ok, log = _run_python_script(eval_nlq_script, nlq_args, cwd=ROOT)
        if not ok:
            _fail("eval_nlq", log)
        else:
            stage_results["eval_nlq"] = "done"
            status_stage = "eval_nlq"
    elif run_nlq:
        stage_results["eval_nlq"] = "skipped"

    updated_entry = dict(entry)
    updated_entry.update(
        {
            "pipeline_json_path": str(json_path),
            "index_prefix": str(index_prefix),
            "cross_eval_dir": str(eval_dir),
            "nlq_eval_dir": str(nlq_dir),
            "perception_dir": str(perception_dir),
            "event_dir": str(event_dir),
            "status_stage": status_stage if status == "ok" else status,
        }
    )

    run_record = {
        "video_uid": uid,
        "status": status,
        "src_path": str(src_path),
        "input_path": str(input_path),
        "json_path": str(json_path),
        "index_prefix": str(index_prefix),
        "eval_dir": str(eval_dir),
        "queries_path": str(queries_path),
        "results_overall_path": str(results_overall_path),
        "report_path": str(report_path),
        "nlq_dir": str(nlq_dir),
        "nlq_results_path": str(nlq_results_path),
        "nlq_summary_path": str(nlq_summary_path),
        "nlq_report_path": str(nlq_report_path),
        "nlq_safety_path": str(nlq_safety_path),
        "perception_dir": str(perception_dir),
        "event_dir": str(event_dir),
        "perception_json_path": str(perception_json_path),
        "perception_report_path": str(perception_report_path),
        "event_json_path": str(event_json_path),
        "status_stage": status_stage if status == "ok" else status,
        "stage_results": stage_results,
        "error_tail": last_error[-800:] if last_error else "",
    }
    return updated_entry, run_record


def _write_nlq_summary_all(out_dir: Path, run_records: list[dict[str, Any]]) -> Path:
    out_path = out_dir / "nlq_summary_all.csv"
    rows: list[dict[str, Any]] = []
    for record in run_records:
        uid = str(record.get("video_uid", ""))
        summary_path = Path(str(record.get("nlq_summary_path", "")))
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            src_rows = list(csv.DictReader(f))
        for row in src_rows:
            item = dict(row)
            item["video_uid"] = uid
            rows.append(item)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _write_summary(
    out_dir: Path,
    root: Path,
    found_count: int,
    chosen_entries: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
    proxy_enabled: bool,
    run_eval: bool,
    sweep: bool,
    run_nlq: bool,
    nlq_mode: str,
    run_perception: bool,
    summary_budget: dict[str, Any],
    nlq_summary_all_path: Path,
) -> tuple[Path, Path]:
    summary_csv = out_dir / "summary.csv"
    summary_md = out_dir / "summary.md"

    csv_rows: list[dict[str, Any]] = []
    for record in run_records:
        row: dict[str, Any] = {
            "video_uid": record.get("video_uid"),
            "status": record.get("status"),
            "status_stage": record.get("status_stage"),
            "src_path": record.get("src_path"),
            "input_path": record.get("input_path"),
            "json_path": record.get("json_path"),
            "index_prefix": record.get("index_prefix"),
            "eval_dir": record.get("eval_dir"),
            "nlq_dir": record.get("nlq_dir"),
            "perception_dir": record.get("perception_dir"),
            "event_dir": record.get("event_dir"),
            "budget_max_total_s": summary_budget["max_total_s"],
            "budget_max_tokens": summary_budget["max_tokens"],
            "budget_max_decisions": summary_budget["max_decisions"],
            "critical_fn_rate": "",
            "critical_fn_count": "",
            "critical_fn_denominator": "",
            "safety_count_granularity": "",
        }
        results_overall_path = Path(str(record.get("results_overall_path", "")))
        if results_overall_path.exists():
            with results_overall_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            chosen_row = _select_default_budget_row(
                rows=rows,
                budget_total_s=float(summary_budget["max_total_s"]),
                budget_tokens=int(summary_budget["max_tokens"]),
                budget_decisions=int(summary_budget["max_decisions"]),
            )
            if chosen_row is not None:
                for key in (
                    "variant",
                    "hit_at_k",
                    "mrr",
                    "coverage_ratio",
                    "compression_ratio",
                    "kept_duration_s",
                    "duration_s",
                    "tokens_total",
                    "decisions_total",
                    "highlights_total",
                ):
                    row[key] = chosen_row.get(key, "")
        nlq_safety_path = Path(str(record.get("nlq_safety_path", "")))
        if nlq_safety_path.exists():
            try:
                safety_payload = json.loads(nlq_safety_path.read_text(encoding="utf-8"))
            except Exception:
                safety_payload = {}
            if isinstance(safety_payload, dict):
                variant_stats = safety_payload.get("variant_stats", {})
                full_stats = variant_stats.get("full", {}) if isinstance(variant_stats, dict) else {}
                row["critical_fn_rate"] = full_stats.get(
                    "critical_fn_rate",
                    safety_payload.get("critical_fn_rate", ""),
                )
                row["critical_fn_count"] = full_stats.get(
                    "critical_fn_count",
                    safety_payload.get("critical_fn_count", ""),
                )
                row["critical_fn_denominator"] = full_stats.get(
                    "critical_fn_denominator",
                    safety_payload.get("critical_fn_denominator", ""),
                )
                row["safety_count_granularity"] = safety_payload.get("count_granularity", "")
        csv_rows.append(row)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in csv_rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    lines: list[str] = []
    lines.append("# Ego4D Smoke Summary")
    lines.append("")
    lines.append(f"- root: `{root}`")
    lines.append(f"- found_mp4: {found_count}")
    lines.append(f"- chosen: {len(chosen_entries)}")
    lines.append(f"- proxy_enabled: {str(proxy_enabled).lower()}")
    lines.append(f"- run_eval: {str(run_eval).lower()}")
    lines.append(f"- sweep: {str(sweep).lower()}")
    lines.append(f"- run_nlq: {str(run_nlq).lower()}")
    lines.append(f"- nlq_mode: {nlq_mode}")
    lines.append(f"- run_perception: {str(run_perception).lower()}")
    lines.append(f"- summary_csv: `{summary_csv}`")
    lines.append(f"- nlq_summary_all_csv: `{nlq_summary_all_path}`")
    lines.append("")
    lines.append("## Per Video")
    lines.append("")
    for record in run_records:
        uid = str(record.get("video_uid", ""))
        stage_results = record.get("stage_results", {})
        if isinstance(stage_results, dict):
            stage_text = ",".join(f"{k}:{v}" for k, v in stage_results.items())
        else:
            stage_text = ""
        lines.append(f"### {uid}")
        lines.append(f"- status: {record.get('status', '')}")
        lines.append(f"- status_stage: {record.get('status_stage', '')}")
        lines.append(f"- stages: `{stage_text}`")
        lines.append(f"- src_path: `{record.get('src_path', '')}`")
        lines.append(f"- input_path: `{record.get('input_path', '')}`")
        lines.append(f"- json_path: `{record.get('json_path', '')}`")
        lines.append(f"- index_prefix: `{record.get('index_prefix', '')}`")
        lines.append(f"- eval_dir: `{record.get('eval_dir', '')}`")
        lines.append(f"- nlq_dir: `{record.get('nlq_dir', '')}`")
        lines.append(f"- perception_dir: `{record.get('perception_dir', '')}`")
        lines.append(f"- event_dir: `{record.get('event_dir', '')}`")
        report_path = Path(str(record.get("report_path", "")))
        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8")
            overall = _extract_markdown_section(report_text, "## Overall Summary")
            deltas = _extract_markdown_section(report_text, "## Key Deltas")
            if overall:
                lines.append("")
                lines.append(overall)
            if deltas:
                lines.append("")
                lines.append(deltas)
        nlq_report_path = Path(str(record.get("nlq_report_path", "")))
        if nlq_report_path.exists():
            nlq_text = nlq_report_path.read_text(encoding="utf-8")
            nlq_deltas = _extract_markdown_section(nlq_text, "## Key Deltas")
            if nlq_deltas:
                lines.append("")
                lines.append("## NLQ Key Deltas")
                lines.append("")
                lines.append(nlq_deltas)
        lines.append("")
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return summary_csv, summary_md


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        print(f"error=root_not_found path={root}")
        return 1

    entries = scan_and_plan(
        root=root,
        n=0,
        seed=int(args.seed),
        min_size_bytes=int(args.min_size_bytes),
    )
    for entry in entries:
        entry["chosen"] = False
    entry_map = {str(entry["video_uid"]): entry for entry in entries}
    found_mp4 = len(entries)

    candidates = filter_entries_by_patterns(
        entries,
        include_patterns=list(args.include),
        exclude_patterns=list(args.exclude),
    )
    candidates_count = len(candidates)
    catalog_map = _load_catalog(Path(args.catalog)) if args.catalog else None
    jobs = max(1, int(args.jobs))
    probed_count = enrich_candidates_with_metadata(
        candidates,
        catalog_map=catalog_map,
        probe_candidates=int(args.probe_candidates),
        seed=int(args.seed),
        jobs=jobs,
    )
    chosen_entries = choose_sample_entries(
        candidates,
        n=int(args.n),
        seed=int(args.seed),
        prefer_short=bool(args.prefer_short),
        prefer_long=bool(args.prefer_long),
        stratified=bool(args.stratified),
        duration_bins=parse_duration_bins(str(args.duration_bins)),
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
    )
    chosen_ids = {str(entry.get("video_uid")) for entry in chosen_entries}
    for entry in entries:
        entry["chosen"] = str(entry.get("video_uid")) in chosen_ids

    known_durations = [
        float(entry["duration_s"])
        for entry in chosen_entries
        if isinstance(entry.get("duration_s"), (int, float))
    ]
    avg_duration = (sum(known_durations) / len(known_durations)) if known_durations else None

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, entries)
    print(f"found_mp4={found_mp4}")
    print(f"candidates={candidates_count}")
    print(f"probed={probed_count}")
    print(f"chosen={len(chosen_entries)}")
    print(f"avg_duration={(f'{avg_duration:.2f}' if avg_duration is not None else 'na')}")

    ffmpeg_available = has_command("ffmpeg")
    proxy_requested = True if args.proxy is None else bool(args.proxy)
    proxy_enabled = bool(proxy_requested and ffmpeg_available)
    if proxy_requested and not ffmpeg_available:
        print("warn=ffmpeg_not_found proxy_skipped=true")
    print(f"proxy_enabled={str(proxy_enabled).lower()}")
    print(f"run_perception={str(bool(args.run_perception)).lower()}")
    if bool(args.run_perception):
        print(f"perception_fps={float(args.perception_fps):.2f}")
        print(f"perception_max_frames={int(args.perception_max_frames)}")
        print(f"perception_backend={str(args.perception_backend)}")
        print(f"perception_fallback_stub={str(bool(args.perception_fallback_stub)).lower()}")
        print(f"perception_strict={str(bool(args.perception_strict)).lower()}")

    # Proxy stage (parallel, capped for safety).
    if proxy_enabled and chosen_entries:
        proxy_dir = out_dir / "proxy"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        proxy_workers = min(2, jobs)
        if proxy_workers <= 1:
            new_entries: list[dict[str, Any]] = []
            for entry in chosen_entries:
                new_entries.append(
                    _proxy_one(
                        entry,
                        proxy_dir=proxy_dir,
                        proxy_height=int(args.proxy_height),
                        crf=int(args.crf),
                        preset=str(args.preset),
                        delete_original=bool(args.delete_original),
                    )
                )
        else:
            new_entries = []
            with ThreadPoolExecutor(max_workers=proxy_workers) as executor:
                futures = [
                    executor.submit(
                        _proxy_one,
                        entry,
                        proxy_dir=proxy_dir,
                        proxy_height=int(args.proxy_height),
                        crf=int(args.crf),
                        preset=str(args.preset),
                        delete_original=bool(args.delete_original),
                    )
                    for entry in chosen_entries
                ]
                for fut in as_completed(futures):
                    new_entries.append(fut.result())
        for updated in new_entries:
            uid = str(updated["video_uid"])
            entry_map[uid].update(updated)
        chosen_entries = [entry_map[str(entry["video_uid"])] for entry in chosen_entries]
        write_jsonl(manifest_path, entries)

    json_dir = out_dir / "json"
    cache_dir = out_dir / "cache"
    eval_root_dir = out_dir / "eval"
    nlq_root_dir = out_dir / "nlq"
    perception_root_dir = out_dir / "perception"
    event_root_dir = out_dir / "event"
    json_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_root_dir.mkdir(parents=True, exist_ok=True)
    nlq_root_dir.mkdir(parents=True, exist_ok=True)
    perception_root_dir.mkdir(parents=True, exist_ok=True)
    event_root_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    kwargs = {
        "json_dir": json_dir,
        "cache_dir": cache_dir,
        "eval_root_dir": eval_root_dir,
        "nlq_root_dir": nlq_root_dir,
        "perception_root_dir": perception_root_dir,
        "event_root_dir": event_root_dir,
        "seed": int(args.seed),
        "run_perception": bool(args.run_perception),
        "perception_fps": float(args.perception_fps),
        "perception_max_frames": int(args.perception_max_frames),
        "perception_backend": str(args.perception_backend),
        "perception_fallback_stub": bool(args.perception_fallback_stub),
        "perception_strict": bool(args.perception_strict),
        "perception_cache_root": out_dir / "perception_cache",
        "run_eval": bool(args.run_eval),
        "sweep": bool(args.sweep),
        "run_nlq": bool(args.run_nlq),
        "nlq_mode": str(args.nlq_mode),
        "nlq_ann": str(args.nlq_ann) if args.nlq_ann else None,
        "nlq_sweep": bool(args.nlq_sweep),
        "nlq_n": int(args.nlq_n),
        "nlq_seed": int(args.nlq_seed),
        "nlq_topk": int(args.nlq_topk),
        "resume": bool(args.resume),
    }

    if jobs <= 1 or len(chosen_entries) <= 1:
        for entry in chosen_entries:
            updated_entry, run_record = _process_video(entry, **kwargs)
            uid = str(updated_entry["video_uid"])
            entry_map[uid].update(updated_entry)
            run_records.append(run_record)
            stage_results = run_record.get("stage_results", {})
            if isinstance(stage_results, dict):
                stage_text = ",".join(f"{k}:{v}" for k, v in stage_results.items())
            else:
                stage_text = ""
            print(
                f"per_video uid={uid} status={run_record['status']} stage={run_record['status_stage']} "
                f"json={run_record['json_path']} index={run_record['index_prefix']} eval={run_record['eval_dir']} "
                f"nlq={run_record['nlq_dir']} perception={run_record.get('perception_dir','')} "
                f"event={run_record.get('event_dir','')} stages={stage_text}"
            )
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_process_video, entry, **kwargs) for entry in chosen_entries]
            for fut in as_completed(futures):
                updated_entry, run_record = fut.result()
                uid = str(updated_entry["video_uid"])
                entry_map[uid].update(updated_entry)
                run_records.append(run_record)
                stage_results = run_record.get("stage_results", {})
                if isinstance(stage_results, dict):
                    stage_text = ",".join(f"{k}:{v}" for k, v in stage_results.items())
                else:
                    stage_text = ""
                print(
                    f"per_video uid={uid} status={run_record['status']} stage={run_record['status_stage']} "
                    f"json={run_record['json_path']} index={run_record['index_prefix']} eval={run_record['eval_dir']} "
                    f"nlq={run_record['nlq_dir']} perception={run_record.get('perception_dir','')} "
                    f"event={run_record.get('event_dir','')} stages={stage_text}"
                )

    # Manifest final update with stage outputs.
    write_jsonl(manifest_path, entries)

    nlq_summary_all_path = _write_nlq_summary_all(out_dir=out_dir, run_records=run_records)
    summary_budget = {
        "max_total_s": float(args.summary_max_total_s),
        "max_tokens": int(args.summary_max_tokens),
        "max_decisions": int(args.summary_max_decisions),
    }
    summary_csv, summary_md = _write_summary(
        out_dir=out_dir,
        root=root,
        found_count=len(entries),
        chosen_entries=chosen_entries,
        run_records=run_records,
        proxy_enabled=proxy_enabled,
        run_eval=bool(args.run_eval),
        sweep=bool(args.sweep),
        run_nlq=bool(args.run_nlq),
        nlq_mode=str(args.nlq_mode),
        run_perception=bool(args.run_perception),
        summary_budget=summary_budget,
        nlq_summary_all_path=nlq_summary_all_path,
    )
    print(f"manifest_saved={manifest_path}")
    print(f"summary saved={summary_md}")
    print(f"summary_csv_saved={summary_csv}")
    print(f"nlq_summary_all_saved={nlq_summary_all_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
