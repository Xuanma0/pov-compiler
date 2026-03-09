from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import load_hard_pseudo_nlq
from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.perception.object_memory_v0 import build_object_memory_v0
from pov_compiler.schemas import Output


def _load_output(path: Path) -> Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _uid_from_json(path: Path) -> str:
    stem = path.stem
    cleaned = re.sub(r"(?i)_v\d+_decisions$", "", stem)
    cleaned = re.sub(r"(?i)_decisions$", "", cleaned)
    match = re.search(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", cleaned)
    if match:
        return str(match.group(1)).lower()
    return cleaned


def _discover_jsons(path: Path) -> dict[str, Path]:
    if path.is_file():
        uid = _uid_from_json(path)
        return {uid: path}
    files = sorted(path.glob("*_v03_decisions.json"), key=lambda p: p.name.lower())
    if not files:
        files = sorted(path.glob("*.json"), key=lambda p: p.name.lower())
    out: dict[str, Path] = {}
    for item in files:
        uid = _uid_from_json(item)
        if uid and uid not in out:
            out[uid] = item
    return out


def _read_uids(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).replace("\ufeff", "")
        if "#" in text:
            text = text.split("#", 1)[0]
        text = text.strip()
        if not text:
            continue
        for token in re.split(r"[,\s]+", text):
            t = token.strip().lower()
            if t.endswith(".mp4"):
                t = t[:-4]
            if t:
                out.append(t)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build object memory and lost-object NLQ smoke outputs")
    parser.add_argument("--pov_json", required=True, help="POV output json path or directory of json files")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--uids-file", default=None, help="Optional uid list file")
    parser.add_argument("--contact-threshold", type=float, default=0.6)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index-dir", default=None, help="Optional index root for eval_nlq")
    parser.add_argument("--eval-script", default=str(ROOT / "scripts" / "eval_nlq.py"))
    return parser.parse_args()


def _attach_perception_if_needed(output: Output, json_path: Path) -> Output:
    if isinstance(output.perception, dict) and output.perception:
        return output
    run_root = json_path.parent.parent
    sidecar = run_root / "perception" / str(output.video_id) / "perception.json"
    if not sidecar.exists():
        return output
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return output
    if isinstance(payload, dict):
        output.perception = payload
    return output


def _write_report(path: Path, rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    lines = [
        "# Object Memory Smoke",
        "",
        "## Selection",
        "",
        f"- selected_uids: {selection.get('selected_uids', [])}",
        f"- requested_uids: {selection.get('requested_uids', 0)}",
        f"- missing_uids: {selection.get('missing_uids', [])}",
        "",
        "| uid | objects_total | lost_object_queries_total | eval_returncode | object_memory_json | lost_object_queries_json | nlq_report_md |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('uid', '')} | {int(row.get('objects_total', 0))} | "
            f"{int(row.get('lost_object_queries_total', 0))} | {int(row.get('eval_returncode', -1))} | "
            f"{row.get('object_memory_json', '')} | {row.get('lost_object_queries_json', '')} | {row.get('nlq_report_md', '')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = Path(args.pov_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    discovered = _discover_jsons(source)
    if not discovered:
        print(f"error=no_json_found source={source}")
        return 2

    rng = random.Random(int(args.seed))
    all_uids = sorted(discovered.keys())
    requested = all_uids
    missing: list[str] = []
    if args.uids_file:
        req = _read_uids(Path(args.uids_file))
        requested = req
        selected_raw = [uid for uid in req if uid in discovered]
        missing = [uid for uid in req if uid not in discovered]
    else:
        selected_raw = list(all_uids)
    if int(args.n) > 0:
        if len(selected_raw) > int(args.n):
            rng.shuffle(selected_raw)
            selected_raw = sorted(selected_raw[: int(args.n)])
    if not selected_raw:
        print("error=no_selected_uids")
        return 2

    rows: list[dict[str, Any]] = []
    saved_object_paths: list[str] = []
    saved_query_paths: list[str] = []
    saved_report_paths: list[str] = []
    for uid in selected_raw:
        json_path = discovered[uid]
        uid_dir = out_dir / uid
        uid_dir.mkdir(parents=True, exist_ok=True)
        output = ensure_events_v1(_attach_perception_if_needed(_load_output(json_path), json_path))

        object_memory = list(output.object_memory_v0 or [])
        if not object_memory:
            object_memory = build_object_memory_v0(
                perception=output.perception,
                events_v1=list(output.events_v1),
                contact_threshold=float(args.contact_threshold),
            )
        output.object_memory_v0 = object_memory

        object_json = uid_dir / "object_memory.json"
        object_json.write_text(
            json.dumps([x.model_dump() if hasattr(x, "model_dump") else x.dict() for x in object_memory], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        samples = load_hard_pseudo_nlq(
            output,
            seed=int(args.seed),
            n_highlight=max(4, int(args.n) * 4),
            n_token=max(4, int(args.n) * 2),
            n_decision=max(4, int(args.n) * 2),
            top_k=6,
        )
        lost_samples = [s for s in samples if str(s.query_type) == "hard_pseudo_lost_object"]
        lost_json = uid_dir / "lost_object_queries.json"
        lost_json.write_text(json.dumps([s.to_dict() for s in lost_samples], ensure_ascii=False, indent=2), encoding="utf-8")

        eval_out_dir = uid_dir / "nlq_eval"
        cmd = [
            sys.executable,
            str(Path(args.eval_script)),
            "--json",
            str(json_path),
            "--out_dir",
            str(eval_out_dir),
            "--mode",
            "hard_pseudo_nlq",
            "--seed",
            str(int(args.seed)),
            "--n",
            "8",
            "--top-k",
            "6",
            "--no-allow-gt-fallback",
            "--no-safety-gate",
        ]
        if args.index_dir:
            cmd.extend(["--index", str(Path(args.index_dir) / uid)])
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        (uid_dir / "eval.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (uid_dir / "eval.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        nlq_report = eval_out_dir / "nlq_report.md"
        if not nlq_report.exists():
            eval_out_dir.mkdir(parents=True, exist_ok=True)
            nlq_report.write_text(
                "# NLQ Report (fallback)\n\n"
                f"- returncode: {int(proc.returncode)}\n"
                f"- note: eval script did not produce nlq_report.md\n",
                encoding="utf-8",
            )

        row = {
            "uid": uid,
            "objects_total": len(object_memory),
            "lost_object_queries_total": len(lost_samples),
            "eval_returncode": int(proc.returncode),
            "object_memory_json": str(object_json),
            "lost_object_queries_json": str(lost_json),
            "nlq_report_md": str(nlq_report),
            "json_path": str(json_path),
        }
        rows.append(row)
        saved_object_paths.append(str(object_json))
        saved_query_paths.append(str(lost_json))
        saved_report_paths.append(str(nlq_report))

    report_path = out_dir / "nlq_report.md"
    selection = {
        "selected_uids": selected_raw,
        "requested_uids": len(requested),
        "missing_uids": missing[:20],
    }
    _write_report(report_path, rows, selection)
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json": str(source),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "contact_threshold": float(args.contact_threshold),
            "n": int(args.n),
            "seed": int(args.seed),
            "index_dir": str(args.index_dir) if args.index_dir else None,
        },
        "selection": selection,
        "rows": rows,
        "outputs": {
            "object_memory_json": saved_object_paths,
            "lost_object_queries_json": saved_query_paths,
            "nlq_report_md": str(report_path),
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected_uids={selected_raw}")
    print(f"objects_total={sum(int(r['objects_total']) for r in rows)}")
    print(f"lost_object_queries_total={sum(int(r['lost_object_queries_total']) for r in rows)}")
    if saved_object_paths:
        print(f"saved_object_memory={saved_object_paths[0]}")
    if saved_query_paths:
        print(f"saved_queries={saved_query_paths[0]}")
    print(f"saved_nlq_report={report_path}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
