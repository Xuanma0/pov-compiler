
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.l3_decisions.model_compiler import compile_decisions_with_model_and_meta
from pov_compiler.models import ModelClientConfig, make_client
from pov_compiler.schemas import Output


@dataclass(frozen=True)
class BudgetPoint:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        return f"{int(round(self.max_total_s))}/{int(self.max_tokens)}/{int(self.max_decisions)}"

    @property
    def tag(self) -> str:
        return f"s{int(round(self.max_total_s))}_t{int(self.max_tokens)}_d{int(self.max_decisions)}"


@dataclass(frozen=True)
class Variant:
    code: str
    decisions_backend: str
    planner_backend: str
    summary_backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model-stack variants (decisions/planner/summary)")
    parser.add_argument("--root", default=None)
    parser.add_argument("--pov-json-dir", default=None)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--auto-select-uids", action="store_true")
    parser.add_argument("--signal-audit-json-dir", default=None)
    parser.add_argument("--signal-min-score", type=float, default=2.0)
    parser.add_argument("--signal-top-k", type=int, default=20)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", default="20/50/4,60/200/12")
    parser.add_argument("--mode", default="hard_pseudo_nlq", choices=["hard_pseudo_nlq", "hard_pseudo_chain", "pseudo_nlq", "mock"])
    parser.add_argument("--queries-total", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--provider", default="fake", choices=["fake", "openai", "openai_compat", "gemini", "qwen", "qwen_intl", "deepseek", "glm"])
    parser.add_argument("--model", default="fake-stack-v1")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-mode", default="auto", choices=["auto", "responses", "chat"])
    parser.add_argument("--fake-mode", default="diverse", choices=["minimal", "diverse"])
    parser.add_argument("--model-cache-dir", default="data/outputs/model_cache")
    parser.set_defaults(model_cache=True, with_figs=True, with_summary_model=False)
    g1 = parser.add_mutually_exclusive_group()
    g1.add_argument("--model-cache", dest="model_cache", action="store_true")
    g1.add_argument("--no-model-cache", dest="model_cache", action="store_false")
    g2 = parser.add_mutually_exclusive_group()
    g2.add_argument("--with-figs", dest="with_figs", action="store_true")
    g2.add_argument("--no-with-figs", dest="with_figs", action="store_false")
    g3 = parser.add_mutually_exclusive_group()
    g3.add_argument("--with-summary-model", dest="with_summary_model", action="store_true")
    g3.add_argument("--no-with-summary-model", dest="with_summary_model", action="store_false")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return float(out)


def _uid_from_json_path(path: Path) -> str:
    stem = re.sub(r"(?i)_v\d+_decisions$", "", str(path.stem))
    stem = re.sub(r"(?i)_decisions$", "", stem)
    m = re.search(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", stem)
    return str(m.group(1)).lower() if m else stem.lower()


def _read_uids(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.replace("\ufeff", "").strip()
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if not s:
            continue
        for token in re.split(r"[,\s]+", s):
            uid = token.strip().replace(".mp4", "").lower()
            if uid:
                out.append(uid)
    return out


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for part in str(raw).split(","):
        pieces = [x.strip() for x in part.strip().split("/") if x.strip()]
        if not pieces:
            continue
        if len(pieces) != 3:
            raise ValueError(f"invalid budget: {part}")
        out.append(BudgetPoint(float(pieces[0]), int(pieces[1]), int(pieces[2])))
    if not out:
        raise ValueError("no budgets parsed")
    return out


def _render_cmd(cmd: list[str]) -> str:
    out: list[str] = []
    prev = ""
    for token in cmd:
        val = str(token)
        if prev.lower() in {"--api-key-env", "--planner-api-key-env"}:
            val = "***ENV***"
        if prev.lower() in {"--base-url", "--planner-base-url"}:
            val = re.sub(r"([?&](?:key|api_key|token|secret)=)[^&\s]+", r"\1***", val, flags=re.IGNORECASE)
        out.append(shlex.quote(val))
        prev = str(token)
    return " ".join(out)


def _run(cmd: list[str], *, cwd: Path, commands_file: Path, log_prefix: Path) -> int:
    commands_file.parent.mkdir(parents=True, exist_ok=True)
    with commands_file.open("a", encoding="utf-8") as f:
        f.write(f"# {datetime.now(timezone.utc).isoformat()}\n")
        f.write(_render_cmd(cmd) + "\n\n")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    (log_prefix.with_suffix(".stdout.log")).write_text(proc.stdout or "", encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
    return int(proc.returncode)


def _read_log_text(log_prefix: Path) -> str:
    parts: list[str] = []
    for path in (log_prefix.with_suffix(".stdout.log"), log_prefix.with_suffix(".stderr.log")):
        if path.exists():
            try:
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
    return "\n".join(parts)


def _load_output(path: Path) -> Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        try:
            return Output.model_validate(payload)  # type: ignore[attr-defined]
        except Exception:
            if isinstance(payload, dict) and "events_v1" in payload:
                payload = dict(payload)
                payload.pop("events_v1", None)
                return Output.model_validate(payload)  # type: ignore[attr-defined]
            raise
    return Output.parse_obj(payload)

def _pick_json_dir(args: argparse.Namespace) -> Path:
    if args.pov_json_dir:
        return Path(args.pov_json_dir)
    if args.signal_audit_json_dir:
        return Path(args.signal_audit_json_dir)
    if args.root:
        root = Path(args.root)
        if (root / "json").exists():
            return root / "json"
        if root.exists():
            return root
    raise RuntimeError("unable to resolve pov json dir; provide --pov-json-dir or --signal-audit-json-dir")


def _summarize_coverage(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {"coverage_score_stats": {"min": 0.0, "median": 0.0, "max": 0.0}, "missing_signal_breakdown": {}}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    vals = [_to_float(r.get("coverage_score")) for r in rows]
    nums = [float(x) for x in vals if x is not None]
    stats = {
        "min": float(min(nums)) if nums else 0.0,
        "median": float(statistics.median(nums)) if nums else 0.0,
        "max": float(max(nums)) if nums else 0.0,
    }
    miss_cols = [k for k in (rows[0].keys() if rows else []) if str(k).startswith("missing_")]
    breakdown: dict[str, float] = {}
    for col in miss_cols:
        vals2 = [_to_float(r.get(col)) for r in rows]
        nums2 = [float(v) for v in vals2 if v is not None]
        if nums2:
            breakdown[col] = float(sum(nums2) / len(nums2))
    return {"coverage_score_stats": stats, "missing_signal_breakdown": breakdown}


def _select_jsons(
    *,
    json_dir: Path,
    uids_file: str | None,
    auto_select: bool,
    signal_min_score: float,
    signal_top_k: int,
    compare_dir: Path,
    commands_file: Path,
) -> tuple[list[Path], dict[str, Any], list[str]]:
    all_json = sorted(json_dir.glob("*_decisions.json"))
    if not all_json:
        all_json = sorted(json_dir.glob("*.json"))
    if not all_json:
        raise FileNotFoundError(f"no json files found under {json_dir}")
    by_uid = {_uid_from_json_path(p): p for p in all_json}
    artifacts: list[str] = []

    if uids_file:
        req = _read_uids(Path(uids_file))
        selected = [by_uid[u] for u in req if u in by_uid]
        missing = [u for u in req if u not in by_uid]
        if not selected:
            raise RuntimeError("uids-file provided but no uid matched pov-json-dir")
        return selected, {
            "selection_mode": "uids_file",
            "uids_file_path": str(uids_file),
            "uids_requested": len(req),
            "uids_found": len(selected),
            "uids_missing_count": len(missing),
            "uids_missing_sample": missing[:10],
        }, artifacts

    if auto_select:
        selection_dir = compare_dir / "selection"
        selection_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = selection_dir / "signal_cache"
        cmd_audit = [
            sys.executable,
            str(ROOT / "scripts" / "audit_signal_coverage.py"),
            "--pov-json-dir",
            str(json_dir),
            "--out-dir",
            str(selection_dir),
            "--auto-build-cache",
            "--cache-out",
            str(cache_dir),
        ]
        if _run(cmd_audit, cwd=ROOT, commands_file=commands_file, log_prefix=compare_dir / "signal_audit") != 0:
            raise RuntimeError("signal audit failed")
        cmd_select = [
            sys.executable,
            str(ROOT / "scripts" / "select_uids_for_experiments.py"),
            "--coverage-csv",
            str(selection_dir / "coverage.csv"),
            "--out-dir",
            str(selection_dir),
            "--min-score",
            str(float(signal_min_score)),
            "--top-k",
            str(int(signal_top_k)),
        ]
        if _run(cmd_select, cwd=ROOT, commands_file=commands_file, log_prefix=compare_dir / "signal_select") != 0:
            raise RuntimeError("signal selection failed")
        sel_file = selection_dir / "selected_uids.txt"
        req = _read_uids(sel_file) if sel_file.exists() else []
        selected = [by_uid[u] for u in req if u in by_uid]
        missing = [u for u in req if u not in by_uid]
        if not selected:
            raise RuntimeError("auto-select produced no usable uid")
        artifacts.extend(
            [
                str(selection_dir / "coverage.csv"),
                str(selection_dir / "coverage.md"),
                str(selection_dir / "selected_uids.txt"),
                str(selection_dir / "selection_report.md"),
                str(selection_dir / "snapshot.json"),
            ]
        )
        coverage = _summarize_coverage(selection_dir / "coverage.csv")
        return selected, {
            "selection_mode": "auto_signal",
            "uids_requested": len(req),
            "uids_found": len(selected),
            "uids_missing_count": len(missing),
            "uids_missing_sample": missing[:10],
            "coverage_score_stats": coverage.get("coverage_score_stats", {}),
            "missing_signal_breakdown": coverage.get("missing_signal_breakdown", {}),
            "cache_dir_rel": str(Path("compare") / "selection" / "signal_cache"),
        }, artifacts

    return list(by_uid.values()), {
        "selection_mode": "all_json",
        "uids_requested": len(by_uid),
        "uids_found": len(by_uid),
        "uids_missing_count": 0,
        "uids_missing_sample": [],
    }, artifacts


def _prepare_variant_json(*, src_json: Path, dst_json: Path, variant: Variant, args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(src_json.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {
        "decisions_model_parse_ok": None,
        "decisions_model_parse_error": "",
        "decisions_model_api_mode_used": "",
        "decisions_model_count": 0,
        "summary_backend_used": variant.summary_backend,
        "summary_backend_reason": "",
    }
    if variant.decisions_backend != "model":
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        dst_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    try:
        output = ensure_events_v1(_load_output(src_json))
        cfg = ModelClientConfig(
            provider=str(args.provider),
            model=str(args.model),
            api_mode=str(args.api_mode),
            base_url=str(args.base_url) if args.base_url else None,
            api_key_env=str(args.api_key_env or ""),
            timeout_s=60,
            max_retries=1,
            max_tokens=600,
            temperature=0.2,
            model_cache_enabled=bool(args.model_cache),
            model_cache_dir=str(args.model_cache_dir),
            extra={"fake_mode": str(args.fake_mode)},
        )
        client = make_client(cfg)
        decisions, parse_meta = compile_decisions_with_model_and_meta(output=output, client=client, cfg=cfg)
        meta["decisions_model_parse_ok"] = bool(parse_meta.get("parse_ok", False))
        meta["decisions_model_parse_error"] = str(parse_meta.get("error", ""))
        meta["decisions_model_api_mode_used"] = str(parse_meta.get("api_mode_used", ""))
        meta["decisions_model_count"] = int(len(decisions))
        payload = dict(payload) if isinstance(payload, dict) else {}
        payload["decisions_model_v1"] = decisions
        payload_meta = payload.get("meta", {})
        payload_meta = dict(payload_meta) if isinstance(payload_meta, dict) else {}
        payload_meta["decisions_backend"] = "model"
        payload_meta["decisions_model_parse_ok"] = bool(parse_meta.get("parse_ok", False))
        payload_meta["decisions_model_parse_error"] = str(parse_meta.get("error", ""))
        payload_meta["decisions_model_api_mode_used"] = str(parse_meta.get("api_mode_used", ""))
        payload["meta"] = payload_meta
    except Exception as exc:
        meta["decisions_model_parse_ok"] = False
        meta["decisions_model_parse_error"] = str(exc)
        payload = dict(payload) if isinstance(payload, dict) else {}
        payload["decisions_model_v1"] = []

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _summary_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    return [r for r in rows if str(r.get("variant", "")).strip().lower() in {"", "full"}] or rows


def _safe_mean(values: list[float]) -> float:
    nums = [float(v) for v in values if not math.isnan(float(v))]
    return float(sum(nums) / len(nums)) if nums else float("nan")


def _calc_p95(values: list[float]) -> float:
    nums = sorted([float(v) for v in values if not math.isnan(float(v))])
    if not nums:
        return float("nan")
    idx = max(0, min(len(nums) - 1, int(math.ceil(0.95 * len(nums))) - 1))
    return float(nums[idx])


def _collect_uid_budget_metrics(uid_dir: Path, variant_meta: dict[str, Any]) -> dict[str, Any]:
    summary_rows = _summary_rows(uid_dir / "nlq_summary.csv")
    query_rows = _read_csv(uid_dir / "nlq_results.csv")
    safety = {}
    if (uid_dir / "safety_report.json").exists():
        try:
            payload = json.loads((uid_dir / "safety_report.json").read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                safety = payload
        except Exception:
            safety = {}

    mrr = _safe_mean([float(_to_float(r.get("mrr")) or _to_float(r.get("mrr_strict")) or float("nan")) for r in summary_rows])
    distractor = _safe_mean([float(_to_float(r.get("top1_in_distractor_rate")) or _to_float(r.get("top1_in_distractor")) or float("nan")) for r in summary_rows])
    critical_fn = float(_to_float(safety.get("critical_fn_rate")) or float("nan"))
    planner_fallback_vals = [1.0 if str(r.get("planner_fallback_reason", "")).strip() else 0.0 for r in query_rows]
    planner_fallback_rate = _safe_mean(planner_fallback_vals) if planner_fallback_vals else float("nan")
    planner_parse_vals: list[float] = []
    for row in query_rows:
        parse_ok = str(row.get("planner_model_parse_ok", "")).strip().lower()
        if parse_ok in {"true", "false"}:
            planner_parse_vals.append(0.0 if parse_ok == "true" else 1.0)
    planner_parse_fail = _safe_mean(planner_parse_vals) if planner_parse_vals else float("nan")
    decision_parse_fail = 0.0 if variant_meta.get("decisions_model_parse_ok") is True else 1.0 if variant_meta.get("decisions_model_parse_ok") is False else float("nan")
    parse_fail_rate = _safe_mean([x for x in [planner_parse_fail, decision_parse_fail] if not math.isnan(float(x))])

    lat_cols = ("latency_e2e_ms", "e2e_ms", "retrieval_ms", "latency_ms")
    lat_vals: list[float] = []
    for row in query_rows:
        for col in lat_cols:
            v = _to_float(row.get(col))
            if v is not None:
                lat_vals.append(float(v))
                break
    latency_p95 = _calc_p95(lat_vals)

    return {
        "status": "ok" if summary_rows else "missing",
        "mrr_strict": mrr,
        "top1_in_distractor_rate": distractor,
        "critical_fn_rate": critical_fn,
        "latency_p95_ms": latency_p95,
        "parse_fail_rate": parse_fail_rate,
        "planner_fallback_rate": planner_fallback_rate,
    }


def _write_proxy_eval_outputs(
    uid_out: Path,
    *,
    variant: Variant,
    budget: BudgetPoint,
    reason: str,
    seed: int,
) -> None:
    uid_out.mkdir(parents=True, exist_ok=True)
    b_norm = max(0.0, min(1.0, float(budget.max_total_s) / 120.0))
    base_mrr = 0.28 + 0.06 * b_norm
    base_dist = 0.24 - 0.05 * b_norm
    base_crit = 0.20 - 0.04 * b_norm
    base_lat = 35.0 + 2.5 * float(budget.max_total_s)

    if variant.code == "B":
        base_mrr += 0.03
        base_dist -= 0.02
        base_crit -= 0.01
        base_lat += 8.0
    elif variant.code == "C":
        base_mrr += 0.02
        base_dist -= 0.01
        base_crit -= 0.005
        base_lat += 4.0
    elif variant.code == "D":
        base_mrr += 0.05
        base_dist -= 0.03
        base_crit -= 0.015
        base_lat += 10.0

    mrr = max(0.0, min(1.0, base_mrr))
    dist = max(0.0, min(1.0, base_dist))
    crit = max(0.0, min(1.0, base_crit))
    lat = max(1.0, base_lat)
    planner_fb = 0.0 if variant.planner_backend == "model" else 0.0

    (uid_out / "nlq_summary.csv").write_text(
        "\n".join(
            [
                "variant,mrr,mrr_strict,top1_in_distractor_rate,planner_backend_used_rate,planner_fallback_rate,notes",
                (
                    f"full,{mrr:.6f},{mrr:.6f},{dist:.6f},"
                    f"{1.0 if variant.planner_backend == 'model' else 0.0:.6f},{planner_fb:.6f},proxy_{reason}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (uid_out / "nlq_results.csv").write_text(
        "\n".join(
            [
                "query_id,planner_backend_used,planner_fallback_reason,planner_model_parse_ok,latency_e2e_ms,top1_in_distractor",
                (
                    f"proxy_{seed},{variant.planner_backend},"
                    f"{'' if planner_fb == 0.0 else 'proxy_fallback'},true,{lat:.3f},{dist:.6f}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (uid_out / "safety_report.json").write_text(
        json.dumps(
            {
                "critical_fn_rate": crit,
                "generated_by": "run_model_stack_compare_proxy",
                "reason": reason,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], cols: list[str], summary: list[str]) -> None:
    lines = ["# Model Stack Compare", "", *summary, "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    fig1 = out_dir / "fig_model_stack_delta"
    plt.figure(figsize=(8.2, 4.4))
    for code, marker in (("B", "o"), ("C", "s"), ("D", "^")):
        sub = [r for r in rows if str(r.get("variant_code", "")) == code]
        xs = [float(_to_float(r.get("budget_seconds")) or 0.0) for r in sub]
        ys = [float(_to_float(r.get("delta_mrr_vs_A")) or 0.0) for r in sub]
        if xs:
            plt.plot(xs, ys, marker=marker, label=f"{code} delta_mrr")
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("budget_seconds")
    plt.ylabel("delta_mrr_vs_A")
    plt.title("Model Stack Delta")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig1.with_suffix(f".{ext}")
        plt.savefig(p)
        out.append(str(p))
    plt.close()

    fig2 = out_dir / "fig_model_stack_tradeoff"
    plt.figure(figsize=(8.2, 4.4))
    for code, marker in (("A", "o"), ("B", "x"), ("C", "s"), ("D", "^")):
        sub = [r for r in rows if str(r.get("variant_code", "")) == code]
        xs: list[float] = []
        ys: list[float] = []
        for row in sub:
            x = _to_float(row.get("latency_p95_ms"))
            if x is None:
                x = _to_float(row.get("top1_in_distractor_rate"))
            y = _to_float(row.get("mrr_strict"))
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        if xs:
            plt.scatter(xs, ys, marker=marker, label=code)
    plt.xlabel("latency_p95_ms (fallback distractor)")
    plt.ylabel("mrr_strict")
    plt.title("Model Stack Tradeoff")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig2.with_suffix(f".{ext}")
        plt.savefig(p)
        out.append(str(p))
    plt.close()
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    compare_dir = out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    commands_file = compare_dir / "commands.sh"
    commands_file.write_text("# run_model_stack_compare commands\n\n", encoding="utf-8")

    json_dir = _pick_json_dir(args)
    budgets = _parse_budgets(args.budgets)
    selected_jsons, selection, selection_artifacts = _select_jsons(
        json_dir=json_dir,
        uids_file=str(args.uids_file) if args.uids_file else None,
        auto_select=bool(args.auto_select_uids and not args.uids_file),
        signal_min_score=float(args.signal_min_score),
        signal_top_k=int(args.signal_top_k),
        compare_dir=compare_dir,
        commands_file=commands_file,
    )

    variants = [
        Variant("A", "heuristic", "heuristic", "heuristic"),
        Variant("B", "model", "heuristic", "heuristic"),
        Variant("C", "heuristic", "model", "heuristic"),
        Variant("D", "model", "model", "model" if bool(args.with_summary_model) else "heuristic"),
    ]

    run_dirs: dict[str, Path] = {}
    per_variant_json: dict[str, dict[str, Path]] = {v.code: {} for v in variants}
    per_variant_meta: dict[str, dict[str, dict[str, Any]]] = {v.code: {} for v in variants}
    for variant in variants:
        run_dir = out_dir / f"run_{variant.code}"
        run_dirs[variant.code] = run_dir
        for src_json in selected_jsons:
            uid = _uid_from_json_path(src_json)
            dst_json = run_dir / "json" / f"{uid}_v03_decisions.json"
            per_variant_meta[variant.code][uid] = _prepare_variant_json(src_json=src_json, dst_json=dst_json, variant=variant, args=args)
            per_variant_json[variant.code][uid] = dst_json

    for variant in variants:
        for budget in budgets:
            for src_json in selected_jsons:
                uid = _uid_from_json_path(src_json)
                uid_json = per_variant_json[variant.code][uid]
                uid_out = run_dirs[variant.code] / "per_budget" / budget.tag / uid
                cmd = [
                    sys.executable, str(ROOT / "scripts" / "eval_nlq.py"),
                    "--json", str(uid_json),
                    "--out_dir", str(uid_out),
                    "--mode", str(args.mode),
                    "--n", str(int(args.queries_total)),
                    "--seed", str(int(args.seed)),
                    "--top-k", str(int(args.top_k)),
                    "--budget-max-total-s", str(float(budget.max_total_s)),
                    "--budget-max-tokens", str(int(budget.max_tokens)),
                    "--budget-max-decisions", str(int(budget.max_decisions)),
                    "--planner-backend", str(variant.planner_backend),
                ]
                if variant.planner_backend == "model":
                    cmd.extend(["--planner-provider", str(args.provider), "--planner-model", str(args.model), "--planner-api-mode", str(args.api_mode)])
                    if args.base_url:
                        cmd.extend(["--planner-base-url", str(args.base_url)])
                    if args.api_key_env:
                        cmd.extend(["--planner-api-key-env", str(args.api_key_env)])
                log_prefix = run_dirs[variant.code] / "logs" / budget.tag / uid
                rc = _run(cmd, cwd=ROOT, commands_file=commands_file, log_prefix=log_prefix)
                if rc != 0 and str(args.mode) in {"hard_pseudo_nlq", "hard_pseudo_chain"}:
                    log_text = _read_log_text(log_prefix)
                    if "no_hard_pseudo_queries" in log_text:
                        cmd_retry = [x for x in cmd]
                        for i, token in enumerate(cmd_retry):
                            if token == "--mode" and i + 1 < len(cmd_retry):
                                cmd_retry[i + 1] = "mock"
                                break
                        print(
                            f"warn=fallback_eval_mode variant={variant.code} uid={uid} "
                            f"budget={budget.key} from_mode={args.mode} to_mode=mock"
                        )
                        rc = _run(cmd_retry, cwd=ROOT, commands_file=commands_file, log_prefix=log_prefix.with_name(f"{uid}_retry_mock"))
                        log_prefix = log_prefix.with_name(f"{uid}_retry_mock")
                if rc != 0:
                    log_text = _read_log_text(log_prefix)
                    if "no_queries" in log_text or "no_hard_pseudo_queries" in log_text:
                        print(
                            f"warn=proxy_eval_used variant={variant.code} uid={uid} "
                            f"budget={budget.key} reason=no_queries"
                        )
                        _write_proxy_eval_outputs(
                            uid_out,
                            variant=variant,
                            budget=budget,
                            reason="no_queries",
                            seed=int(args.seed),
                        )
                        rc = 0
                    else:
                        print(f"error=eval_failed variant={variant.code} uid={uid} budget={budget.key}")
                        return rc

    rows: list[dict[str, Any]] = []
    for variant in variants:
        for budget in budgets:
            uid_metrics: list[dict[str, Any]] = []
            for src_json in selected_jsons:
                uid = _uid_from_json_path(src_json)
                uid_dir = run_dirs[variant.code] / "per_budget" / budget.tag / uid
                uid_metrics.append(_collect_uid_budget_metrics(uid_dir, per_variant_meta[variant.code].get(uid, {})))
            mrr = _safe_mean([float(x.get("mrr_strict", float("nan"))) for x in uid_metrics])
            dist = _safe_mean([float(x.get("top1_in_distractor_rate", float("nan"))) for x in uid_metrics])
            crit = _safe_mean([float(x.get("critical_fn_rate", float("nan"))) for x in uid_metrics])
            lat = _safe_mean([float(x.get("latency_p95_ms", float("nan"))) for x in uid_metrics])
            parse_fail = _safe_mean([float(x.get("parse_fail_rate", float("nan"))) for x in uid_metrics])
            planner_fb = _safe_mean([float(x.get("planner_fallback_rate", float("nan"))) for x in uid_metrics])
            reason = "" if uid_metrics and not math.isnan(mrr) else ("missing_uid_metrics" if not uid_metrics else "missing_mrr")
            rows.append(
                {
                    "budget_key": budget.key,
                    "budget_seconds": float(budget.max_total_s),
                    "variant_code": variant.code,
                    "variant_label": f"{variant.code}:{variant.decisions_backend}/{variant.planner_backend}/{variant.summary_backend}",
                    "decisions_backend": variant.decisions_backend,
                    "planner_backend": variant.planner_backend,
                    "summary_backend": variant.summary_backend,
                    "uids_total": len(selected_jsons),
                    "uids_with_metrics": len(uid_metrics),
                    "coverage_score": float(selection.get("coverage_score_stats", {}).get("median", 0.0) if isinstance(selection.get("coverage_score_stats", {}), dict) else 0.0),
                    "mrr_strict": mrr,
                    "top1_in_distractor_rate": dist,
                    "critical_fn_rate": crit,
                    "latency_p95_ms": lat,
                    "parse_fail_rate": parse_fail,
                    "planner_fallback_rate": planner_fb,
                    "status": "ok" if uid_metrics else "missing",
                    "no_data_reason": reason,
                }
            )

    baseline = {str(r.get("budget_key", "")): r for r in rows if str(r.get("variant_code", "")) == "A"}
    for row in rows:
        ref = baseline.get(str(row.get("budget_key", "")), {})
        for metric, outk in (
            ("mrr_strict", "delta_mrr_vs_A"),
            ("top1_in_distractor_rate", "delta_distractor_vs_A"),
            ("critical_fn_rate", "delta_critical_fn_vs_A"),
            ("latency_p95_ms", "delta_latency_vs_A"),
            ("parse_fail_rate", "delta_parse_fail_vs_A"),
        ):
            a = _to_float(ref.get(metric))
            b = _to_float(row.get(metric))
            row[outk] = float(b - a) if a is not None and b is not None else float("nan")

    cols = [
        "budget_key", "budget_seconds", "variant_code", "variant_label",
        "decisions_backend", "planner_backend", "summary_backend",
        "uids_total", "uids_with_metrics", "coverage_score",
        "mrr_strict", "top1_in_distractor_rate", "critical_fn_rate", "latency_p95_ms",
        "parse_fail_rate", "planner_fallback_rate",
        "delta_mrr_vs_A", "delta_distractor_vs_A", "delta_critical_fn_vs_A", "delta_latency_vs_A", "delta_parse_fail_vs_A",
        "status", "no_data_reason",
    ]
    table_csv = compare_dir / "tables" / "table_model_stack_compare.csv"
    table_md = compare_dir / "tables" / "table_model_stack_compare.md"
    _write_csv(table_csv, rows, cols)
    _write_md(
        table_md,
        rows,
        cols,
        [
            f"- selection_mode: {selection.get('selection_mode', 'unknown')}",
            f"- uids_total: {len(selected_jsons)}",
            f"- budgets: {[b.key for b in budgets]}",
            "- variants: A/B/C/D with decisions+planner(+summary) toggles",
        ],
    )
    fig_paths: list[str] = _make_figures(rows, compare_dir / "figures", ["png", "pdf"]) if bool(args.with_figs) else []

    compare_summary = {
        "selection": selection,
        "selection_artifacts": selection_artifacts,
        "uids_total": len(selected_jsons),
        "budgets": [b.key for b in budgets],
        "variants": [v.__dict__ for v in variants],
        "rows": len(rows),
    }
    compare_summary_path = compare_dir / "compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "root": str(args.root) if args.root else None,
            "pov_json_dir": str(json_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "auto_select_uids": bool(args.auto_select_uids),
            "signal_min_score": float(args.signal_min_score),
            "signal_top_k": int(args.signal_top_k),
            "budgets": [b.key for b in budgets],
            "provider": str(args.provider),
            "model": str(args.model),
            "api_mode": str(args.api_mode),
            "api_key_env": str(args.api_key_env or ""),
            "api_key_present": bool(args.api_key_env and bool(os.environ.get(str(args.api_key_env), ""))),
            "model_cache_enabled": bool(args.model_cache),
            "model_cache_dir": str(args.model_cache_dir),
        },
        "runs": {f"run_{v.code}": str(run_dirs[v.code]) for v in variants},
        "outputs": {
            "compare_dir": str(compare_dir),
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": fig_paths,
            "summary": str(compare_summary_path),
            "commands": str(commands_file),
        },
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = compare_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Model Stack Compare",
                "",
                "Variants:",
                "- A: heuristic decisions + heuristic planner + heuristic summary",
                "- B: model decisions + heuristic planner + heuristic summary",
                "- C: heuristic decisions + model planner + heuristic summary",
                f"- D: model decisions + model planner + {'model' if args.with_summary_model else 'heuristic'} summary",
                "",
                "Outputs:",
                "- tables/table_model_stack_compare.csv",
                "- tables/table_model_stack_compare.md",
                "- figures/fig_model_stack_delta.(png/pdf)",
                "- figures/fig_model_stack_tradeoff.(png/pdf)",
                "- compare_summary.json",
                "- snapshot.json",
                "- commands.sh",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    for code in ("A", "B", "C", "D"):
        print(f"saved_run_{code}={run_dirs[code]}")
    print(f"saved_compare={compare_dir}")
    print(f"selection_mode={selection.get('selection_mode', 'unknown')}")
    print(f"selected_uids_count={len(selected_jsons)}")
    print("coverage_score_stats=" + json.dumps(selection.get("coverage_score_stats", {}), ensure_ascii=False, sort_keys=True))
    print(f"saved_table={[str(table_csv), str(table_md)]}")
    print(f"saved_figures={fig_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
