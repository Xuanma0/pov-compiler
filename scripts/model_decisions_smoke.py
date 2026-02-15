from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.l3_decisions.model_compiler import compile_decisions_with_model
from pov_compiler.models import ModelClientConfig, get_model_cache_stats, make_client
from pov_compiler.schemas import Output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-based decision compiler smoke")
    parser.add_argument("--json", required=True, help="Input *_v03_decisions.json path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--provider",
        default="fake",
        choices=["fake", "openai_compat", "gemini", "qwen", "deepseek", "glm"],
    )
    parser.add_argument("--model", default=None, help="Model name (provider default if omitted)")
    parser.add_argument("--base_url", default=None, help="Optional base URL")
    parser.add_argument("--api_key_env", default=None, help="Optional API key environment variable name override")
    parser.add_argument("--timeout_s", type=int, default=60)
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--fake-mode", choices=["minimal", "diverse"], default="minimal")
    parser.add_argument("--model-cache-dir", default="data/outputs/model_cache")
    parser.set_defaults(model_cache=True)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--model-cache", dest="model_cache", action="store_true")
    cache_group.add_argument("--no-model-cache", dest="model_cache", action="store_false")
    parser.add_argument("--print_env_hint", action="store_true", help="Print expected env var name (never prints values)")
    return parser.parse_args()


def _provider_default_model(provider: str) -> str:
    defaults = {
        "fake": "fake-decision-v1",
        "openai_compat": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "qwen": "qwen-plus",
        "deepseek": "deepseek-chat",
        "glm": "glm-4-flash",
    }
    return defaults.get(str(provider).strip().lower(), "fake-decision-v1")


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
    try:
        return Output.parse_obj(payload)
    except Exception:
        if isinstance(payload, dict) and "events_v1" in payload:
            payload = dict(payload)
            payload.pop("events_v1", None)
            return Output.parse_obj(payload)
        raise


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    in_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = ensure_events_v1(_load_output(in_path))
    cfg = ModelClientConfig(
        provider=str(args.provider),
        model=str(args.model or _provider_default_model(str(args.provider))),
        base_url=str(args.base_url) if args.base_url else None,
        api_key_env=str(args.api_key_env or ""),
        timeout_s=int(args.timeout_s),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        model_cache_enabled=bool(args.model_cache),
        model_cache_dir=str(args.model_cache_dir),
        extra={"fake_mode": str(args.fake_mode)},
    )
    if args.print_env_hint and cfg.provider != "fake":
        print(f"env_hint={cfg.api_key_env}")

    client = make_client(cfg)
    decisions_model_v1 = compile_decisions_with_model(output=output, client=client, cfg=cfg)
    cache_stats = get_model_cache_stats(client)

    decisions_path = out_dir / "decisions_model_v1.json"
    _write_json(decisions_path, {"video_id": output.video_id, "decisions_model_v1": decisions_model_v1})

    report_path = out_dir / "report.md"
    report_lines = [
        "# Model Decisions Smoke",
        "",
        f"- video_id: `{output.video_id}`",
        f"- provider: `{cfg.provider}`",
        f"- model: `{cfg.model}`",
        f"- decisions_model_v1_total: `{len(decisions_model_v1)}`",
        f"- model_cache_enabled: `{str(bool(cache_stats.get('enabled', False))).lower()}`",
        f"- model_cache_dir: `{cache_stats.get('dir', '')}`",
        f"- model_cache_stats: `{json.dumps({'hit': int(cache_stats.get('hit', 0)), 'miss': int(cache_stats.get('miss', 0)), 'write_fail': int(cache_stats.get('write_fail', 0))}, ensure_ascii=False, sort_keys=True)}`",
        f"- model_cache_hash_prefix: `{cache_stats.get('hash_prefix', '')}`",
        f"- output: `{decisions_path}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json": str(in_path),
            "provider": cfg.provider,
            "model": cfg.model,
        },
        "client_cfg": cfg.to_public_dict(),
        "outputs": {
            "decisions_model_v1_json": str(decisions_path),
            "report_md": str(report_path),
            "decisions_model_v1_total": len(decisions_model_v1),
            "model_cache_stats": {
                "enabled": bool(cache_stats.get("enabled", False)),
                "dir": str(cache_stats.get("dir", "")),
                "hit": int(cache_stats.get("hit", 0)),
                "miss": int(cache_stats.get("miss", 0)),
                "write_fail": int(cache_stats.get("write_fail", 0)),
                "hash_prefix": str(cache_stats.get("hash_prefix", "")),
            },
        },
    }
    snapshot = ModelClientConfig.redact_dict(snapshot)
    snapshot_path = out_dir / "snapshot.json"
    _write_json(snapshot_path, snapshot)

    print(f"video_id={output.video_id}")
    print(f"provider={cfg.provider}")
    print(f"model={cfg.model}")
    print(f"decisions_model_v1_total={len(decisions_model_v1)}")
    print(f"model_cache_enabled={str(bool(cache_stats.get('enabled', False))).lower()}")
    print(f"model_cache_dir={cache_stats.get('dir', '')}")
    print(
        "model_cache_stats="
        + json.dumps(
            {
                "hit": int(cache_stats.get("hit", 0)),
                "miss": int(cache_stats.get("miss", 0)),
                "write_fail": int(cache_stats.get("write_fail", 0)),
                "hash_prefix": str(cache_stats.get("hash_prefix", "")),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print(f"saved_decisions={decisions_path}")
    print(f"saved_report={report_path}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
