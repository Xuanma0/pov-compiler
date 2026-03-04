from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models import (
    ModelClientConfig,
    capability_states,
    get_last_model_call_meta,
    get_preset,
    make_client,
    resolve_model_capabilities,
)
from pov_compiler.models.client import redact_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provider health check (supports dry-run; no key values are printed).")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "openai_compat", "gemini", "deepseek", "qwen", "qwen_intl", "glm", "fake"],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-mode", choices=["auto", "responses", "chat"], default="auto")
    parser.add_argument("--timeout-s", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--probe-capabilities", action="store_true", help="Optionally probe structured-output capability (requires key)")
    parser.add_argument("--probe-timeout-s", type=int, default=8)
    parser.add_argument("--probe-cache-dir", default="data/outputs/model_capabilities")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", dest="dry_run", action="store_true", help="Only print resolved route, do not request")
    mode_group.add_argument("--real", dest="real", action="store_true", help="Execute one minimal request (requires key)")
    parser.set_defaults(dry_run=True, real=False)
    return parser.parse_args()


def _resolved_mode_order(provider: str, requested_mode: str) -> list[str]:
    p = str(provider).strip().lower()
    mode = str(requested_mode).strip().lower()
    if mode == "responses":
        return ["responses", "chat"]
    if mode == "chat":
        return ["chat"]
    # auto
    preset = get_preset(p)
    if bool(getattr(preset, "supports_responses", False)):
        return ["responses", "chat"]
    return ["chat"]


def _print_preview(cfg: ModelClientConfig, *, dry_run: bool, api_key_present: bool) -> None:
    print(f"provider={cfg.provider}")
    print(f"model={cfg.model}")
    print(f"base_url={redact_url(str(cfg.base_url or ''))}")
    print(f"api_key_env={cfg.api_key_env}")
    print(f"api_key_present={str(bool(api_key_present)).lower()}")
    print(f"api_mode={cfg.api_mode}")
    print(f"api_mode_order={_resolved_mode_order(cfg.provider, cfg.api_mode)}")
    print(f"timeout_s={int(cfg.timeout_s)}")
    print(f"max_retries={int(cfg.max_retries)}")
    print(f"dry_run={str(bool(dry_run)).lower()}")


def main() -> int:
    args = parse_args()
    preset = get_preset(str(args.provider))
    api_key_env = str(args.api_key_env or preset.default_api_key_env)
    cfg = ModelClientConfig(
        provider=str(args.provider),
        model=str(args.model),
        base_url=str(args.base_url) if args.base_url else None,
        api_key_env=api_key_env,
        api_mode=str(args.api_mode),
        timeout_s=int(args.timeout_s),
        max_tokens=32,
        temperature=0.0,
        max_retries=int(args.max_retries),
        model_cache_enabled=False,
    )
    api_key_present = bool(os.environ.get(cfg.api_key_env, ""))
    dry_run = bool(args.dry_run) and not bool(args.real)
    _print_preview(cfg, dry_run=dry_run, api_key_present=api_key_present)
    caps, caps_meta = resolve_model_capabilities(
        cfg,
        probe=bool(args.probe_capabilities),
        probe_timeout_s=int(args.probe_timeout_s),
        probe_cache_dir=str(args.probe_cache_dir),
    )
    states = capability_states(caps)
    if dry_run and not bool(args.probe_capabilities) and not api_key_present and cfg.provider != "fake":
        states = {
            "json_schema": "unknown",
            "json_object": "unknown",
            "tools": "unknown",
            "responses_api": states.get("responses_api", "unknown"),
            "chat_api": states.get("chat_api", "unknown"),
        }
    print(f"capabilities_json_schema={states.get('json_schema', 'unknown')}")
    print(f"capabilities_json_object={states.get('json_object', 'unknown')}")
    print(f"capabilities_tools={states.get('tools', 'unknown')}")
    print(f"probe_used={str(bool(caps_meta.get('probe_used', False))).lower()}")
    print(f"probe_cached={str(bool(caps_meta.get('probe_cached', False))).lower()}")

    if dry_run:
        print("status=dry_run_ok")
        return 0
    if cfg.provider != "fake" and not api_key_present:
        print(f"missing_env={cfg.api_key_env}")
        print("status=missing_env")
        return 2

    client = make_client(cfg)
    system = "Return strict JSON only."
    user = '{"ping":"ok","echo":"health_check"}'
    t0 = time.perf_counter()
    try:
        response = client.complete_json(
            system=system,
            user=user,
            timeout_s=int(args.timeout_s),
            max_tokens=32,
            temperature=0.0,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        preview = json.dumps(response, ensure_ascii=False, sort_keys=True)[:40]
        print("status=ok")
        print("status_code=200")
        print(f"latency_ms={latency_ms}")
        print(f"first_40_chars={preview}")
        last_meta = get_last_model_call_meta(client)
        if last_meta:
            print(
                "telemetry="
                + json.dumps(
                    {
                        "api_mode_used": str(last_meta.get("api_mode_used", "")),
                        "prompt_tokens": int(last_meta.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(last_meta.get("completion_tokens", 0) or 0),
                        "estimated_cost_usd": last_meta.get("estimated_cost_usd"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return 0
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        print("status=error")
        print("status_code=0")
        print(f"latency_ms={latency_ms}")
        print(f"error={str(exc)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
