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

from pov_compiler.models import ModelClientConfig, get_preset, make_client
from pov_compiler.models.client import redact_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provider health check (supports dry-run; no key values are printed).")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "openai_compat", "gemini", "deepseek", "qwen", "glm", "fake"],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--timeout-s", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.set_defaults(dry_run=False, do_request=False)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--do-request", dest="do_request", action="store_true")
    return parser.parse_args()


def _print_preview(cfg: ModelClientConfig, *, dry_run: bool, api_key_present: bool) -> None:
    print(f"provider={cfg.provider}")
    print(f"model={cfg.model}")
    print(f"base_url={redact_url(str(cfg.base_url or ''))}")
    print(f"api_key_env={cfg.api_key_env}")
    print(f"api_key_present={str(bool(api_key_present)).lower()}")
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
        timeout_s=int(args.timeout_s),
        max_tokens=32,
        temperature=0.0,
        max_retries=int(args.max_retries),
        model_cache_enabled=False,
    )
    api_key_present = bool(os.environ.get(cfg.api_key_env, ""))
    dry_run = bool(args.dry_run) or not bool(args.do_request)
    _print_preview(cfg, dry_run=dry_run, api_key_present=api_key_present)

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

