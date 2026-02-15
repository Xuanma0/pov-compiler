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

from pov_compiler.models import ModelClientConfig, make_client
from pov_compiler.models.client import DEFAULT_API_KEY_ENV, redact_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional live model connectivity check (no network in tests).")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai_compat", "gemini", "qwen", "deepseek", "glm", "fake"],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--timeout-s", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provider = str(args.provider)
    api_key_env = str(args.api_key_env or DEFAULT_API_KEY_ENV.get(provider, "OPENAI_API_KEY"))
    cfg = ModelClientConfig(
        provider=provider,
        model=str(args.model),
        base_url=str(args.base_url) if args.base_url else None,
        api_key_env=api_key_env,
        timeout_s=int(args.timeout_s),
        max_tokens=32,
        temperature=0.0,
        model_cache_enabled=False,
    )
    if provider != "fake" and not bool(os.environ.get(api_key_env, "")):
        print(f"missing_env={api_key_env}")
        print(f"provider={provider}")
        print(f"model={cfg.model}")
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
        print(f"provider={provider}")
        print(f"model={cfg.model}")
        print(f"base_url={redact_url(str(cfg.base_url or ''))}")
        print(f"status=ok")
        print("status_code=200")
        print(f"latency_ms={latency_ms}")
        print(f"first_40_chars={preview}")
        return 0
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        print(f"provider={provider}")
        print(f"model={cfg.model}")
        print(f"base_url={redact_url(str(cfg.base_url or ''))}")
        print("status=error")
        print("status_code=0")
        print(f"latency_ms={latency_ms}")
        print(f"error={str(exc)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
