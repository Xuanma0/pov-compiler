from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pov_compiler.models import ModelClientConfig, generate_structured, make_client
from pov_compiler.models.client import redact_url


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load provider health configs.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "ok"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


@contextmanager
def _temporary_env(env_name: str, value: str | None):
    original = os.environ.get(env_name)
    set_override = value is not None
    if set_override:
        os.environ[env_name] = value
    try:
        yield
    finally:
        if set_override:
            if original is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = original


def _sanitize_error(exc: Exception) -> str:
    text = redact_url(str(exc))
    lowered = text.lower()
    if "authorization" in lowered or "bearer" in lowered or "api key" in lowered:
        return "secret_redacted"
    return text


def _error_status(exc: Exception) -> str:
    text = str(exc).lower()
    if "missing api key env" in text:
        return "missing_key"
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "auth_fail"
    if any(
        token in text
        for token in (
            "timed out",
            "connection refused",
            "name or service not known",
            "winerror 10061",
            "failed to establish a new connection",
        )
    ):
        return "server_unavailable"
    if "schema" in text:
        return "schema_fail"
    return "unknown"


class _FixtureHandler(BaseHTTPRequestHandler):
    server_version = "LocalOpenAIProof/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _write_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/v1/models", "/models"}:
            self._write_json(
                {
                    "object": "list",
                    "data": [{"id": "local-openai-proof", "object": "model"}],
                },
                status=200,
            )
            return
        self._write_json({"error": {"message": "not_found"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        fixture = getattr(self.server, "fixture_config", {})
        if bool(fixture.get("require_auth", True)) and not str(self.headers.get("Authorization", "")).strip():
            self._write_json({"error": {"message": "unauthorized"}}, status=401)
            return
        time.sleep(float(fixture.get("latency_sleep_s", 0.02) or 0.02))
        payload = self._read_json()
        usage = {
            "prompt_tokens": 13,
            "completion_tokens": 11,
            "total_tokens": 24,
            "input_tokens": 13,
            "output_tokens": 11,
        }
        structured_content = {
            "status": "ok",
            "provider": "local_openai_fixture",
            "mode": "structured",
            "proof": "live",
        }
        plain_text = "live proof ok"
        if self.path.endswith("/responses"):
            text_payload = payload.get("text", {})
            if isinstance(text_payload, dict) and isinstance(text_payload.get("format"), dict):
                output_text = json.dumps(structured_content, ensure_ascii=False)
            else:
                output_text = plain_text
            self._write_json(
                {
                    "id": "resp_local_proof",
                    "object": "response",
                    "output_text": output_text,
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": output_text}],
                        }
                    ],
                    "usage": usage,
                },
                status=200,
            )
            return
        if self.path.endswith("/chat/completions"):
            response_format = payload.get("response_format", {})
            if isinstance(response_format, dict) and str(response_format.get("type", "")).strip():
                content = json.dumps(structured_content, ensure_ascii=False)
            else:
                content = plain_text
            self._write_json(
                {
                    "id": "chat_local_proof",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                    "usage": {
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                    },
                },
                status=200,
            )
            return
        self._write_json({"error": {"message": "not_found"}}, status=404)


@dataclass
class _FixtureHandle:
    server: ThreadingHTTPServer
    thread: threading.Thread
    base_url: str

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)


def _launch_fixture_server(target: dict[str, Any]) -> _FixtureHandle:
    host = str(target.get("fixture_host", "127.0.0.1")).strip() or "127.0.0.1"
    server = ThreadingHTTPServer((host, 0), _FixtureHandler)
    server.fixture_config = {
        "require_auth": True,
        "latency_sleep_s": float(_to_float(target.get("fixture_latency_sleep_s")) or 0.02),
    }
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return _FixtureHandle(server=server, thread=thread, base_url=f"http://{host}:{port}/v1")


def _probe_target(target: dict[str, Any]) -> dict[str, Any]:
    fixture_handle: _FixtureHandle | None = None
    attempted_modes: list[str] = []
    proof_status = "fail"
    real_call_status = "unknown"
    reachable = False
    usage_present = False
    structured_output_supported: bool | str = "unknown"
    latency_ms: float | None = None
    error = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model_cost_usd_total: float | None = None

    provider = str(target.get("provider", target.get("target_id", ""))).strip()
    model = str(target.get("model", "")).strip()
    client_provider = str(target.get("client_provider", "openai_compat")).strip() or "openai_compat"
    api_key_env = str(target.get("api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
    base_url = str(target.get("base_url", "")).strip()
    use_dummy_key = bool(_to_bool(target.get("use_dummy_key")))
    dummy_api_key = str(target.get("dummy_api_key", "local-openai-proof")).strip() or "local-openai-proof"
    timeout_s = int(_to_float(target.get("timeout_s")) or 8)
    max_tokens = int(_to_float(target.get("max_tokens")) or 80)
    temperature = float(_to_float(target.get("temperature")) or 0.0)
    structured_probe = bool(_to_bool(target.get("structured_probe")))
    expected_usage = bool(_to_bool(target.get("expected_usage")))
    api_mode_tested = str(target.get("api_mode", "auto")).strip() or "auto"

    if bool(_to_bool(target.get("launch_fixture_server"))):
        fixture_handle = _launch_fixture_server(target)
        base_url = fixture_handle.base_url

    env_value: str | None = None
    if use_dummy_key and not os.environ.get(api_key_env):
        env_value = dummy_api_key

    try:
        with _temporary_env(api_key_env, env_value):
            cfg = ModelClientConfig(
                provider=client_provider,
                model=model,
                api_mode=str(target.get("api_mode", "auto") or "auto"),
                base_url=base_url,
                api_key_env=api_key_env,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                temperature=temperature,
                model_cache_enabled=False,
                max_retries=0,
            )
            client = make_client(cfg)
            text, call_meta = client.generate_text(
                system="You are a minimal provider health check.",
                user="Reply exactly with: live proof ok",
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            reachable = bool(str(text).strip())
            api_mode_tested = str(call_meta.get("api_mode_used", api_mode_tested)).strip() or api_mode_tested
            attempted_modes.append(api_mode_tested)
            usage_present = all(call_meta.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            prompt_tokens = int(call_meta.get("prompt_tokens", 0) or 0) if usage_present else None
            completion_tokens = int(call_meta.get("completion_tokens", 0) or 0) if usage_present else None
            total_tokens = int(call_meta.get("total_tokens", 0) or 0) if usage_present else None
            latency_ms = _to_float(call_meta.get("latency_ms"))
            model_cost_usd_total = _to_float(call_meta.get("estimated_cost_usd"))
            real_call_status = "ok"
            proof_status = "partial" if expected_usage and not usage_present else "ok"

            if structured_probe:
                structured_obj, _raw_text, structured_meta = generate_structured(
                    client,
                    schema_name="provider_health_probe",
                    schema_json={
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "provider": {"type": "string"},
                            "mode": {"type": "string"},
                        },
                        "required": ["status", "provider", "mode"],
                        "additionalProperties": True,
                    },
                    system_prompt="Return a compact JSON health proof.",
                    user_prompt="Return a JSON object confirming structured-output support.",
                    strategy="json_schema",
                    strict=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )
                if structured_obj:
                    structured_output_supported = True
                    attempted_modes.append(str(structured_meta.get("api_mode_used", api_mode_tested)))
                    probe_latency = _to_float(structured_meta.get("latency_ms"))
                    if probe_latency is not None:
                        latency_ms = probe_latency if latency_ms is None else float((latency_ms + probe_latency) / 2.0)
                else:
                    structured_output_supported = False
                    real_call_status = "schema_fail"
                    proof_status = "partial"
            else:
                structured_output_supported = "unknown"
    except Exception as exc:
        error = _sanitize_error(exc)
        real_call_status = _error_status(exc)
        proof_status = "fail"
    finally:
        if fixture_handle is not None:
            fixture_handle.close()

    if isinstance(structured_output_supported, str) and structured_output_supported == "unknown" and structured_probe and real_call_status == "schema_fail":
        structured_output_supported = False

    return {
        "provider": provider,
        "base_url": redact_url(base_url),
        "model": model,
        "api_mode_tested": api_mode_tested,
        "reachable": bool(reachable),
        "real_call_status": real_call_status,
        "usage_present": bool(usage_present),
        "latency_ms": latency_ms,
        "structured_output_supported": structured_output_supported,
        "proof_status": proof_status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model_cost_usd_total": model_cost_usd_total,
        "calls_total": 1 if real_call_status in {"ok", "schema_fail"} else 0,
        "calls_with_usage": 1 if usage_present and real_call_status in {"ok", "schema_fail"} else 0,
        "calls_with_cost": 1 if model_cost_usd_total is not None and real_call_status in {"ok", "schema_fail"} else 0,
        "attempted_modes": [item for item in attempted_modes if str(item).strip()],
        "normalization_status": "ok" if proof_status == "ok" else ("partial" if proof_status == "partial" else "fail"),
        "error": error,
    }


def build_provider_reachability_summary(
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    payload = _load_yaml(config_file)
    targets = payload.get("targets", [])
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"No targets declared in provider health config: {config_file}")
    target = targets[0] if isinstance(targets[0], dict) else {}
    summary = _probe_target(target)
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["provider_health_config"] = str(config_file)
    summary["target_id"] = str(target.get("target_id", "")).strip()
    return summary


def write_provider_reachability_outputs(
    *,
    config_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary = build_provider_reachability_summary(config_path=config_path)
    summary_path = out_root / "summary.json"
    report_path = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    _write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))
    report_lines = [
        "# Provider Reachability",
        "",
        f"- provider_health_config: `{summary.get('provider_health_config')}`",
        f"- target_id: `{summary.get('target_id', '')}`",
        f"- provider: `{summary.get('provider', '')}`",
        f"- base_url: `{summary.get('base_url', '')}`",
        f"- model: `{summary.get('model', '')}`",
        f"- api_mode_tested: `{summary.get('api_mode_tested', '')}`",
        f"- reachable: `{summary.get('reachable', False)}`",
        f"- real_call_status: `{summary.get('real_call_status', 'unknown')}`",
        f"- proof_status: `{summary.get('proof_status', 'fail')}`",
        f"- usage_present: `{summary.get('usage_present', False)}`",
        f"- latency_ms: `{summary.get('latency_ms')}`",
        f"- structured_output_supported: `{summary.get('structured_output_supported', 'unknown')}`",
        f"- error: `{summary.get('error', '')}`",
        "",
        "## Files",
        "",
        f"- summary_json: `{summary_path}`",
        f"- report_md: `{report_path}`",
        f"- snapshot_json: `{snapshot_path}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    snapshot = dict(summary)
    snapshot["outputs"] = {
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "snapshot_json": str(snapshot_path),
    }
    _write_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2))
    return {
        "summary_json": summary_path,
        "report_md": report_path,
        "snapshot_json": snapshot_path,
        "summary": summary,
    }


def load_provider_reachability_outputs(reachability_dir: str | Path | None) -> dict[str, Any]:
    if reachability_dir is None:
        return {}
    root = Path(reachability_dir).resolve()
    if not root.exists():
        return {}
    return _read_json(root / "summary.json")
