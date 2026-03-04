from __future__ import annotations

import hashlib
import json
from typing import Any

from pov_compiler.models.client import ModelClientConfig


class FakeModelClient:
    def __init__(self, cfg: ModelClientConfig):
        self.cfg = cfg
        self._last_call_meta: dict[str, Any] = {}

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        strategy = str(kwargs.get("structured_strategy", ""))
        response_format = kwargs.get("response_format")
        if str(strategy) == "json_schema" and isinstance(response_format, dict):
            js = response_format.get("json_schema", {})
            if isinstance(js, dict):
                schema = js.get("schema", {})
                if isinstance(schema, dict) and "ok" in dict(schema.get("properties", {})):
                    text = '{"ok": true}'
                    meta = {
                        "mode": "fake_json_schema",
                        "api_mode_used": "chat",
                        "latency_ms": 1,
                        "prompt_tokens": 8,
                        "completion_tokens": 4,
                        "total_tokens": 12,
                        "estimated_cost_usd": 0.0,
                        "strategy_used": "json_schema",
                    }
                    self._last_call_meta = dict(meta)
                    return text, meta
        payload = self.complete_json(
            system=system,
            user=user,
            timeout_s=int(timeout_s),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        text = json.dumps(payload, ensure_ascii=False)
        meta = {
            "mode": "fake_json",
            "api_mode_used": "chat",
            "latency_ms": 1,
            "prompt_tokens": max(1, len(str(system).split()) + len(str(user).split())),
            "completion_tokens": max(1, len(text.split())),
            "total_tokens": max(1, len(str(system).split()) + len(str(user).split()) + len(text.split())),
            "estimated_cost_usd": 0.0,
            "strategy_used": strategy or "prompted_json",
        }
        self._last_call_meta = dict(meta)
        return text, meta

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        mode = str(self.cfg.extra.get("fake_mode", "minimal")).strip().lower() if isinstance(self.cfg.extra, dict) else "minimal"
        h = hashlib.sha256(f"{self.cfg.model}|{system}|{user}".encode("utf-8")).hexdigest()
        kind_pool = [
            "ATTENTION_TURN_HEAD",
            "ATTENTION_STOP_LOOK",
            "TRANSITION",
            "REORIENT_AND_SCAN",
        ]
        if mode != "diverse":
            idx = int(h[:2], 16) % len(kind_pool)
            t0_ms = int(int(h[2:6], 16) % 4000)
            span_ms = 1000 + int(int(h[6:8], 16) % 1000)
            t1_ms = t0_ms + span_ms
            conf = round(0.4 + (int(h[8:10], 16) % 60) / 100.0, 3)
            payload = {
                "decisions": [
                    {
                        "decision_type": kind_pool[idx],
                        "t0_ms": t0_ms,
                        "t1_ms": t1_ms,
                        "conf": conf,
                        "evidence": {"event_id": "event_0001", "span": "fake deterministic model output"},
                    }
                ]
            }
            self._last_call_meta = {
                "mode": "fake_json",
                "api_mode_used": "chat",
                "latency_ms": 1,
                "prompt_tokens": max(1, len(str(system).split()) + len(str(user).split())),
                "completion_tokens": max(1, len(json.dumps(payload, ensure_ascii=False).split())),
                "total_tokens": max(
                    1,
                    len(str(system).split()) + len(str(user).split()) + len(json.dumps(payload, ensure_ascii=False).split()),
                ),
                "estimated_cost_usd": 0.0,
            }
            return payload
        decisions: list[dict[str, object]] = []
        count = 3 + (int(h[0:2], 16) % 4)
        for i in range(count):
            idx = (int(h[2 + i : 4 + i], 16) + i) % len(kind_pool)
            t0_ms = 600 * i + int(int(h[8 + i : 10 + i], 16) % 500)
            span_ms = 800 + int(int(h[12 + i : 14 + i], 16) % 900)
            t1_ms = t0_ms + span_ms
            conf = round(0.45 + (int(h[16 + i : 18 + i], 16) % 50) / 100.0, 3)
            decisions.append(
                {
                    "id": f"model_decision_{i+1:04d}",
                    "decision_type": kind_pool[idx],
                    "t0_ms": int(t0_ms),
                    "t1_ms": int(t1_ms),
                    "conf": float(max(0.0, min(1.0, conf))),
                    "evidence": {
                        "event_id": f"event_{(i % 3) + 1:04d}",
                        "span": f"fake diverse decision {i+1}",
                    },
                }
            )
        payload = {"decisions": decisions}
        self._last_call_meta = {
            "mode": "fake_json",
            "api_mode_used": "chat",
            "latency_ms": 1,
            "prompt_tokens": max(1, len(str(system).split()) + len(str(user).split())),
            "completion_tokens": max(1, len(json.dumps(payload, ensure_ascii=False).split())),
            "total_tokens": max(
                1,
                len(str(system).split()) + len(str(user).split()) + len(json.dumps(payload, ensure_ascii=False).split()),
            ),
            "estimated_cost_usd": 0.0,
        }
        return payload

    @property
    def last_call_meta(self) -> dict[str, Any]:
        return dict(self._last_call_meta or {})
