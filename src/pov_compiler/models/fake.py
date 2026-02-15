from __future__ import annotations

import hashlib
from typing import Any

from pov_compiler.models.client import ModelClientConfig


class FakeModelClient:
    def __init__(self, cfg: ModelClientConfig):
        self.cfg = cfg

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
            return {
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
        return {"decisions": decisions}
