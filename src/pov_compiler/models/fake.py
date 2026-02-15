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
        h = hashlib.sha256(f"{self.cfg.model}|{system}|{user}".encode("utf-8")).hexdigest()
        kind_pool = [
            "ATTENTION_TURN_HEAD",
            "ATTENTION_STOP_LOOK",
            "TRANSITION",
            "REORIENT_AND_SCAN",
        ]
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
