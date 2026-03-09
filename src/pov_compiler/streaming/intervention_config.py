from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _to_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_scalar(text: str) -> Any:
    raw = str(text).strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except Exception:
        return raw.strip('"').strip("'")


def _load_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return dict(payload)
    except Exception:
        pass

    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).strip()
        if not text or text.startswith("#"):
            continue
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        out[str(key).strip()] = _parse_scalar(value)
    return out


@dataclass
class InterventionConfig:
    name: str = "default"
    w_safety: float = 1.0
    w_latency: float = 0.2
    w_trials: float = 0.1
    penalty_budget_up: float = 0.35
    penalty_retry: float = 0.1
    penalty_relax: float = 0.15
    max_trials_cap: int = 5

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "InterventionConfig":
        data = dict(payload or {})
        return cls(
            name=str(data.get("name", "default") or "default"),
            w_safety=_to_float(data.get("w_safety", 1.0), 1.0),
            w_latency=_to_float(data.get("w_latency", 0.2), 0.2),
            w_trials=_to_float(data.get("w_trials", 0.1), 0.1),
            penalty_budget_up=_to_float(data.get("penalty_budget_up", 0.35), 0.35),
            penalty_retry=_to_float(data.get("penalty_retry", 0.1), 0.1),
            penalty_relax=_to_float(data.get("penalty_relax", 0.15), 0.15),
            max_trials_cap=max(1, _to_int(data.get("max_trials_cap", 5), 5)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InterventionConfig":
        payload = _load_yaml_like(Path(path))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {
            "name": str(d["name"]),
            "w_safety": float(d["w_safety"]),
            "w_latency": float(d["w_latency"]),
            "w_trials": float(d["w_trials"]),
            "penalty_budget_up": float(d["penalty_budget_up"]),
            "penalty_retry": float(d["penalty_retry"]),
            "penalty_relax": float(d["penalty_relax"]),
            "max_trials_cap": int(d["max_trials_cap"]),
        }

    def to_yaml(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        try:
            import yaml  # type: ignore

            out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
            return out
        except Exception:
            lines = [f"{k}: {data[k]}" for k in (
                "name",
                "w_safety",
                "w_latency",
                "w_trials",
                "penalty_budget_up",
                "penalty_retry",
                "penalty_relax",
                "max_trials_cap",
            )]
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return out

    def stable_hash(self) -> str:
        payload = {
            "name": str(self.name),
            "w_safety": round(float(self.w_safety), 8),
            "w_latency": round(float(self.w_latency), 8),
            "w_trials": round(float(self.w_trials), 8),
            "penalty_budget_up": round(float(self.penalty_budget_up), 8),
            "penalty_retry": round(float(self.penalty_retry), 8),
            "penalty_relax": round(float(self.penalty_relax), 8),
            "max_trials_cap": int(self.max_trials_cap),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def resolve_intervention_config(cfg: InterventionConfig | dict[str, Any] | str | Path | None) -> InterventionConfig:
    if cfg is None:
        return InterventionConfig()
    if isinstance(cfg, InterventionConfig):
        return cfg
    if isinstance(cfg, dict):
        return InterventionConfig.from_dict(cfg)
    return InterventionConfig.from_yaml(Path(cfg))
