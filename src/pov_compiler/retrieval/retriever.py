from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any

import numpy as np

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.memory.vector_index import VectorIndex
from pov_compiler.retrieval.query_parser import ParsedQuery, QueryChain, parse_query, parse_query_chain
from pov_compiler.retrieval.reranker import Hit
from pov_compiler.models.client import redact_url
from pov_compiler.schemas import DecisionPoint, Output


def _as_output(output_json: str | Path | dict[str, Any] | Output) -> Output:
    if isinstance(output_json, Output):
        return output_json
    if isinstance(output_json, (str, Path)):
        data = json.loads(Path(output_json).read_text(encoding="utf-8"))
    elif isinstance(output_json, dict):
        data = output_json
    else:
        raise TypeError("output_json must be Output, dict, or path")
    if hasattr(Output, "model_validate"):
        return Output.model_validate(data)  # type: ignore[attr-defined]
    return Output.parse_obj(data)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-12:
        return arr
    return arr / denom


@dataclass
class RetrievalConfig:
    default_top_k: int = 8
    prefer: str = "highlight"


class _OpenCLIPTextEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
        import open_clip  # type: ignore
        import torch  # type: ignore

        self._torch = torch
        self._device = device
        self._model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

    def encode(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text]).to(self._device)
        with self._torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].detach().cpu().numpy().astype(np.float32)


class Retriever:
    def __init__(
        self,
        output_json: str | Path | dict[str, Any] | Output,
        index: VectorIndex | str | Path | None = None,
        config: dict[str, Any] | None = None,
    ):
        cfg = config or {}
        self.cfg = RetrievalConfig(
            default_top_k=int(cfg.get("default_top_k", 8)),
            prefer=str(cfg.get("prefer", "highlight")),
        )
        self.output = ensure_events_v1(_as_output(output_json))
        self.decision_pool_kind, self.decision_points = self._resolve_decision_pool()
        self.decision_pool_count = len(self.decision_points)
        self.decision_model_meta = self._decision_model_meta()
        if isinstance(index, VectorIndex):
            self.index = index
        elif index is None:
            self.index = None
        else:
            self.index = VectorIndex.load(index)
        self._text_encoder: _OpenCLIPTextEncoder | None = None

    def _decision_model_meta(self) -> dict[str, Any]:
        meta = dict(getattr(self.output, "meta", {}) or {})
        provider = str(meta.get("decisions_model_provider", "")).strip()
        model = str(meta.get("decisions_model_name", "")).strip()
        base_url = redact_url(str(meta.get("decisions_model_base_url", "")).strip())
        if not provider and not model and not base_url:
            return {}
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
        }

    def _nearest_event_id(self, t0_s: float, t1_s: float) -> str:
        best_id = ""
        best_dist = float("inf")
        center = 0.5 * (float(t0_s) + float(t1_s))
        for ev in list(getattr(self.output, "events_v1", []) or []):
            c = 0.5 * (float(getattr(ev, "t0", 0.0)) + float(getattr(ev, "t1", 0.0)))
            dist = abs(c - center)
            if dist < best_dist:
                best_dist = dist
                best_id = str(getattr(ev, "id", ""))
        if best_id:
            return best_id
        for ev in list(getattr(self.output, "events", []) or []):
            c = 0.5 * (float(getattr(ev, "t0", 0.0)) + float(getattr(ev, "t1", 0.0)))
            dist = abs(c - center)
            if dist < best_dist:
                best_dist = dist
                best_id = str(getattr(ev, "id", ""))
        return best_id

    def _model_decision_to_decision_point(self, item: dict[str, Any], idx: int) -> DecisionPoint | None:
        if not isinstance(item, dict):
            return None
        decision_type = str(item.get("decision_type", item.get("action_type", ""))).strip()
        if not decision_type:
            return None
        try:
            t0_ms = int(round(float(item.get("t0_ms", item.get("t0", 0.0) * 1000.0))))
            t1_ms = int(round(float(item.get("t1_ms", item.get("t1", 0.0) * 1000.0))))
        except Exception:
            return None
        t0_s = max(0.0, float(t0_ms) / 1000.0)
        t1_s = max(t0_s, float(t1_ms) / 1000.0)
        if t1_s <= t0_s:
            t1_s = t0_s + 1e-3
        try:
            conf = float(item.get("conf", item.get("confidence", 0.5)))
        except Exception:
            conf = 0.5
        conf = float(max(0.0, min(1.0, conf)))
        evidence = item.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        source_event = str(evidence.get("event_id", "")).strip()
        if not source_event:
            source_event = self._nearest_event_id(t0_s, t1_s)
        dp_id = str(item.get("id", "")).strip() or f"model_decision_{idx:04d}"
        return DecisionPoint(
            id=dp_id,
            t=0.5 * (t0_s + t1_s),
            t0=t0_s,
            t1=t1_s,
            source_event=source_event,
            source_highlight=None,
            trigger={"type": "MODEL", "anchor_types": [], "conf": conf},
            state={"nearby_tokens": [], "scene_change_nearby": False},
            action={"type": decision_type},
            constraints=[],
            outcome={"type": "MODEL_PRED"},
            alternatives=[],
            conf=conf,
            meta={
                "decision_source_kind": "decisions_model_v1",
                "evidence_span": str(evidence.get("span", "")),
                "model_provider": str(self._decision_model_meta().get("provider", "")),
                "model_name": str(self._decision_model_meta().get("model", "")),
                "model_base_url": str(self._decision_model_meta().get("base_url", "")),
            },
        )

    def _resolve_decision_pool(self) -> tuple[str, list[DecisionPoint]]:
        model_rows = list(getattr(self.output, "decisions_model_v1", []) or [])
        out_model: list[DecisionPoint] = []
        for i, item in enumerate(model_rows, start=1):
            dp = self._model_decision_to_decision_point(item, i)
            if dp is not None:
                out_model.append(dp)
        if out_model:
            return "decisions_model_v1", out_model
        return "decision_points", list(getattr(self.output, "decision_points", []) or [])

    def _event_pool(self) -> list[Any]:
        if self.output.events_v1:
            return list(self.output.events_v1)
        return list(self.output.events) + list(self.output.events_v0)

    @staticmethod
    def _event_label(event: Any) -> str:
        if hasattr(event, "label"):
            return str(getattr(event, "label", ""))
        return str(getattr(event, "meta", {}).get("label", ""))

    @staticmethod
    def _event_boundary_conf(event: Any) -> float:
        try:
            return float(getattr(event, "scores", {}).get("boundary_conf", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _event_contact_peak(event: Any) -> float:
        try:
            return float(getattr(event, "scores", {}).get("contact_peak", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _event_place_segment_id(event: Any) -> str:
        value = getattr(event, "place_segment_id", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return str(getattr(event, "meta", {}).get("place_segment_id", "")).strip()

    @staticmethod
    def _event_place_segment_conf(event: Any) -> float:
        value = getattr(event, "place_segment_conf", None)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(getattr(event, "meta", {}).get("place_segment_conf", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _event_place_segment_reason(event: Any) -> str:
        value = getattr(event, "place_segment_reason", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return str(getattr(event, "meta", {}).get("place_segment_reason", "")).strip()

    @staticmethod
    def _event_interaction_primary_object(event: Any) -> str:
        value = getattr(event, "interaction_primary_object", None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        sig = getattr(event, "interaction_signature", {}) if hasattr(event, "interaction_signature") else {}
        if isinstance(sig, dict):
            v2 = str(sig.get("active_object_top1", sig.get("active_object", ""))).strip().lower()
            if v2:
                return v2
        return str(getattr(event, "meta", {}).get("interaction_primary_object", "")).strip().lower()

    @staticmethod
    def _event_interaction_score(event: Any) -> float:
        value = getattr(event, "interaction_score", None)
        if isinstance(value, (int, float)):
            return float(value)
        sig = getattr(event, "interaction_signature", {}) if hasattr(event, "interaction_signature") else {}
        if isinstance(sig, dict):
            try:
                return float(sig.get("interaction_score", 0.0))
            except Exception:
                return 0.0
        try:
            return float(getattr(event, "meta", {}).get("interaction_score", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _event_anchor_types(event: Any) -> set[str]:
        if hasattr(event, "anchors"):
            return {str(anchor.type).lower() for anchor in getattr(event, "anchors", [])}
        out: set[str] = set()
        for evidence in getattr(event, "evidence", []):
            if str(getattr(evidence, "type", "")) == "anchor":
                anchor_type = str(getattr(evidence, "source", {}).get("anchor_type", "")).lower()
                if anchor_type:
                    out.add(anchor_type)
        return out

    def _get_text_encoder(self) -> _OpenCLIPTextEncoder | None:
        if self._text_encoder is not None:
            return self._text_encoder
        try:
            self._text_encoder = _OpenCLIPTextEncoder()
            return self._text_encoder
        except Exception:
            return None

    def _index_supports_text(self) -> bool:
        if self.index is None or self.index.size == 0:
            return False
        backends = {str(meta.get("embedding_backend", "")) for meta in self.index.metas}
        return backends == {"clip"} or "clip" in backends

    def _tokens_in_range(self, t0: float, t1: float) -> set[str]:
        return {token.id for token in self.output.token_codec.tokens if _overlap(token.t0, token.t1, t0, t1)}

    def _tokens_for_decision(self, decision: DecisionPoint) -> set[str]:
        ids = set()
        trigger_ids = decision.trigger.get("token_ids", [])
        if isinstance(trigger_ids, list):
            ids.update(str(x) for x in trigger_ids)
        evidence_ids = decision.state.get("evidence", {}).get("token_ids", [])
        if isinstance(evidence_ids, list):
            ids.update(str(x) for x in evidence_ids)
        outcome_ids = decision.outcome.get("evidence", {}).get("token_ids", [])
        if isinstance(outcome_ids, list):
            ids.update(str(x) for x in outcome_ids)
        ids.update(self._tokens_in_range(float(decision.t0), float(decision.t1)))
        return ids

    def _apply_constraint(
        self,
        current_events: set[str] | None,
        current_highlights: set[str] | None,
        current_tokens: set[str] | None,
        current_decisions: set[str] | None,
        events_new: set[str],
        highlights_new: set[str],
        tokens_new: set[str],
        decisions_new: set[str],
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        next_events = set(events_new) if current_events is None else current_events.intersection(events_new)
        next_highlights = (
            set(highlights_new) if current_highlights is None else current_highlights.intersection(highlights_new)
        )
        next_tokens = set(tokens_new) if current_tokens is None else current_tokens.intersection(tokens_new)
        next_decisions = (
            set(decisions_new) if current_decisions is None else current_decisions.intersection(decisions_new)
        )
        return next_events, next_highlights, next_tokens, next_decisions

    @staticmethod
    def _count_selected(result: dict[str, Any]) -> int:
        total = 0
        for key in ("selected_events", "selected_highlights", "selected_tokens", "selected_decisions"):
            values = result.get(key, [])
            if isinstance(values, list):
                total += len(values)
        return int(total)

    @staticmethod
    def _parsed_constraints(parsed: ParsedQuery) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if parsed.time_range is not None:
            out["time"] = {"t0": float(parsed.time_range[0]), "t1": float(parsed.time_range[1])}
        if parsed.token_types:
            out["token_types"] = list(parsed.token_types)
        if parsed.anchor_types:
            out["anchor_types"] = list(parsed.anchor_types)
        if parsed.decision_types:
            out["decision_types"] = list(parsed.decision_types)
        if parsed.event_ids:
            out["event_ids"] = list(parsed.event_ids)
        if parsed.event_labels:
            out["event_labels"] = list(parsed.event_labels)
        if parsed.contact_min is not None:
            out["contact_min"] = float(parsed.contact_min)
        if parsed.place:
            out["place"] = str(parsed.place)
        if parsed.place_segment_ids:
            out["place_segment_ids"] = list(parsed.place_segment_ids)
        if parsed.interaction_min is not None:
            out["interaction_min"] = float(parsed.interaction_min)
        if parsed.interaction_object:
            out["interaction_object"] = str(parsed.interaction_object)
        if parsed.object_name:
            out["object_name"] = str(parsed.object_name)
        if parsed.which:
            out["which"] = str(parsed.which)
        if parsed.prefer_contact:
            out["prefer_contact"] = bool(parsed.prefer_contact)
        if parsed.need_object_match:
            out["need_object_match"] = bool(parsed.need_object_match)
        if parsed.chain_time_min_s is not None:
            out["chain_time_min_s"] = float(parsed.chain_time_min_s)
        if parsed.chain_time_max_s is not None:
            out["chain_time_max_s"] = float(parsed.chain_time_max_s)
        if parsed.chain_time_mode:
            out["chain_time_mode"] = str(parsed.chain_time_mode)
        if parsed.chain_place_value:
            out["chain_place_value"] = str(parsed.chain_place_value)
        if parsed.chain_place_mode:
            out["chain_place_mode"] = str(parsed.chain_place_mode)
        if parsed.chain_object_value:
            out["chain_object_value"] = str(parsed.chain_object_value)
        if parsed.chain_object_mode:
            out["chain_object_mode"] = str(parsed.chain_object_mode)
        return out

    @staticmethod
    def _strip_query_keys(query: str, keys: set[str]) -> str:
        try:
            parts = shlex.split(str(query))
        except Exception:
            parts = str(query).split()
        kept: list[str] = []
        for part in parts:
            if "=" not in part:
                kept.append(part)
                continue
            key = str(part.split("=", 1)[0]).strip().lower().replace("-", "_")
            if key in keys:
                continue
            kept.append(part)
        return " ".join(kept).strip()

    @staticmethod
    def _derive_constraints_from_hit(
        hit: Hit,
        *,
        rel: str,
        window_s: float,
        derive: str = "time_only",
        place_mode: str = "soft",
        object_mode: str = "soft",
        time_mode: str = "hard",
        output: Output | None = None,
        step1_parsed: ParsedQuery | None = None,
    ) -> dict[str, Any]:
        rel_norm = str(rel).strip().lower()
        t0 = float(hit["t0"])
        t1 = float(hit["t1"])
        if rel_norm == "before":
            t_min_s = 0.0
            t_max_s = max(0.0, float(t0))
        elif rel_norm == "around":
            w = max(0.0, float(window_s))
            t_min_s = max(0.0, float(t0) - w)
            t_max_s = float(t1) + w
        else:
            rel_norm = "after"
            t_min_s = max(0.0, float(t1))
            t_max_s = float("inf")
        meta = dict(hit.get("meta", {}))

        place_value = str(meta.get("place_segment_id", "")).strip()
        place_source = "step1_hit_meta" if place_value else ""
        place_disabled_reason = ""

        object_value = str(meta.get("interaction_primary_object", "")).strip().lower()
        if not object_value:
            for candidate_key in ("object_name", "active_object_top1", "label", "name"):
                val = str(meta.get(candidate_key, "")).strip().lower()
                if val:
                    object_value = val
                    break
        object_source = "step1_hit_meta" if object_value else ""
        object_disabled_reason = ""

        if (not object_value) and step1_parsed is not None:
            parsed_obj = str(
                step1_parsed.object_name
                or step1_parsed.lost_object
                or step1_parsed.object_last_seen
                or step1_parsed.interaction_object
                or ""
            ).strip().lower()
            if parsed_obj:
                object_value = parsed_obj
                object_source = "step1_query"
            else:
                object_disabled_reason = "missing_from_query"

        if (not place_value) and step1_parsed is not None:
            parsed_place = str(step1_parsed.place or "").strip().lower()
            if parsed_place in {"first", "last", "any"}:
                place_value = parsed_place
                place_source = "step1_query"
            else:
                place_disabled_reason = "missing_from_query"

        if (not object_value) and output is not None:
            center_ms = int(round((0.5 * (t0 + t1)) * 1000.0))
            nearest_name = ""
            nearest_dist: int | None = None
            for item in list(getattr(output, "object_memory_v0", []) or []):
                name = str(getattr(item, "object_name", "")).strip().lower()
                if not name:
                    continue
                ts = getattr(item, "last_contact_t_ms", None)
                if ts is None:
                    ts = getattr(item, "last_seen_t_ms", None)
                if ts is None:
                    continue
                try:
                    dist = abs(int(ts) - int(center_ms))
                except Exception:
                    continue
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_name = name
            if nearest_name:
                object_value = nearest_name
                object_source = "object_memory_nearby"
            elif not object_disabled_reason:
                object_disabled_reason = "missing_from_object_memory"

        if not place_value and not place_disabled_reason:
            place_disabled_reason = "missing_from_hit"
        if not object_value and not object_disabled_reason:
            object_disabled_reason = "missing_from_hit"
        derive_norm = str(derive or "time_only").strip().lower().replace(" ", "")
        if derive_norm not in {"time_only", "time+place", "time+object", "time+place+object"}:
            derive_norm = "time_only"
        place_mode_norm = str(place_mode or "soft").strip().lower()
        if place_mode_norm not in {"soft", "hard", "off"}:
            place_mode_norm = "soft"
        object_mode_norm = str(object_mode or "soft").strip().lower()
        if object_mode_norm not in {"soft", "hard", "off"}:
            object_mode_norm = "soft"
        time_mode_norm = str(time_mode or "hard").strip().lower()
        if time_mode_norm not in {"hard", "off"}:
            time_mode_norm = "hard"
        use_place = derive_norm in {"time+place", "time+place+object"}
        use_object = derive_norm in {"time+object", "time+place+object"}
        time_enabled = bool(time_mode_norm != "off")
        use_place = derive_norm in {"time+place", "time+place+object"}
        use_object = derive_norm in {"time+object", "time+place+object"}
        place_enabled = bool(use_place and place_mode_norm != "off" and bool(place_value))
        object_enabled = bool(use_object and object_mode_norm != "off" and bool(object_value))
        if use_place and place_mode_norm != "off" and not place_enabled and not place_disabled_reason:
            place_disabled_reason = "missing_from_hit"
        if use_object and object_mode_norm != "off" and not object_enabled and not object_disabled_reason:
            object_disabled_reason = "missing_from_hit"

        return {
            "rel": rel_norm,
            "t_min_s": float(t_min_s),
            "t_max_s": float(t_max_s),
            "from_hit_kind": str(hit["kind"]),
            "from_hit_id": str(hit["id"]),
            "from_hit_t0": float(t0),
            "from_hit_t1": float(t1),
            "place_segment_id": str(place_value),
            "interaction_primary_object": str(object_value),
            "derive": str(derive_norm),
            "time": {
                "enabled": bool(time_enabled),
                "mode": str(time_mode_norm),
                "source": "step1_top1",
                "t_min_s": float(t_min_s),
                "t_max_s": None if not np.isfinite(t_max_s) else float(t_max_s),
                "disabled_reason": "" if time_enabled else "time_mode_off",
            },
            "place": {
                "enabled": bool(place_enabled),
                "mode": str(place_mode_norm),
                "source": str(place_source),
                "value": str(place_value),
                "disabled_reason": "" if place_enabled else str(place_disabled_reason or ("place_mode_off" if place_mode_norm == "off" else "missing_from_hit")),
            },
            "object": {
                "enabled": bool(object_enabled),
                "mode": str(object_mode_norm),
                "source": str(object_source),
                "value": str(object_value),
                "disabled_reason": "" if object_enabled else str(object_disabled_reason or ("object_mode_off" if object_mode_norm == "off" else "missing_from_hit")),
            },
        }

    def _merge_step2_query(self, step2_query: str, derived: dict[str, Any], *, default_top_k: int) -> str:
        parsed = parse_query(step2_query)
        time_existing = parsed.time_range
        t_min = float(derived.get("t_min_s", 0.0))
        t_max = float(derived.get("t_max_s", float("inf")))
        if time_existing is not None:
            t_min = max(float(t_min), float(time_existing[0]))
            t_max = min(float(t_max), float(time_existing[1]))
        if not np.isfinite(t_max):
            duration_s = float(self.output.meta.get("duration_s", 0.0) or 0.0)
            t_max = duration_s if duration_s > 0 else max(float(t_min) + 60.0, float(t_min))
        if t_max <= t_min:
            t_max = float(t_min) + 0.5

        cleaned = self._strip_query_keys(
            step2_query,
            {
                "time",
                "chain_rel",
                "chain_window_s",
                "chain_top1_only",
                "chain_derive",
                "chain_place_mode",
                "chain_object_mode",
                "chain_time_mode",
                "chain_time_min_s",
                "chain_time_max_s",
                "chain_place_value",
                "chain_object_value",
            },
        )
        parts = [cleaned] if cleaned else []
        d_time = dict(derived.get("time", {})) if isinstance(derived.get("time", {}), dict) else {}
        d_place = dict(derived.get("place", {})) if isinstance(derived.get("place", {}), dict) else {}
        d_object = dict(derived.get("object", {})) if isinstance(derived.get("object", {}), dict) else {}
        if bool(d_time.get("enabled", True)):
            parts.append(f"time={float(t_min):.3f}-{float(t_max):.3f}")
            parts.append(f"chain_time_mode={str(d_time.get('mode', 'hard'))}")
            parts.append(f"chain_time_min_s={float(t_min):.3f}")
            if np.isfinite(float(t_max)):
                parts.append(f"chain_time_max_s={float(t_max):.3f}")
        else:
            parts.append("chain_time_mode=off")

        if bool(d_place.get("enabled", False)):
            place_id = str(d_place.get("value", "")).strip()
            if place_id:
                parts.append(f"chain_place_value={place_id}")
                parts.append(f"chain_place_mode={str(d_place.get('mode', 'soft'))}")
        elif str(d_place.get("mode", "")).strip():
            parts.append(f"chain_place_mode={str(d_place.get('mode', 'off'))}")

        if bool(d_object.get("enabled", False)):
            object_name = str(d_object.get("value", "")).strip().lower()
            if object_name:
                parts.append(f"chain_object_value={object_name}")
                parts.append(f"chain_object_mode={str(d_object.get('mode', 'soft'))}")
        elif str(d_object.get("mode", "")).strip():
            parts.append(f"chain_object_mode={str(d_object.get('mode', 'off'))}")

        parts.append(f"chain_derive={str(derived.get('derive', 'time_only'))}")
        top_k = int(parsed.top_k if parsed.top_k is not None else int(default_top_k))
        if "top_k=" not in " ".join(parts).lower():
            parts.append(f"top_k={max(1, top_k)}")
        return " ".join([p for p in parts if str(p).strip()]).strip()

    def retrieve(self, query: str) -> dict[str, Any]:
        chain: QueryChain | None = parse_query_chain(str(query))
        if chain is not None:
            return self.retrieve_chain(chain, raw_query=str(query))

        parsed: ParsedQuery = parse_query(query)
        top_k = max(1, int(parsed.top_k if parsed.top_k is not None else self.cfg.default_top_k))

        event_pool = self._event_pool()
        event_obj_map = {event.id: event for event in event_pool}
        highlight_map = {hl.id: hl for hl in self.output.highlights}
        token_map = {token.id: token for token in self.output.token_codec.tokens}
        decision_map = {decision.id: decision for decision in self.decision_points}

        current_events: set[str] | None = None
        current_highlights: set[str] | None = None
        current_tokens: set[str] | None = None
        current_decisions: set[str] | None = None

        reasons: list[str] = []
        search_hits_payload: list[dict[str, Any]] = []

        if parsed.time_range is not None:
            t0, t1 = parsed.time_range
            events_new = {event.id for event in event_pool if _overlap(event.t0, event.t1, t0, t1)}
            highlights_new = {hl.id for hl in self.output.highlights if _overlap(hl.t0, hl.t1, t0, t1)}
            tokens_new = {token.id for token in self.output.token_codec.tokens if _overlap(token.t0, token.t1, t0, t1)}
            decisions_new = {
                decision.id
                for decision in self.decision_points
                if _overlap(decision.t0, decision.t1, t0, t1)
            }
            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"time overlap {t0:.3f}-{t1:.3f}")

        if parsed.token_types:
            token_types = set(parsed.token_types)
            matched_tokens = [token for token in self.output.token_codec.tokens if token.type in token_types]
            tokens_new = {token.id for token in matched_tokens}
            events_new = {token.source_event for token in matched_tokens if token.source_event}
            highlights_new: set[str] = set()
            for hl in self.output.highlights:
                if any(_overlap(hl.t0, hl.t1, token.t0, token.t1) for token in matched_tokens):
                    highlights_new.add(hl.id)
            decisions_new: set[str] = set()
            if "DECISION" in token_types:
                decisions_new = {decision.id for decision in self.decision_points}
                events_new.update(decision.source_event for decision in self.decision_points)
                highlights_new.update(
                    decision.source_highlight
                    for decision in self.decision_points
                    if decision.source_highlight is not None
                )
                for decision in self.decision_points:
                    tokens_new.update(self._tokens_for_decision(decision))
            else:
                for decision in self.decision_points:
                    if any(token_id in tokens_new for token_id in self._tokens_for_decision(decision)):
                        decisions_new.add(decision.id)

            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"token types={','.join(sorted(token_types))}")

        if parsed.anchor_types:
            anchor_types = set(parsed.anchor_types)
            highlights_new: set[str] = set()
            for hl in self.output.highlights:
                types = hl.meta.get("anchor_types")
                hl_types = {str(x).lower() for x in types} if isinstance(types, list) else {str(hl.anchor_type).lower()}
                if hl_types.intersection(anchor_types):
                    highlights_new.add(hl.id)

            events_new = {event.id for event in event_pool if self._event_anchor_types(event).intersection(anchor_types)}
            tokens_new = {
                token.id
                for token in self.output.token_codec.tokens
                if (
                    token.type == "ATTENTION_STOP_LOOK"
                    and "stop_look" in anchor_types
                    or token.type == "ATTENTION_TURN_HEAD"
                    and "turn_head" in anchor_types
                )
            }
            decisions_new = {
                decision.id
                for decision in self.decision_points
                if anchor_types.intersection({str(x).lower() for x in decision.trigger.get("anchor_types", [])})
            }
            for decision_id in decisions_new:
                tokens_new.update(self._tokens_for_decision(decision_map[decision_id]))

            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"anchor types={','.join(sorted(anchor_types))}")

        if parsed.decision_types:
            decision_types = set(parsed.decision_types)
            matched = [
                decision
                for decision in self.decision_points
                if str(decision.action.get("type", "")).upper() in decision_types
            ]
            decisions_new = {decision.id for decision in matched}
            events_new = {decision.source_event for decision in matched if decision.source_event}
            highlights_new = {decision.source_highlight for decision in matched if decision.source_highlight}
            tokens_new: set[str] = set()
            for decision in matched:
                tokens_new.update(self._tokens_for_decision(decision))

            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"decision types={','.join(sorted(decision_types))}")

        if parsed.event_ids:
            requested = set(parsed.event_ids)
            events_new = {event_id for event_id in requested if event_id in event_obj_map}
            highlights_new = {hl.id for hl in self.output.highlights if hl.source_event in events_new}
            tokens_new = {token.id for token in self.output.token_codec.tokens if token.source_event in events_new}
            decisions_new = {decision.id for decision in self.decision_points if decision.source_event in events_new}
            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"event ids={','.join(sorted(events_new))}")

        if parsed.event_labels:
            labels = {str(x).lower() for x in parsed.event_labels}
            events_new = {event.id for event in event_pool if self._event_label(event).strip().lower() in labels}
            highlights_new = {hl.id for hl in self.output.highlights if hl.source_event in events_new}
            tokens_new = {token.id for token in self.output.token_codec.tokens if token.source_event in events_new}
            decisions_new = {decision.id for decision in self.decision_points if decision.source_event in events_new}
            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"event labels={','.join(sorted(labels))}")

        if parsed.contact_min is not None:
            threshold = float(parsed.contact_min)
            events_new = {event.id for event in event_pool if self._event_contact_peak(event) >= threshold}
            highlights_new = {hl.id for hl in self.output.highlights if hl.source_event in events_new}
            tokens_new = {token.id for token in self.output.token_codec.tokens if token.source_event in events_new}
            decisions_new = {decision.id for decision in self.decision_points if decision.source_event in events_new}
            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"contact_min={threshold:.3f}")

        if parsed.place_segment_ids:
            segment_ids = {str(x).strip() for x in parsed.place_segment_ids if str(x).strip()}
            events_new = {event.id for event in event_pool if self._event_place_segment_id(event) in segment_ids}
            highlights_new = {hl.id for hl in self.output.highlights if hl.source_event in events_new}
            tokens_new = {token.id for token in self.output.token_codec.tokens if token.source_event in events_new}
            decisions_new = {decision.id for decision in self.decision_points if decision.source_event in events_new}
            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"place_segment_id={','.join(sorted(segment_ids))}")

        if parsed.interaction_min is not None:
            threshold = float(parsed.interaction_min)
            events_new = {event.id for event in event_pool if self._event_interaction_score(event) >= threshold}
            highlights_new = {hl.id for hl in self.output.highlights if hl.source_event in events_new}
            tokens_new = {token.id for token in self.output.token_codec.tokens if token.source_event in events_new}
            decisions_new = {decision.id for decision in self.decision_points if decision.source_event in events_new}
            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"interaction_min={threshold:.3f}")

        if parsed.interaction_object:
            object_key = str(parsed.interaction_object).strip().lower()
            events_new = {
                event.id
                for event in event_pool
                if object_key
                and (
                    object_key in self._event_interaction_primary_object(event)
                    or self._event_interaction_primary_object(event) in object_key
                )
            }
            highlights_new = {hl.id for hl in self.output.highlights if hl.source_event in events_new}
            tokens_new = {token.id for token in self.output.token_codec.tokens if token.source_event in events_new}
            decisions_new = {decision.id for decision in self.decision_points if decision.source_event in events_new}
            current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                current_events,
                current_highlights,
                current_tokens,
                current_decisions,
                events_new,
                highlights_new,
                tokens_new,
                decisions_new,
            )
            reasons.append(f"interaction_object={object_key}")

        if parsed.which in {"first", "last"}:
            events_candidates = set(current_events if current_events is not None else {event.id for event in event_pool})
            if events_candidates:
                ordered = sorted(
                    [event for event in event_pool if event.id in events_candidates],
                    key=lambda ev: (float(ev.t0), float(ev.t1), str(ev.id)),
                )
                if ordered:
                    pick = ordered[0] if parsed.which == "first" else ordered[-1]
                    kept_events = {str(pick.id)}
                    highlights_new = {hl.id for hl in self.output.highlights if hl.source_event in kept_events}
                    tokens_new = {token.id for token in self.output.token_codec.tokens if token.source_event in kept_events}
                    decisions_new = {
                        decision.id for decision in self.decision_points if decision.source_event in kept_events
                    }
                    current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                        current_events,
                        current_highlights,
                        current_tokens,
                        current_decisions,
                        kept_events,
                        highlights_new,
                        tokens_new,
                        decisions_new,
                    )
                    reasons.append(f"which={parsed.which}")

        if parsed.text:
            if self.index is None:
                reasons.append("text query ignored: index not provided")
                current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                    current_events, current_highlights, current_tokens, current_decisions, set(), set(), set(), set()
                )
            elif not self._index_supports_text():
                reasons.append("text query not supported: index embedding backend is not clip")
                current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                    current_events, current_highlights, current_tokens, current_decisions, set(), set(), set(), set()
                )
            else:
                encoder = self._get_text_encoder()
                if encoder is None:
                    reasons.append("text query not supported: open_clip/torch unavailable")
                    current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                        current_events, current_highlights, current_tokens, current_decisions, set(), set(), set(), set()
                    )
                else:
                    qvec = _l2_normalize(encoder.encode(parsed.text))
                    hits = self.index.search(qvec, top_k=top_k)
                    search_hits_payload = [
                        {"id": hit.id, "score": float(hit.score), "kind": hit.meta.get("kind"), "meta": hit.meta}
                        for hit in hits
                    ]
                    events_new: set[str] = set()
                    highlights_new: set[str] = set()
                    tokens_new: set[str] = set()
                    decisions_new: set[str] = set()
                    for hit in hits:
                        kind = str(hit.meta.get("kind", ""))
                        hid = str(hit.meta.get("id", hit.id))
                        if kind == "highlight":
                            highlights_new.add(hid)
                            source_event = str(hit.meta.get("source_event", ""))
                            if source_event:
                                events_new.add(source_event)
                            t0 = float(hit.meta.get("t0", 0.0))
                            t1 = float(hit.meta.get("t1", 0.0))
                            tokens_new.update(self._tokens_in_range(t0, t1))
                            decisions_new.update(
                                decision.id
                                for decision in self.decision_points
                                if _overlap(decision.t0, decision.t1, t0, t1)
                            )
                        elif kind in {"event", "event_v0", "event_v1"}:
                            events_new.add(hid)
                            t0 = float(hit.meta.get("t0", 0.0))
                            t1 = float(hit.meta.get("t1", 0.0))
                            tokens_new.update(self._tokens_in_range(t0, t1))
                            decisions_new.update(
                                decision.id
                                for decision in self.decision_points
                                if decision.source_event == hid
                            )
                    current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                        current_events,
                        current_highlights,
                        current_tokens,
                        current_decisions,
                        events_new,
                        highlights_new,
                        tokens_new,
                        decisions_new,
                    )
                    reasons.append(f"text search hits={len(hits)}")

        if parsed.place in {"first", "last"}:
            events_candidates = set(current_events if current_events is not None else {event.id for event in event_pool})
            if events_candidates:
                grouped: dict[str, list[Any]] = {}
                for event in event_pool:
                    if event.id not in events_candidates:
                        continue
                    seg_id = self._event_place_segment_id(event) or "__unknown__"
                    grouped.setdefault(seg_id, []).append(event)
                kept_events: set[str] = set()
                for values in grouped.values():
                    ordered = sorted(values, key=lambda ev: (float(ev.t0), float(ev.t1), str(ev.id)))
                    pick = ordered[0] if parsed.place == "first" else ordered[-1]
                    kept_events.add(str(pick.id))
                highlights_new = {hl.id for hl in self.output.highlights if hl.source_event in kept_events}
                tokens_new = {token.id for token in self.output.token_codec.tokens if token.source_event in kept_events}
                decisions_new = {decision.id for decision in self.decision_points if decision.source_event in kept_events}
                current_events, current_highlights, current_tokens, current_decisions = self._apply_constraint(
                    current_events,
                    current_highlights,
                    current_tokens,
                    current_decisions,
                    kept_events,
                    highlights_new,
                    tokens_new,
                    decisions_new,
                )
                reasons.append(f"place={parsed.place}")

        if current_events is None and current_highlights is None and current_tokens is None and current_decisions is None:
            sorted_hls = sorted(self.output.highlights, key=lambda h: (h.conf, h.t1 - h.t0), reverse=True)
            selected_highlights = [hl.id for hl in sorted_hls[:top_k]]
            sorted_decisions = sorted(
                self.decision_points,
                key=lambda d: (1 if d.source_highlight else 0, float(d.conf)),
                reverse=True,
            )
            selected_decisions = [decision.id for decision in sorted_decisions[:top_k]]
            selected_events = sorted(
                {
                    highlight_map[hid].source_event
                    for hid in selected_highlights
                    if hid in highlight_map
                }.union(
                    {
                        decision_map[did].source_event
                        for did in selected_decisions
                        if did in decision_map
                    }
                ),
                key=lambda eid: event_obj_map[eid].t0 if eid in event_obj_map else 0.0,
            )
            selected_tokens_set = {
                token.id
                for token in self.output.token_codec.tokens
                if token.source_event in set(selected_events)
            }
            for hid in selected_highlights:
                hl = highlight_map.get(hid)
                if hl is not None:
                    selected_tokens_set.update(self._tokens_in_range(float(hl.t0), float(hl.t1)))
            for did in selected_decisions:
                decision = decision_map.get(did)
                if decision is not None:
                    selected_tokens_set.update(self._tokens_for_decision(decision))
            selected_tokens = sorted(
                [tid for tid in selected_tokens_set if tid in token_map],
                key=lambda tid: (token_map[tid].t0, token_map[tid].t1, token_map[tid].type),
            )
            reasons.append("default ranking (no filter)")
        else:
            selected_highlights = sorted(
                list(current_highlights or set()),
                key=lambda hid: (
                    float(highlight_map[hid].conf) if hid in highlight_map else 0.0,
                    float(highlight_map[hid].t1 - highlight_map[hid].t0) if hid in highlight_map else 0.0,
                ),
                reverse=True,
            )
            if len(selected_highlights) > top_k:
                selected_highlights = selected_highlights[:top_k]

            selected_decisions = sorted(
                list(current_decisions or set()),
                key=lambda did: (
                    1 if did in decision_map and decision_map[did].source_highlight else 0,
                    float(decision_map[did].conf) if did in decision_map else 0.0,
                ),
                reverse=True,
            )
            if len(selected_decisions) > top_k:
                selected_decisions = selected_decisions[:top_k]

            selected_events_set = set(current_events or set())
            selected_events_set.update(
                highlight_map[hid].source_event for hid in selected_highlights if hid in highlight_map
            )
            selected_events_set.update(
                decision_map[did].source_event for did in selected_decisions if did in decision_map
            )
            selected_events = sorted(
                [eid for eid in selected_events_set if eid in event_obj_map],
                key=lambda eid: event_obj_map[eid].t0,
            )

            selected_tokens_set = set(current_tokens or set())
            if not selected_tokens_set:
                for token in self.output.token_codec.tokens:
                    if token.source_event in set(selected_events):
                        selected_tokens_set.add(token.id)
                for hid in selected_highlights:
                    hl = highlight_map.get(hid)
                    if hl is not None:
                        selected_tokens_set.update(self._tokens_in_range(float(hl.t0), float(hl.t1)))
                for did in selected_decisions:
                    decision = decision_map.get(did)
                    if decision is not None:
                        selected_tokens_set.update(self._tokens_for_decision(decision))
            selected_tokens = sorted(
                [tid for tid in selected_tokens_set if tid in token_map],
                key=lambda tid: (token_map[tid].t0, token_map[tid].t1, token_map[tid].type),
            )

        result = {
            "selected_events": selected_events,
            "selected_highlights": selected_highlights,
            "selected_decisions": selected_decisions,
            "selected_tokens": selected_tokens,
            "mode": parsed.mode,
            "budget_overrides": dict(parsed.budget_overrides),
            "debug": {
                "reason": "; ".join(reasons),
                "filters_applied": list(parsed.filters_applied),
                "parse_warnings": list(parsed.parse_warnings),
                "search_hits": search_hits_payload,
                "decision_pool_kind": str(self.decision_pool_kind),
                "decision_pool_count": int(self.decision_pool_count),
                "decision_model_meta": dict(self.decision_model_meta),
            },
        }
        return result

    def retrieve_chain(self, chain: QueryChain, raw_query: str | None = None) -> dict[str, Any]:
        step1 = chain.steps[0]
        step2 = chain.steps[1]
        step1_query = str(step1.raw).strip()
        step2_query = str(step2.raw).strip()
        top_k = max(
            1,
            int(
                step2.parsed.top_k
                if step2.parsed.top_k is not None
                else (step1.parsed.top_k if step1.parsed.top_k is not None else self.cfg.default_top_k)
            ),
        )

        step1_result = self.retrieve(step1_query)
        step1_hits = self._result_to_hits(step1_query, step1_result)
        step1_top = step1_hits[0] if step1_hits else None
        step1_count = self._count_selected(step1_result)

        if step1_top is None:
            return {
                "selected_events": [],
                "selected_highlights": [],
                "selected_decisions": [],
                "selected_tokens": [],
                "mode": step2.parsed.mode,
                "budget_overrides": dict(step2.parsed.budget_overrides),
                "debug": {
                    "reason": "chain_step1_no_hit",
                    "filters_applied": list(step2.parsed.filters_applied),
                    "parse_warnings": list(step2.parsed.parse_warnings),
                    "search_hits": [],
                    "chain": {
                        "is_chain": True,
                        "raw_query": str(raw_query or f"{step1_query} then {step2_query}"),
                        "chain_rel": str(chain.rel),
                        "window_s": float(chain.window_s),
                        "top1_only": bool(chain.top1_only),
                        "chain_derive": str(getattr(chain, "derive", "time_only")),
                        "chain_place_mode": str(getattr(chain, "place_mode", "soft")),
                        "chain_object_mode": str(getattr(chain, "object_mode", "soft")),
                        "chain_time_mode": str(getattr(chain, "time_mode", "hard")),
                        "step1": {
                            "query": step1_query,
                            "parsed_constraints": self._parsed_constraints(step1.parsed),
                            "filtered_hits_before": int(step1_count),
                            "filtered_hits_after": int(step1_count),
                            "has_hit": False,
                            "hit_count": int(step1_count),
                        },
                        "derived_constraints": {},
                        "step2": {
                            "query": step2_query,
                            "query_derived": step2_query,
                            "parsed_constraints": self._parsed_constraints(step2.parsed),
                            "filtered_hits_before": 0,
                            "filtered_hits_after": 0,
                            "has_hit": False,
                            "hit_count": 0,
                        },
                    },
                },
            }

        derived = self._derive_constraints_from_hit(
            step1_top,
            rel=str(chain.rel),
            window_s=float(chain.window_s),
            derive=str(getattr(chain, "derive", "time_only")),
            place_mode=str(getattr(chain, "place_mode", "soft")),
            object_mode=str(getattr(chain, "object_mode", "soft")),
            time_mode=str(getattr(chain, "time_mode", "hard")),
            output=self.output,
            step1_parsed=step1.parsed,
        )
        step2_query_derived = self._merge_step2_query(step2_query, derived, default_top_k=int(top_k))
        step2_plain = self.retrieve(step2_query)
        step2_result = self.retrieve(step2_query_derived)
        step2_before = self._count_selected(step2_plain)
        step2_after = self._count_selected(step2_result)
        step2_hits = self._result_to_hits(step2_query_derived, step2_result)

        final = dict(step2_result)
        debug = dict(final.get("debug", {}) if isinstance(final.get("debug", {}), dict) else {})
        base_reason = str(debug.get("reason", "")).strip()
        chain_reason = f"chain:{step1_query} -> {step2_query_derived}"
        debug["reason"] = f"{base_reason}; {chain_reason}" if base_reason else chain_reason
        debug["chain"] = {
            "is_chain": True,
            "raw_query": str(raw_query or f"{step1_query} then {step2_query}"),
            "chain_rel": str(chain.rel),
            "window_s": float(chain.window_s),
            "top1_only": bool(chain.top1_only),
            "chain_derive": str(getattr(chain, "derive", "time_only")),
            "chain_place_mode": str(getattr(chain, "place_mode", "soft")),
            "chain_object_mode": str(getattr(chain, "object_mode", "soft")),
            "chain_time_mode": str(getattr(chain, "time_mode", "hard")),
            "step1": {
                "query": step1_query,
                "parsed_constraints": self._parsed_constraints(step1.parsed),
                "filtered_hits_before": int(step1_count),
                "filtered_hits_after": int(step1_count),
                "has_hit": bool(step1_top is not None),
                "hit_count": int(step1_count),
                "top1": {
                    "kind": str(step1_top["kind"]),
                    "id": str(step1_top["id"]),
                    "t0": float(step1_top["t0"]),
                    "t1": float(step1_top["t1"]),
                },
            },
            "derived_constraints": dict(derived),
            "step2": {
                "query": step2_query,
                "query_derived": step2_query_derived,
                "parsed_constraints": self._parsed_constraints(step2.parsed),
                "filtered_hits_before": int(step2_before),
                "filtered_hits_after": int(step2_after),
                "has_hit": bool(len(step2_hits) > 0),
                "hit_count": int(step2_after),
            },
        }
        final["debug"] = debug
        return final

    @staticmethod
    def has_hits(result: dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        for key in ("selected_highlights", "selected_events", "selected_tokens", "selected_decisions"):
            values = result.get(key, [])
            if isinstance(values, list) and len(values) > 0:
                return True
        return False

    def retrieve_cascade(self, candidate_queries: list[str]) -> tuple[str | None, dict[str, Any]]:
        """Try candidate structured queries in order, return first non-empty retrieval result."""

        last_result: dict[str, Any] = {
            "selected_events": [],
            "selected_highlights": [],
            "selected_decisions": [],
            "selected_tokens": [],
            "debug": {"reason": "no_candidate"},
        }
        for query in candidate_queries:
            q = str(query).strip()
            if not q:
                continue
            result = self.retrieve(q)
            last_result = result
            if self.has_hits(result):
                return q, result
        return None, last_result

    def _result_to_hits(self, query: str, result: dict[str, Any]) -> list[Hit]:
        query = str(query).strip()
        chain_debug = None
        debug_payload = result.get("debug", {})
        if isinstance(debug_payload, dict):
            chain_candidate = debug_payload.get("chain", None)
            if isinstance(chain_candidate, dict):
                chain_debug = dict(chain_candidate)
        event_map = {event.id: event for event in (list(self.output.events) + list(self.output.events_v0))}
        if self.output.events_v1:
            event_map = {event.id: event for event in self.output.events_v1}
        highlight_map = {hl.id: hl for hl in self.output.highlights}
        token_map = {token.id: token for token in self.output.token_codec.tokens}
        decision_map = {decision.id: decision for decision in self.decision_points}

        text_score_lookup: dict[tuple[str, str], float] = {}
        for item in result.get("debug", {}).get("search_hits", []) or []:
            kind = str(item.get("kind", ""))
            hit_id = str(item.get("id", ""))
            if not kind or not hit_id:
                continue
            try:
                text_score_lookup[(kind, hit_id)] = float(item.get("score", 0.0))
            except Exception:
                text_score_lookup[(kind, hit_id)] = 0.0

        def _rank_score(rank: int, total: int) -> float:
            if total <= 0:
                return 0.0
            return float(1.0 - (rank / float(total + 1)))

        hits: list[Hit] = []
        selected_highlights = [str(x) for x in (result.get("selected_highlights", []) or [])]
        for idx, hid in enumerate(selected_highlights):
            hl = highlight_map.get(hid)
            if hl is None:
                continue
            src_event = event_map.get(str(getattr(hl, "source_event", "")))
            token_types = sorted(
                {
                    str(tok.type)
                    for tok in self.output.token_codec.tokens
                    if _overlap(float(hl.t0), float(hl.t1), float(tok.t0), float(tok.t1))
                }
            )
            base = text_score_lookup.get(("highlight", hid), _rank_score(idx, len(selected_highlights)))
            hits.append(
                Hit(
                    kind="highlight",
                    id=hid,
                    t0=float(hl.t0),
                    t1=float(hl.t1),
                    score=float(base),
                    source_query=query,
                    meta={
                        "conf": float(getattr(hl, "conf", 0.0)),
                        "anchor_type": str(getattr(hl, "anchor_type", "")),
                        "anchor_types": hl.meta.get("anchor_types", []),
                        "token_types": token_types,
                        "source_event": str(getattr(hl, "source_event", "")),
                        "place_segment_id": self._event_place_segment_id(src_event) if src_event is not None else "",
                        "place_segment_conf": self._event_place_segment_conf(src_event) if src_event is not None else 0.0,
                        "place_segment_reason": self._event_place_segment_reason(src_event) if src_event is not None else "",
                        "interaction_primary_object": self._event_interaction_primary_object(src_event) if src_event is not None else "",
                        "interaction_score": self._event_interaction_score(src_event) if src_event is not None else 0.0,
                        "chain": chain_debug,
                    },
                )
            )

        selected_tokens = [str(x) for x in (result.get("selected_tokens", []) or [])]
        for idx, tid in enumerate(selected_tokens):
            tok = token_map.get(tid)
            if tok is None:
                continue
            src_event = event_map.get(str(getattr(tok, "source_event", "")))
            base = text_score_lookup.get(("token", tid), _rank_score(idx, len(selected_tokens)))
            hits.append(
                Hit(
                    kind="token",
                    id=tid,
                    t0=float(tok.t0),
                    t1=float(tok.t1),
                    score=float(base),
                    source_query=query,
                    meta={
                        "conf": float(getattr(tok, "conf", 0.0)),
                        "token_type": str(getattr(tok, "type", "")),
                        "source_event": str(getattr(tok, "source_event", "")),
                        "place_segment_id": self._event_place_segment_id(src_event) if src_event is not None else "",
                        "place_segment_conf": self._event_place_segment_conf(src_event) if src_event is not None else 0.0,
                        "place_segment_reason": self._event_place_segment_reason(src_event) if src_event is not None else "",
                        "interaction_primary_object": self._event_interaction_primary_object(src_event) if src_event is not None else "",
                        "interaction_score": self._event_interaction_score(src_event) if src_event is not None else 0.0,
                        "chain": chain_debug,
                    },
                )
            )

        selected_decisions = [str(x) for x in (result.get("selected_decisions", []) or [])]
        for idx, did in enumerate(selected_decisions):
            dp = decision_map.get(did)
            if dp is None:
                continue
            src_event = event_map.get(str(getattr(dp, "source_event", "")))
            trigger = dict(getattr(dp, "trigger", {}) or {})
            action = dict(getattr(dp, "action", {}) or {})
            outcome = dict(getattr(dp, "outcome", {}) or {})
            state = dict(getattr(dp, "state", {}) or {})
            constraints = list(getattr(dp, "constraints", []) or [])
            trigger_anchor_types = trigger.get("anchor_types", [])
            if not isinstance(trigger_anchor_types, list):
                trigger_anchor_types = [trigger_anchor_types] if trigger_anchor_types else []
            evidence_ids = state.get("evidence", {}).get("token_ids", [])
            if not isinstance(evidence_ids, list):
                evidence_ids = []
            nearby_tokens = state.get("nearby_tokens", [])
            if not isinstance(nearby_tokens, list):
                nearby_tokens = []
            base = text_score_lookup.get(("decision", did), _rank_score(idx, len(selected_decisions)))
            hits.append(
                Hit(
                    kind="decision",
                    id=did,
                    t0=float(dp.t0),
                    t1=float(dp.t1),
                    score=float(base),
                    source_query=query,
                    meta={
                        "conf": float(getattr(dp, "conf", 0.0)),
                        "decision_source_kind": str(getattr(dp, "meta", {}).get("decision_source_kind", self.decision_pool_kind)),
                        "model_provider": str(getattr(dp, "meta", {}).get("model_provider", "")),
                        "model_name": str(getattr(dp, "meta", {}).get("model_name", "")),
                        "model_base_url": str(getattr(dp, "meta", {}).get("model_base_url", "")),
                        "action_type": str(action.get("type", "")),
                        "source_event": str(getattr(dp, "source_event", "")),
                        "source_highlight": getattr(dp, "source_highlight", None),
                        "trigger_anchor_types": [str(x).lower() for x in trigger_anchor_types if str(x).strip()],
                        "trigger_anchor_type": str(trigger.get("anchor_type", "")),
                        "trigger_conf": float(trigger.get("conf", 0.0) or 0.0),
                        "decision_constraints": constraints,
                        "state_nearby_tokens": [str(x).upper() for x in nearby_tokens if str(x).strip()],
                        "state_scene_change_nearby": bool(state.get("scene_change_nearby", False)),
                        "state_boundary_nearby": bool(state.get("boundary_nearby", False)),
                        "outcome_type": str(outcome.get("type", "")),
                        "evidence_token_count": int(len(evidence_ids)),
                        "evidence_coverage": float(min(1.0, max(0.0, len(evidence_ids) / 5.0))),
                        "place_segment_id": self._event_place_segment_id(src_event) if src_event is not None else "",
                        "place_segment_conf": self._event_place_segment_conf(src_event) if src_event is not None else 0.0,
                        "place_segment_reason": self._event_place_segment_reason(src_event) if src_event is not None else "",
                        "interaction_primary_object": self._event_interaction_primary_object(src_event) if src_event is not None else "",
                        "interaction_score": self._event_interaction_score(src_event) if src_event is not None else 0.0,
                        "chain": chain_debug,
                    },
                )
            )

        selected_events = [str(x) for x in (result.get("selected_events", []) or [])]
        for idx, eid in enumerate(selected_events):
            event = event_map.get(eid)
            if event is None:
                continue
            base = text_score_lookup.get(("event_v1", eid), text_score_lookup.get(("event", eid), _rank_score(idx, len(selected_events))))
            hits.append(
                Hit(
                    kind="event",
                    id=eid,
                    t0=float(event.t0),
                    t1=float(event.t1),
                    score=float(base),
                    source_query=query,
                    meta={
                        "boundary_conf": self._event_boundary_conf(event),
                        "contact_peak": self._event_contact_peak(event),
                        "label": self._event_label(event),
                        "layer": str(getattr(event, "meta", {}).get("layer", "events_v1" if hasattr(event, "label") else "")),
                        "place_segment_id": self._event_place_segment_id(event),
                        "place_segment_conf": self._event_place_segment_conf(event),
                        "place_segment_reason": self._event_place_segment_reason(event),
                        "interaction_primary_object": self._event_interaction_primary_object(event),
                        "interaction_score": self._event_interaction_score(event),
                        "chain": chain_debug,
                    },
                )
            )

        return hits

    def retrieve_multi(self, candidates: list[Any]) -> list[Hit]:
        merged: dict[tuple[str, str], Hit] = {}
        for query in candidates:
            q = str(query.get("query", "")) if isinstance(query, dict) else str(query)
            q = q.strip()
            if not q:
                continue
            result = self.retrieve(q)
            for hit in self._result_to_hits(q, result):
                key = (str(hit["kind"]), str(hit["id"]))
                existing = merged.get(key)
                if existing is None or float(hit["score"]) > float(existing["score"]):
                    merged[key] = hit
                elif existing is not None:
                    existing_meta = dict(existing.get("meta", {}))
                    source_queries = existing_meta.get("source_queries", [])
                    if not isinstance(source_queries, list):
                        source_queries = []
                    if str(hit["source_query"]) not in source_queries:
                        source_queries.append(str(hit["source_query"]))
                    existing_meta["source_queries"] = source_queries
                    merged[key] = Hit(
                        kind=str(existing["kind"]),
                        id=str(existing["id"]),
                        t0=float(existing["t0"]),
                        t1=float(existing["t1"]),
                        score=float(existing["score"]),
                        source_query=str(existing["source_query"]),
                        meta=existing_meta,
                    )
        out = list(merged.values())
        out.sort(key=lambda x: (-float(x["score"]), float(x["t0"]), str(x["kind"]), str(x["id"])))
        return out
