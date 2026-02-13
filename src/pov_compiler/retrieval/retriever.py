from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.memory.vector_index import VectorIndex
from pov_compiler.retrieval.query_parser import ParsedQuery, parse_query
from pov_compiler.retrieval.reranker import Hit
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
        if isinstance(index, VectorIndex):
            self.index = index
        elif index is None:
            self.index = None
        else:
            self.index = VectorIndex.load(index)
        self._text_encoder: _OpenCLIPTextEncoder | None = None

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

    def retrieve(self, query: str) -> dict[str, Any]:
        parsed: ParsedQuery = parse_query(query)
        top_k = max(1, int(parsed.top_k if parsed.top_k is not None else self.cfg.default_top_k))

        event_pool = self._event_pool()
        event_obj_map = {event.id: event for event in event_pool}
        highlight_map = {hl.id: hl for hl in self.output.highlights}
        token_map = {token.id: token for token in self.output.token_codec.tokens}
        decision_map = {decision.id: decision for decision in self.output.decision_points}

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
                for decision in self.output.decision_points
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
                decisions_new = {decision.id for decision in self.output.decision_points}
                events_new.update(decision.source_event for decision in self.output.decision_points)
                highlights_new.update(
                    decision.source_highlight
                    for decision in self.output.decision_points
                    if decision.source_highlight is not None
                )
                for decision in self.output.decision_points:
                    tokens_new.update(self._tokens_for_decision(decision))
            else:
                for decision in self.output.decision_points:
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
                for decision in self.output.decision_points
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
                for decision in self.output.decision_points
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
            decisions_new = {decision.id for decision in self.output.decision_points if decision.source_event in events_new}
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
            decisions_new = {decision.id for decision in self.output.decision_points if decision.source_event in events_new}
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
            decisions_new = {decision.id for decision in self.output.decision_points if decision.source_event in events_new}
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
                                for decision in self.output.decision_points
                                if _overlap(decision.t0, decision.t1, t0, t1)
                            )
                        elif kind in {"event", "event_v0", "event_v1"}:
                            events_new.add(hid)
                            t0 = float(hit.meta.get("t0", 0.0))
                            t1 = float(hit.meta.get("t1", 0.0))
                            tokens_new.update(self._tokens_in_range(t0, t1))
                            decisions_new.update(
                                decision.id
                                for decision in self.output.decision_points
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

        if current_events is None and current_highlights is None and current_tokens is None and current_decisions is None:
            sorted_hls = sorted(self.output.highlights, key=lambda h: (h.conf, h.t1 - h.t0), reverse=True)
            selected_highlights = [hl.id for hl in sorted_hls[:top_k]]
            sorted_decisions = sorted(
                self.output.decision_points,
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
                "search_hits": search_hits_payload,
            },
        }
        return result

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
        event_map = {event.id: event for event in (list(self.output.events) + list(self.output.events_v0))}
        if self.output.events_v1:
            event_map = {event.id: event for event in self.output.events_v1}
        highlight_map = {hl.id: hl for hl in self.output.highlights}
        token_map = {token.id: token for token in self.output.token_codec.tokens}
        decision_map = {decision.id: decision for decision in self.output.decision_points}

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
                    },
                )
            )

        selected_tokens = [str(x) for x in (result.get("selected_tokens", []) or [])]
        for idx, tid in enumerate(selected_tokens):
            tok = token_map.get(tid)
            if tok is None:
                continue
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
                    },
                )
            )

        selected_decisions = [str(x) for x in (result.get("selected_decisions", []) or [])]
        for idx, did in enumerate(selected_decisions):
            dp = decision_map.get(did)
            if dp is None:
                continue
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
                        "action_type": str(dp.action.get("type", "")),
                        "source_event": str(getattr(dp, "source_event", "")),
                        "source_highlight": getattr(dp, "source_highlight", None),
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
