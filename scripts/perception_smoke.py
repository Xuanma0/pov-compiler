from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.l1_events.event_segmentation_v0 import events_v0_to_dict, segment_events_v0
from pov_compiler.perception.runner import run_perception


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perception v0 smoke for one video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run-perception", action="store_true", help="No-op compatibility flag")
    parser.add_argument("--backend", "--perception-backend", dest="backend", choices=["real", "stub"], default="stub")
    parser.set_defaults(perception_fallback_stub=True)
    parser.add_argument(
        "--perception-fallback-stub",
        dest="perception_fallback_stub",
        action="store_true",
        help="Allow fallback from real backend to stub backend",
    )
    parser.add_argument(
        "--no-perception-fallback-stub",
        dest="perception_fallback_stub",
        action="store_false",
        help="Disable fallback to stub backend",
    )
    parser.add_argument(
        "--perception-strict",
        action="store_true",
        help="Strict perception mode: no fallback, fail on missing deps/frame errors",
    )
    parser.add_argument("--fps", "--perception-fps", dest="fps", type=float, default=10.0)
    parser.add_argument("--max-frames", "--perception-max-frames", dest="max_frames", type=int, default=300)
    parser.add_argument("--contact-min-score", type=float, default=0.25)
    parser.add_argument(
        "--hand-task-model",
        dest="hand_task_model",
        default="assets/mediapipe/hand_landmarker.task",
        help="Path to MediaPipe hand_landmarker.task",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory for perception json reuse")
    parser.add_argument("--save-vis-frames", type=int, default=0, help="Save N visualization jpgs (optional)")
    parser.add_argument("--event-thresh", type=float, default=0.45)
    parser.add_argument("--event-min-s", type=float, default=3.0)
    return parser.parse_args()


def _draw_overlay(frame: Any, frame_info: dict[str, Any]) -> Any:
    img = frame.copy()
    for obj in frame_info.get("objects", []):
        bbox = obj.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        label = f"{obj.get('label', '')}:{float(obj.get('conf', 0.0)):.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (64, 220, 64), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (64, 220, 64), 1, cv2.LINE_AA)
    for hand in frame_info.get("hands", []):
        bbox = hand.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        label = f"hand:{hand.get('handedness', 'unk')}:{float(hand.get('conf', 0.0)):.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (64, 128, 255), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (64, 128, 255), 1, cv2.LINE_AA)
    active = frame_info.get("contact", {}).get("active")
    if isinstance(active, dict):
        cv2.putText(
            img,
            f"contact:{active.get('label','')}:{float(active.get('score',0.0)):.2f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 230),
            2,
            cv2.LINE_AA,
        )
    return img


def _save_vis_frames(video_path: Path, frames: list[dict[str, Any]], out_dir: Path, n: int, fps: float) -> int:
    if n <= 0 or not frames:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    target = frames[: min(int(n), len(frames))]
    time_to_info = {round(float(f.get("t", 0.0)), 3): f for f in target}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    saved = 0
    sample_period = 1.0 / max(1e-6, float(fps))
    next_t = 0.0
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_idx = 0
    try:
        while saved < len(target):
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / max(1e-6, src_fps)
            if t + 1e-9 >= next_t:
                key = round(float(next_t), 3)
                info = time_to_info.get(key)
                if info is not None:
                    vis = _draw_overlay(frame, info)
                    out_path = out_dir / f"frame_{saved:04d}.jpg"
                    cv2.imwrite(str(out_path), vis)
                    saved += 1
                next_t += sample_period
            frame_idx += 1
    finally:
        cap.release()
    return saved


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fallback_to_stub = bool(args.perception_fallback_stub)
    strict = bool(args.perception_strict)
    if strict:
        fallback_to_stub = False
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "cache")

    try:
        perception = run_perception(
            video_path=video_path,
            sample_fps=float(args.fps),
            max_frames=int(args.max_frames),
            backend_name=str(args.backend),
            backend_kwargs={
                "model_candidates": ["yolo26n.pt", "yolov8n.pt"],
                "hand_task_model_path": str(args.hand_task_model),
            },
            fallback_to_stub=bool(fallback_to_stub),
            strict=bool(strict),
            cache_dir=cache_dir,
            contact_min_score=float(args.contact_min_score),
            objects_topk=10,
        )
    except Exception as exc:
        print(f"error=perception_failed detail={exc}")
        if str(args.backend).lower() == "real":
            print("hint=install_optional_deps pip install ultralytics mediapipe")
        return 1
    signals = perception.get("signals", {})
    times = signals.get("time", [])
    visual = signals.get("visual_change", [])
    contact = signals.get("contact_score", [])
    duration_s = float(perception.get("meta", {}).get("duration_s", 0.0))
    events_v0, boundary_v0 = segment_events_v0(
        times=times,
        visual_change=visual,
        contact_score=contact,
        duration_s=duration_s,
        thresh=float(args.event_thresh),
        min_event_s=float(args.event_min_s),
    )
    events_payload = events_v0_to_dict(events_v0)

    perception_json = out_dir / "perception.json"
    event_json = out_dir / "events_v0.json"
    report_md = out_dir / "report.md"
    perception_json.write_text(json.dumps(perception, ensure_ascii=False, indent=2), encoding="utf-8")
    event_json.write_text(
        json.dumps(
            {
                "video_id": str(perception.get("video_id", video_path.stem)),
                "events_v0": events_payload,
                "signals": {
                    "time": [float(x) for x in times],
                    "visual_change": [float(x) for x in visual],
                    "contact_score": [float(x) for x in contact],
                    "boundary_score_v0": [float(x) for x in boundary_v0.tolist()],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    vis_saved = _save_vis_frames(
        video_path=video_path,
        frames=list(perception.get("frames", [])),
        out_dir=out_dir / "vis_frames",
        n=int(args.save_vis_frames),
        fps=float(args.fps),
    )

    meta = perception.get("meta", {})
    summary = perception.get("summary", {})
    lines: list[str] = []
    lines.append("# Perception Smoke Report")
    lines.append("")
    lines.append(f"- video: `{video_path}`")
    lines.append(f"- backend: {meta.get('backend', '')}")
    lines.append(f"- processed_frames: {meta.get('processed_frames', 0)}")
    lines.append(f"- elapsed_s: {float(meta.get('elapsed_s', 0.0)):.3f}")
    lines.append(f"- throughput_fps: {float(meta.get('throughput_fps', 0.0)):.3f}")
    lines.append(f"- hand_presence_rate: {float(summary.get('hand_presence_rate', 0.0)):.4f}")
    lines.append(f"- contact_events_count: {int(summary.get('contact_events_count', 0))}")
    lines.append(f"- events_v0_count: {len(events_payload)}")
    lines.append(f"- vis_frames_saved: {int(vis_saved)}")
    lines.append("")
    lines.append("## Objects TopK")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|---|---:|")
    for row in summary.get("objects_topk", []):
        lines.append(f"| {row.get('label','')} | {int(row.get('count', 0))} |")
    lines.append("")
    lines.append("## Events V0")
    lines.append("")
    lines.append("| id | t0 | t1 | label | boundary_conf |")
    lines.append("|---|---:|---:|---|---:|")
    for ev in events_payload:
        lines.append(
            f"| {ev.get('id','')} | {float(ev.get('t0', 0.0)):.2f} | {float(ev.get('t1', 0.0)):.2f} | "
            f"{ev.get('label','')} | {float(ev.get('scores',{}).get('boundary_conf',0.0)):.3f} |"
        )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"video_id={perception.get('video_id', video_path.stem)}")
    print(f"backend={meta.get('backend', '')}")
    print(f"processed_frames={meta.get('processed_frames', 0)}")
    print(f"throughput_fps={float(meta.get('throughput_fps', 0.0)):.3f}")
    print(f"contact_events_count={int(summary.get('contact_events_count', 0))}")
    print(f"fallback_used={str(bool(summary.get('fallback_used', False))).lower()}")
    if summary.get("fallback_reason"):
        print(f"fallback_reason={summary.get('fallback_reason')}")
    print(f"cache_hit={str(bool(summary.get('cache_hit', False))).lower()}")
    print(f"events_v0={len(events_payload)}")
    print(f"saved_perception={perception_json}")
    print(f"saved_events={event_json}")
    print(f"saved_report={report_md}")
    if vis_saved > 0:
        print(f"saved_vis_frames={vis_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
