from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export event/highlight clips from pipeline JSON")
    parser.add_argument("--video", required=True, help="Input source video")
    parser.add_argument("--json", required=True, help="Pipeline output JSON")
    parser.add_argument("--out_dir", required=True, help="Output clips directory")
    parser.add_argument(
        "--mode",
        choices=["events", "highlights", "both"],
        default="events",
        help="What to export",
    )
    return parser.parse_args()


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _export_segments(
    cap: cv2.VideoCapture,
    segments: list[dict],
    out_dir: Path,
    fps: float,
    width: int,
    height: int,
    include_anchor_type: bool = False,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    exported = 0

    for i, segment in enumerate(segments):
        segment_id = str(segment.get("id", f"clip_{i + 1:04d}"))
        t0 = float(segment.get("t0", 0.0))
        t1 = float(segment.get("t1", 0.0))
        if t1 <= t0:
            continue

        filename = segment_id
        if include_anchor_type:
            anchor_type = _sanitize(str(segment.get("anchor_type", "na")))
            filename = f"{segment_id}_{anchor_type}"
        clip_path = out_dir / f"{filename}.mp4"

        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open clip writer: {clip_path}")

        start_frame = max(0, int(round(t0 * fps)))
        end_frame = max(start_frame + 1, int(round(t1 * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        exported += 1
        print(f"exported {segment_id} -> {clip_path}")

    return exported


def export_clips(video_path: Path, json_path: Path, out_dir: Path, mode: str = "events") -> tuple[int, int]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    events = payload.get("events", [])
    highlights = payload.get("highlights", [])
    if not isinstance(events, list):
        raise ValueError("Invalid JSON: events must be a list")
    if not isinstance(highlights, list):
        raise ValueError("Invalid JSON: highlights must be a list")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Failed to read video dimensions")

    event_count = 0
    highlight_count = 0
    if mode == "events":
        event_count = _export_segments(cap, events, out_dir, fps, width, height, include_anchor_type=False)
    elif mode == "highlights":
        highlight_count = _export_segments(
            cap,
            highlights,
            out_dir,
            fps,
            width,
            height,
            include_anchor_type=True,
        )
    else:
        event_count = _export_segments(
            cap,
            events,
            out_dir / "events",
            fps,
            width,
            height,
            include_anchor_type=False,
        )
        highlight_count = _export_segments(
            cap,
            highlights,
            out_dir / "highlights",
            fps,
            width,
            height,
            include_anchor_type=True,
        )

    cap.release()
    return event_count, highlight_count


def main() -> int:
    args = parse_args()
    event_count, highlight_count = export_clips(
        video_path=Path(args.video),
        json_path=Path(args.json),
        out_dir=Path(args.out_dir),
        mode=args.mode,
    )
    print(f"events_exported={event_count}")
    print(f"highlights_exported={highlight_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
