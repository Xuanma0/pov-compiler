from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def _human_gb(size_bytes: int) -> float:
    return float(size_bytes) / (1024.0**3)


def _render_tree(root: Path, depth: int, max_children: int = 40) -> list[str]:
    lines: list[str] = [root.name]

    def walk(node: Path, level: int, prefix: str) -> None:
        if level >= depth:
            return
        try:
            children = [p for p in node.iterdir() if p.is_dir()]
        except OSError:
            return
        children.sort(key=lambda p: p.name.lower())
        if len(children) > max_children:
            shown = children[:max_children]
            extra = len(children) - max_children
        else:
            shown = children
            extra = 0

        for i, child in enumerate(shown):
            is_last = i == len(shown) - 1 and extra == 0
            connector = "`-- " if is_last else "|-- "
            lines.append(f"{prefix}{connector}{child.name}")
            child_prefix = prefix + ("    " if is_last else "|   ")
            walk(child, level + 1, child_prefix)

        if extra > 0:
            lines.append(f"{prefix}`-- ... (+{extra} more dirs)")

    walk(root, level=0, prefix="")
    return lines


def _scan_stats(root: Path) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[str, tuple[int, int]], int]:
    folder_count_by_level: dict[int, int] = defaultdict(int)
    mp4_count_by_level: dict[int, int] = defaultdict(int)
    mp4_size_by_level: dict[int, int] = defaultdict(int)
    mp4_by_parent: dict[str, tuple[int, int]] = {}
    total_mp4 = 0

    for directory in root.rglob("*"):
        if directory.is_dir():
            depth = len(directory.relative_to(root).parts)
            folder_count_by_level[depth] += 1

    parent_count: dict[str, int] = defaultdict(int)
    parent_size: dict[str, int] = defaultdict(int)
    for file_path in root.rglob("*.mp4"):
        if not file_path.is_file():
            continue
        try:
            size = int(file_path.stat().st_size)
        except OSError:
            continue
        depth = len(file_path.parent.relative_to(root).parts)
        mp4_count_by_level[depth] += 1
        mp4_size_by_level[depth] += size
        parent_key = file_path.parent.relative_to(root).as_posix()
        parent_count[parent_key] += 1
        parent_size[parent_key] += size
        total_mp4 += 1

    for key in set(parent_count.keys()).union(parent_size.keys()):
        mp4_by_parent[key] = (parent_count.get(key, 0), parent_size.get(key, 0))

    return folder_count_by_level, mp4_count_by_level, mp4_size_by_level, mp4_by_parent, total_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Ego4D unpacked directory structure")
    parser.add_argument("--root", default=r"D:\Ego4D_Dataset", help="Dataset root")
    parser.add_argument("--depth", type=int, default=3, help="Tree depth")
    parser.add_argument("--out", required=True, help="Output markdown path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"error=root_not_found path={root}")
        return 1

    tree_lines = _render_tree(root=root, depth=max(0, int(args.depth)))
    folder_count_by_level, mp4_count_by_level, mp4_size_by_level, mp4_by_parent, total_mp4 = _scan_stats(root=root)

    out_lines: list[str] = []
    out_lines.append("# Ego4D Directory Report")
    out_lines.append("")
    out_lines.append(f"- root: `{root}`")
    out_lines.append(f"- depth: {int(args.depth)}")
    out_lines.append(f"- total_mp4: {total_mp4}")
    out_lines.append("")
    out_lines.append("## Top-level Tree")
    out_lines.append("")
    out_lines.append("```text")
    out_lines.extend(tree_lines)
    out_lines.append("```")
    out_lines.append("")
    out_lines.append("## Folder Stats By Level")
    out_lines.append("")
    out_lines.append("| level | folders | mp4_count | total_size_gb |")
    out_lines.append("|---:|---:|---:|---:|")
    levels = sorted(set(folder_count_by_level.keys()) | set(mp4_count_by_level.keys()))
    for level in levels:
        folders = int(folder_count_by_level.get(level, 0))
        mp4_count = int(mp4_count_by_level.get(level, 0))
        size_gb = _human_gb(int(mp4_size_by_level.get(level, 0)))
        out_lines.append(f"| {level} | {folders} | {mp4_count} | {size_gb:.3f} |")
    out_lines.append("")
    out_lines.append("## MP4 Distribution By Parent Folder (Top 20)")
    out_lines.append("")
    out_lines.append("| folder | mp4_count | total_size_gb |")
    out_lines.append("|---|---:|---:|")
    ranked = sorted(mp4_by_parent.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)[:20]
    for folder, (count, size_bytes) in ranked:
        out_lines.append(f"| `{folder}` | {int(count)} | {_human_gb(int(size_bytes)):.3f} |")
    out_lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"root={root}")
    print(f"total_mp4={total_mp4}")
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
