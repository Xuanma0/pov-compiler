from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".adoc"}
SUSPICIOUS_SNIPPETS = [
    "\ufffd",
    "Ã",
    "â€™",
    "â€œ",
    "â€\x9d",
    "â€“",
    "â€”",
    "ï»¿",
    "锟",
]


def _iter_targets(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in TEXT_EXTENSIONS else []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan repository docs for encoding and mojibake issues.")
    parser.add_argument("roots", nargs="*", help="Optional roots to scan. Defaults to README.md, CHANGELOG.md, and docs/.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [Path(p) for p in args.roots] if args.roots else [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "docs"]
    files: list[Path] = []
    for root in roots:
        target = root if root.is_absolute() else (ROOT / root)
        files.extend(_iter_targets(target))

    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)

    non_utf8: list[str] = []
    suspicious: list[str] = []
    control_chars: list[str] = []
    for path in deduped:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            non_utf8.append(str(path.relative_to(ROOT)))
            continue
        if any(snippet in text for snippet in SUSPICIOUS_SNIPPETS):
            suspicious.append(str(path.relative_to(ROOT)))
        if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
            control_chars.append(str(path.relative_to(ROOT)))

    status = "ok" if not non_utf8 and not suspicious and not control_chars else "issues_found"
    print(f"scanned_files={len(deduped)}")
    print(f"non_utf8_files={len(non_utf8)}")
    print(f"suspicious_files={len(suspicious)}")
    print(f"control_char_files={len(control_chars)}")
    print(f"status={status}")
    if non_utf8:
        print(f"non_utf8_list={non_utf8}")
    if suspicious:
        print(f"suspicious_list={suspicious}")
    if control_chars:
        print(f"control_char_list={control_chars}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
