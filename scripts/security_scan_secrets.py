from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_sk", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("google_ai", re.compile(r"AIza[0-9A-Za-z\-_]{20,}")),
    ("aws_akia", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer", re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE)),
    ("api_key_literal", re.compile(r"(?i)api[_-]?key\s*[:=]\s*[\"'][A-Za-z0-9_\-]{12,}[\"']")),
    ("query_key_like", re.compile(r"(?i)[?&](?:key|api_key|token|secret)=[A-Za-z0-9_\-]{12,}")),
    (
        "provider_env_assignment",
        re.compile(
            r"(?i)(?:OPENAI|GEMINI|QWEN|DEEPSEEK|GLM|DASHSCOPE|ZHIPU)_API_KEY\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{12,}[\"']?"
        ),
    ),
    ("url_api_key_literal", re.compile(r"(?i)(?:key|api_key)=[A-Za-z0-9_\-]{12,}")),
    ("dashscope_like", re.compile(r"(?i)dashscope[^\\n]{0,40}(key|token)[^\\n]{0,20}[A-Za-z0-9_\-]{12,}")),
    ("zhipu_like", re.compile(r"(?i)zhipu[^\\n]{0,40}(key|token)[^\\n]{0,20}[A-Za-z0-9_\-]{12,}")),
]

TEXT_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".sh",
    ".ps1",
}


def _mask(value: str) -> str:
    token = str(value or "")
    if len(token) <= 6:
        return "***"
    return f"{token[:3]}***{token[-3:]}"


def list_tracked_files(repo_root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {proc.stderr.strip()}")
    out: list[Path] = []
    for line in proc.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        out.append((repo_root / rel).resolve())
    return out


def _is_text_candidate(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTS:
        return True
    return path.name.lower().startswith(".env")


def scan_file(path: Path) -> list[tuple[str, str]]:
    if not _is_text_candidate(path):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    findings: list[tuple[str, str]] = []
    for name, pattern in PATTERNS:
        for match in pattern.finditer(text):
            findings.append((name, _mask(match.group(0))))
            if len(findings) >= 8:
                return findings
    return findings


def scan_repository(repo_root: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for path in list_tracked_files(repo_root):
        file_findings = scan_file(path)
        if file_findings:
            findings.append(
                {
                    "file": str(path.relative_to(repo_root)),
                    "matches": [{"pattern": pat, "masked": masked} for pat, masked in file_findings[:5]],
                }
            )
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic secret scanner over git-tracked files")
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    try:
        findings = scan_repository(repo_root)
    except Exception as exc:
        print(f"error={exc}")
        return 2
    print(f"found_count={len(findings)}")
    sample = [f.get("file", "") for f in findings[:10]]
    print(f"found_files_sample={sample}")
    for item in findings[:10]:
        file_path = str(item.get("file", ""))
        matches = item.get("matches", [])
        if isinstance(matches, list):
            preview = ", ".join([f"{m.get('pattern')}:{m.get('masked')}" for m in matches[:3] if isinstance(m, dict)])
        else:
            preview = ""
        print(f"warn_file={file_path} matches={preview}")
    return 2 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
