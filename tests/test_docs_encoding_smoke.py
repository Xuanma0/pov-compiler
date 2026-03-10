from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_encoding_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_docs_encoding.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "status=ok" in proc.stdout
    assert "scanned_files=" in proc.stdout
