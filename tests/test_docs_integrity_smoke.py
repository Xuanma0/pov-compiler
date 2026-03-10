from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_integrity_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_docs_integrity.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "status=ok" in proc.stdout
    assert "paper_ready_status=" in proc.stdout
    assert "submission_pack_status=" in proc.stdout
