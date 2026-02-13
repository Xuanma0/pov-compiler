from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.integrations.bye.entrypoints import EntryPointResolver


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("print('ok')\n", encoding="utf-8")


def test_entrypoint_resolver_search_finds_scripts(tmp_path: Path) -> None:
    bye_root = tmp_path / "bye_repo"
    _touch(bye_root / "a" / "lint_run_package.py")
    _touch(bye_root / "b" / "report_run.py")
    _touch(bye_root / "c" / "run_regression_suite.py")

    res = EntryPointResolver(bye_root=bye_root).resolve()
    assert res.found is True
    assert res.lint is not None and res.lint.name == "lint_run_package.py"
    assert res.report is not None and res.report.name == "report_run.py"
    assert res.regression is not None and res.regression.name == "run_regression_suite.py"
    assert res.methods.get("lint") == "search"
    assert res.methods.get("report") == "search"
    assert res.methods.get("regression") == "search"


def test_entrypoint_resolver_override_priority(tmp_path: Path) -> None:
    bye_root = tmp_path / "bye_repo"
    _touch(bye_root / "x" / "lint_run_package.py")
    _touch(bye_root / "x" / "report_run.py")
    _touch(bye_root / "x" / "run_regression_suite.py")
    custom = bye_root / "custom_lint.py"
    _touch(custom)

    res = EntryPointResolver(bye_root=bye_root, overrides={"lint": str(custom)}).resolve()
    assert res.lint is not None and res.lint.resolve() == custom.resolve()
    assert res.methods.get("lint") == "override"
    assert res.report is not None
    assert res.regression is not None

