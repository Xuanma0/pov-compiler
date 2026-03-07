from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.prompts.registry import PromptRegistry


def _copy_file_if_exists(src: Path, dst: Path, copied: list[str], missing: list[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    copied.append(str(dst))


def _copy_dir_if_exists(src: Path, dst: Path, copied: list[str], missing: list[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    copied.append(str(dst))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a submission-ready archive from paper-ready and suite artifacts.")
    parser.add_argument("--paper-ready-dir", required=True, help="Existing paper_ready output directory")
    parser.add_argument("--out-dir", default=None, help="Output directory (defaults to <paper-ready-dir>/submission_pack)")
    parser.add_argument("--suite-dir", default=None, help="Benchmark suite root containing manifest/ and ledger/")
    parser.add_argument("--significance-dir", default=None, help="Optional significance output directory")
    parser.add_argument("--result-health-dir", default=None, help="Optional result health output directory")
    parser.add_argument("--benchmark-freeze-dir", default=None, help="Optional freeze output directory")
    parser.add_argument("--paper-freeze-dir", default=None, help="Optional canonical paper freeze directory")
    parser.add_argument("--paper-map", default=None, help="Optional canonical paper-map YAML")
    parser.add_argument("--prompt-registry", default=None, help="Optional prompt registry YAML path")
    parser.add_argument("--prompt-lock", default=None, help="Optional prompt lock JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paper_ready_dir = Path(args.paper_ready_dir)
    if not paper_ready_dir.exists():
        raise FileNotFoundError(f"paper_ready_dir not found: {paper_ready_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else paper_ready_dir / "submission_pack"
    copied: list[str] = []
    missing: list[str] = []

    pack_paper_ready = out_dir / "paper_ready"
    pack_compare = out_dir / "compare"
    pack_manifest = out_dir / "manifest"
    pack_significance = out_dir / "significance"
    pack_result_health = out_dir / "result_health"
    pack_freeze = out_dir / "freeze"
    pack_paper_freeze = out_dir / "paper_freeze"
    pack_prompts = out_dir / "prompts"
    pack_provenance = out_dir / "provenance"
    for path in (
        pack_paper_ready,
        pack_compare,
        pack_manifest,
        pack_significance,
        pack_result_health,
        pack_freeze,
        pack_paper_freeze,
        pack_prompts,
        pack_provenance,
    ):
        path.mkdir(parents=True, exist_ok=True)

    _copy_dir_if_exists(paper_ready_dir / "tables", pack_paper_ready / "tables", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "figures", pack_paper_ready / "figures", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "canonical", pack_paper_ready / "canonical", copied, missing)
    _copy_file_if_exists(paper_ready_dir / "report.md", pack_paper_ready / "report.md", copied, missing)
    _copy_file_if_exists(paper_ready_dir / "snapshot.json", pack_paper_ready / "snapshot.json", copied, missing)

    suite_dir = Path(args.suite_dir) if args.suite_dir else None
    if suite_dir is not None:
        _copy_file_if_exists(suite_dir / "manifest" / "experiment_manifest.yaml", pack_manifest / "experiment_manifest.yaml", copied, missing)
        _copy_file_if_exists(suite_dir / "manifest" / "manifest_resolved.json", pack_manifest / "manifest_resolved.json", copied, missing)
        _copy_file_if_exists(suite_dir / "manifest" / "prompt_lock.json", pack_manifest / "prompt_lock.json", copied, missing)
        _copy_file_if_exists(suite_dir / "manifest" / "query_bank_lock.json", pack_manifest / "query_bank_lock.json", copied, missing)
        _copy_dir_if_exists(suite_dir / "manifest" / "query_banks", pack_manifest / "query_banks", copied, missing)
        _copy_file_if_exists(suite_dir / "ledger" / "results_long.csv", pack_provenance / "results_long.csv", copied, missing)
        _copy_file_if_exists(suite_dir / "ledger" / "runs.jsonl", pack_provenance / "runs.jsonl", copied, missing)
        _copy_file_if_exists(suite_dir / "compare" / "commands.sh", pack_provenance / "commands.sh", copied, missing)
        _copy_file_if_exists(suite_dir / "compare" / "compare_summary.json", pack_provenance / "compare_summary.json", copied, missing)
        _copy_file_if_exists(suite_dir / "compare" / "snapshot.json", pack_provenance / "compare_snapshot.json", copied, missing)
        _copy_dir_if_exists(suite_dir / "compare", pack_compare, copied, missing)

    significance_dir = Path(args.significance_dir) if args.significance_dir else None
    if significance_dir is not None:
        _copy_dir_if_exists(significance_dir / "tables", pack_significance / "tables", copied, missing)
        _copy_dir_if_exists(significance_dir / "figures", pack_significance / "figures", copied, missing)
        _copy_file_if_exists(significance_dir / "report.md", pack_significance / "report.md", copied, missing)
        _copy_file_if_exists(significance_dir / "snapshot.json", pack_significance / "snapshot.json", copied, missing)

    result_health_dir = Path(args.result_health_dir) if args.result_health_dir else None
    if result_health_dir is not None:
        _copy_dir_if_exists(result_health_dir / "tables", pack_result_health / "tables", copied, missing)
        _copy_dir_if_exists(result_health_dir / "figures", pack_result_health / "figures", copied, missing)
        _copy_file_if_exists(result_health_dir / "snapshot.json", pack_result_health / "snapshot.json", copied, missing)

    freeze_dir = Path(args.benchmark_freeze_dir) if args.benchmark_freeze_dir else None
    if freeze_dir is not None:
        _copy_file_if_exists(freeze_dir / "freeze_manifest.json", pack_freeze / "freeze_manifest.json", copied, missing)
        _copy_file_if_exists(freeze_dir / "artifacts_sha256.csv", pack_freeze / "artifacts_sha256.csv", copied, missing)

    paper_freeze_dir = Path(args.paper_freeze_dir) if args.paper_freeze_dir else None
    if paper_freeze_dir is not None:
        _copy_file_if_exists(paper_freeze_dir / "freeze_manifest.json", pack_paper_freeze / "freeze_manifest.json", copied, missing)
        _copy_file_if_exists(paper_freeze_dir / "paper_artifacts_sha256.csv", pack_paper_freeze / "paper_artifacts_sha256.csv", copied, missing)

    paper_map_path = Path(args.paper_map) if args.paper_map else None
    if paper_map_path is not None:
        _copy_file_if_exists(paper_map_path, pack_manifest / paper_map_path.name, copied, missing)

    prompt_lock_path = Path(args.prompt_lock) if args.prompt_lock else None
    if prompt_lock_path is not None:
        _copy_file_if_exists(prompt_lock_path, pack_manifest / "prompt_lock.json", copied, missing)

    prompt_registry_path = Path(args.prompt_registry) if args.prompt_registry else None
    if prompt_registry_path is not None:
        _copy_file_if_exists(prompt_registry_path, pack_prompts / prompt_registry_path.name, copied, missing)
        if prompt_registry_path.exists():
            registry = PromptRegistry.from_path(prompt_registry_path)
            for entry in registry.entries:
                source_path = entry.resolved_path(prompt_registry_path)
                try:
                    relative_path = source_path.relative_to(ROOT)
                    target_path = pack_prompts / relative_path
                except Exception:
                    target_path = pack_prompts / entry.task / source_path.name
                _copy_file_if_exists(source_path, target_path, copied, missing)

    canonical_map_payload = _load_json(paper_ready_dir / "canonical" / "paper_map_resolved.json")
    canonical_rows = canonical_map_payload.get("rows", []) if isinstance(canonical_map_payload.get("rows"), list) else []
    readme_lines = [
        "# Submission Pack",
        "",
        f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- paper_ready_dir: `{paper_ready_dir}`",
        f"- suite_dir: `{suite_dir}`",
        f"- significance_dir: `{significance_dir}`",
        f"- result_health_dir: `{result_health_dir}`",
        f"- benchmark_freeze_dir: `{freeze_dir}`",
        f"- paper_freeze_dir: `{paper_freeze_dir}`",
        f"- paper_map: `{paper_map_path}`",
        f"- prompt_registry: `{prompt_registry_path}`",
        f"- copied_items: `{len(copied)}`",
        f"- missing_inputs: `{len(missing)}`",
        "",
        "## Sections",
        "",
        "- `manifest/`: manifest copy, resolved manifest, prompt lock",
        "- `compare/`: frozen compare tables, figures, summary, and snapshot",
        "- `paper_ready/`: tables, figures, report, snapshot",
        "- `significance/`: significance tables, figures, report, snapshot",
        "- `result_health/`: result-health tables, figures, snapshot",
        "- `freeze/`: freeze manifest and artifact hashes",
        "- `paper_freeze/`: canonical paper-artifact freeze manifest and hashes",
        "- `prompts/`: registry and prompt source files",
        "- `provenance/`: ledger and compare-side provenance files",
    ]
    if canonical_rows:
        readme_lines.extend(
            [
                "",
                "## Paper Numbering",
                "",
                "- Use the canonical copies under `paper_ready/canonical/` when writing the paper.",
            ]
        )
        for row in canonical_rows:
            readme_lines.append(
                f"- `{row.get('canonical_id')}` -> `{Path('paper_ready') / row.get('canonical_relpath', '')}`"
            )
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper_ready_dir": str(paper_ready_dir),
        "suite_dir": str(suite_dir) if suite_dir is not None else None,
        "significance_dir": str(significance_dir) if significance_dir is not None else None,
        "result_health_dir": str(result_health_dir) if result_health_dir is not None else None,
        "benchmark_freeze_dir": str(freeze_dir) if freeze_dir is not None else None,
        "paper_freeze_dir": str(paper_freeze_dir) if paper_freeze_dir is not None else None,
        "paper_map": str(paper_map_path) if paper_map_path is not None else None,
        "prompt_registry": str(prompt_registry_path) if prompt_registry_path is not None else None,
        "prompt_lock": str(prompt_lock_path) if prompt_lock_path is not None else None,
        "copied": copied,
        "missing": missing,
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_submission_pack={out_dir}")
    print(f"saved_submission_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
