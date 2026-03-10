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
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if dst_resolved == src_resolved:
        copied.append(str(dst))
        return
    if dst_resolved.is_relative_to(src_resolved):
        excluded_root: Path | None = None
        relative_parts = dst_resolved.relative_to(src_resolved).parts
        if relative_parts:
            excluded_root = src_resolved / relative_parts[0]
        for path in src.rglob("*"):
            resolved = path.resolve()
            if excluded_root is not None and (resolved == excluded_root or resolved.is_relative_to(excluded_root)):
                continue
            if resolved == dst_resolved or resolved.is_relative_to(dst_resolved):
                continue
            rel = path.relative_to(src)
            target = dst / rel
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        copied.append(str(dst))
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


def _display_optional(path_value: Path | str | None, *, missing: str = "not_provided") -> str:
    if path_value is None:
        return missing
    text = str(path_value).strip()
    return text if text else missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a submission-ready archive from paper-ready and suite artifacts.")
    parser.add_argument("--paper-ready-dir", default=None, help="Existing paper_ready output directory")
    parser.add_argument("--compare-dir", default=None, help="Optional compare root override for packed compare artifacts")
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default=None,
        help="Output directory (defaults to <paper-ready-dir>/submission_pack)",
    )
    parser.add_argument("--suite-dir", default=None, help="Benchmark suite root containing manifest/ and ledger/")
    parser.add_argument("--significance-dir", default=None, help="Optional significance output directory")
    parser.add_argument("--result-health-dir", default=None, help="Optional result health output directory")
    parser.add_argument("--admission-calibration-dir", default=None, help="Optional admission calibration output directory")
    parser.add_argument("--result-diagnosis-dir", default=None, help="Optional result diagnosis output directory")
    parser.add_argument("--delta-audit-dir", default=None, help="Optional delta audit output directory")
    parser.add_argument("--signal-uplift-dir", default=None, help="Optional signal uplift output directory")
    parser.add_argument("--query-bank-compare-dir", default=None, help="Optional query-bank compare output directory")
    parser.add_argument(
        "--query-bank-promotion-decision-dir",
        default=None,
        help="Optional query-bank promotion decision output directory",
    )
    parser.add_argument(
        "--persistent-memory-main-compare-dir",
        default=None,
        help="Optional persistent-memory main compare output directory",
    )
    parser.add_argument(
        "--persistent-memory-main-decision-dir",
        default=None,
        help="Optional persistent-memory main decision output directory",
    )
    parser.add_argument(
        "--mainline-admission-cleanup-dir",
        default=None,
        help="Optional mainline admission cleanup output directory",
    )
    parser.add_argument(
        "--sample-contract-dir",
        default=None,
        help="Optional sample contract output directory",
    )
    parser.add_argument(
        "--harder-sample-contract-dir",
        default=None,
        help="Optional harder sample contract output directory",
    )
    parser.add_argument(
        "--mainline-admission-closure-dir",
        default=None,
        help="Optional mainline admission closure output directory",
    )
    parser.add_argument("--query-strength-audit-dir", default=None, help="Optional query-strength audit output directory")
    parser.add_argument("--provider-telemetry-dir", default=None, help="Optional provider telemetry output directory")
    parser.add_argument("--provider-reachability-dir", default=None, help="Optional provider reachability proof directory")
    parser.add_argument("--provider-normalization-dir", default=None, help="Optional normalized provider telemetry output directory")
    parser.add_argument("--query-promotion-pack-dir", default=None, help="Optional query promotion pack output directory")
    parser.add_argument("--repeatability-audit-dir", default=None, help="Optional repeatability audit output directory")
    parser.add_argument("--sample-size-recommendation-dir", default=None, help="Optional sample-size recommendation output directory")
    parser.add_argument("--query-uplift-candidates-dir", default=None, help="Optional query uplift candidates output directory")
    parser.add_argument("--golden-real-sample-dir", default=None, help="Optional golden real sample output directory")
    parser.add_argument("--benchmark-freeze-dir", default=None, help="Optional freeze output directory")
    parser.add_argument("--paper-freeze-dir", default=None, help="Optional canonical paper freeze directory")
    parser.add_argument("--paper-map", default=None, help="Optional canonical paper-map YAML")
    parser.add_argument("--prompt-registry", default=None, help="Optional prompt registry YAML path")
    parser.add_argument("--prompt-lock", default=None, help="Optional prompt lock JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paper_ready_dir: Path | None = Path(args.paper_ready_dir) if args.paper_ready_dir else None
    if paper_ready_dir is None and args.out_dir:
        inferred = Path(args.out_dir).resolve().parent / "paper_ready"
        if inferred.exists():
            paper_ready_dir = inferred
    if paper_ready_dir is None:
        raise FileNotFoundError("paper_ready_dir not provided and could not be inferred from --out-dir")
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
    pack_admission = out_dir / "admission_control"
    pack_admission_calibration = out_dir / "admission_calibration"
    pack_result_diagnosis = out_dir / "result_diagnosis"
    pack_delta_audit = out_dir / "delta_audit"
    pack_signal_uplift = out_dir / "signal_uplift"
    pack_query_bank_compare = out_dir / "query_bank_compare"
    pack_query_bank_promotion_decision = out_dir / "query_bank_promotion_decision"
    pack_persistent_memory_main_compare = out_dir / "persistent_memory_main_compare"
    pack_persistent_memory_main_decision = out_dir / "persistent_memory_main_decision"
    pack_mainline_admission_cleanup = out_dir / "mainline_admission_cleanup"
    pack_sample_contract = out_dir / "sample_contract"
    pack_harder_sample_contract = out_dir / "harder_sample_contract"
    pack_mainline_admission_closure = out_dir / "mainline_admission_closure"
    pack_query_strength_audit = out_dir / "query_strength_audit"
    pack_provider_telemetry = out_dir / "provider_telemetry"
    pack_provider_reachability = out_dir / "provider_reachability"
    pack_provider_normalization = out_dir / "provider_normalization"
    pack_query_promotion_pack = out_dir / "query_promotion_pack"
    pack_repeatability_audit = out_dir / "repeatability_audit"
    pack_sample_size_recommendation = out_dir / "sample_size_recommendation"
    pack_query_uplift_candidates = out_dir / "query_uplift_candidates"
    pack_golden_real_sample = out_dir / "golden_real_sample"
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
        pack_admission,
        pack_admission_calibration,
        pack_result_diagnosis,
        pack_delta_audit,
        pack_signal_uplift,
        pack_query_bank_compare,
        pack_query_bank_promotion_decision,
        pack_persistent_memory_main_compare,
        pack_persistent_memory_main_decision,
        pack_mainline_admission_cleanup,
        pack_sample_contract,
        pack_harder_sample_contract,
        pack_mainline_admission_closure,
        pack_query_strength_audit,
        pack_provider_telemetry,
        pack_provider_reachability,
        pack_provider_normalization,
        pack_query_promotion_pack,
        pack_repeatability_audit,
        pack_sample_size_recommendation,
        pack_query_uplift_candidates,
        pack_golden_real_sample,
        pack_freeze,
        pack_paper_freeze,
        pack_prompts,
        pack_provenance,
    ):
        path.mkdir(parents=True, exist_ok=True)

    _copy_dir_if_exists(paper_ready_dir / "tables", pack_paper_ready / "tables", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "figures", pack_paper_ready / "figures", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "canonical", pack_paper_ready / "canonical", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "admission_control", pack_paper_ready / "admission_control", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "admission_calibration", pack_paper_ready / "admission_calibration", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "result_diagnosis", pack_paper_ready / "result_diagnosis", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "delta_audit", pack_paper_ready / "delta_audit", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "signal_uplift", pack_paper_ready / "signal_uplift", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "query_bank_compare", pack_paper_ready / "query_bank_compare", copied, missing)
    _copy_dir_if_exists(
        paper_ready_dir / "query_bank_promotion_decision",
        pack_paper_ready / "query_bank_promotion_decision",
        copied,
        missing,
    )
    _copy_dir_if_exists(
        paper_ready_dir / "persistent_memory_main_compare",
        pack_paper_ready / "persistent_memory_main_compare",
        copied,
        missing,
    )
    _copy_dir_if_exists(
        paper_ready_dir / "persistent_memory_main_decision",
        pack_paper_ready / "persistent_memory_main_decision",
        copied,
        missing,
    )
    _copy_dir_if_exists(
        paper_ready_dir / "mainline_admission_cleanup",
        pack_paper_ready / "mainline_admission_cleanup",
        copied,
        missing,
    )
    _copy_dir_if_exists(
        paper_ready_dir / "sample_contract",
        pack_paper_ready / "sample_contract",
        copied,
        missing,
    )
    _copy_dir_if_exists(
        paper_ready_dir / "harder_sample_contract",
        pack_paper_ready / "harder_sample_contract",
        copied,
        missing,
    )
    _copy_dir_if_exists(
        paper_ready_dir / "mainline_admission_closure",
        pack_paper_ready / "mainline_admission_closure",
        copied,
        missing,
    )
    _copy_dir_if_exists(paper_ready_dir / "query_strength_audit", pack_paper_ready / "query_strength_audit", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "provider_telemetry", pack_paper_ready / "provider_telemetry", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "provider_reachability", pack_paper_ready / "provider_reachability", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "provider_normalization", pack_paper_ready / "provider_normalization", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "query_promotion_pack", pack_paper_ready / "query_promotion_pack", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "repeatability_audit", pack_paper_ready / "repeatability_audit", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "sample_size_recommendation", pack_paper_ready / "sample_size_recommendation", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "query_uplift_candidates", pack_paper_ready / "query_uplift_candidates", copied, missing)
    _copy_dir_if_exists(paper_ready_dir / "golden_real_sample", pack_paper_ready / "golden_real_sample", copied, missing)
    _copy_file_if_exists(paper_ready_dir / "report.md", pack_paper_ready / "report.md", copied, missing)
    _copy_file_if_exists(paper_ready_dir / "snapshot.json", pack_paper_ready / "snapshot.json", copied, missing)

    suite_dir = Path(args.suite_dir) if args.suite_dir else None
    compare_dir = Path(args.compare_dir) if args.compare_dir else None
    if suite_dir is not None:
        _copy_file_if_exists(suite_dir / "manifest" / "experiment_manifest.yaml", pack_manifest / "experiment_manifest.yaml", copied, missing)
        _copy_file_if_exists(suite_dir / "manifest" / "manifest_resolved.json", pack_manifest / "manifest_resolved.json", copied, missing)
        _copy_file_if_exists(suite_dir / "manifest" / "prompt_lock.json", pack_manifest / "prompt_lock.json", copied, missing)
        _copy_file_if_exists(suite_dir / "manifest" / "query_bank_lock.json", pack_manifest / "query_bank_lock.json", copied, missing)
        _copy_dir_if_exists(suite_dir / "manifest" / "query_banks", pack_manifest / "query_banks", copied, missing)
        _copy_file_if_exists(suite_dir / "ledger" / "results_long.csv", pack_provenance / "results_long.csv", copied, missing)
        _copy_file_if_exists(suite_dir / "ledger" / "runs.jsonl", pack_provenance / "runs.jsonl", copied, missing)
        compare_source = compare_dir if compare_dir is not None else suite_dir / "compare"
        _copy_file_if_exists(compare_source / "commands.sh", pack_provenance / "commands.sh", copied, missing)
        _copy_file_if_exists(compare_source / "compare_summary.json", pack_provenance / "compare_summary.json", copied, missing)
        _copy_file_if_exists(compare_source / "snapshot.json", pack_provenance / "compare_snapshot.json", copied, missing)
        _copy_dir_if_exists(compare_source, pack_compare, copied, missing)

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

    admission_dir = suite_dir / "admission_control" if suite_dir is not None else None
    if admission_dir is not None and admission_dir.exists():
        _copy_file_if_exists(admission_dir / "report.md", pack_admission / "report.md", copied, missing)
        _copy_file_if_exists(admission_dir / "snapshot.json", pack_admission / "snapshot.json", copied, missing)

    admission_calibration_dir = Path(args.admission_calibration_dir) if args.admission_calibration_dir else None
    if admission_calibration_dir is not None:
        _copy_dir_if_exists(admission_calibration_dir / "tables", pack_admission_calibration / "tables", copied, missing)
        _copy_file_if_exists(admission_calibration_dir / "report.md", pack_admission_calibration / "report.md", copied, missing)
        _copy_file_if_exists(admission_calibration_dir / "snapshot.json", pack_admission_calibration / "snapshot.json", copied, missing)

    result_diagnosis_dir = Path(args.result_diagnosis_dir) if args.result_diagnosis_dir else None
    if result_diagnosis_dir is not None:
        _copy_dir_if_exists(result_diagnosis_dir / "tables", pack_result_diagnosis / "tables", copied, missing)
        _copy_dir_if_exists(result_diagnosis_dir / "figures", pack_result_diagnosis / "figures", copied, missing)
        _copy_file_if_exists(result_diagnosis_dir / "report.md", pack_result_diagnosis / "report.md", copied, missing)
        _copy_file_if_exists(result_diagnosis_dir / "snapshot.json", pack_result_diagnosis / "snapshot.json", copied, missing)

    delta_audit_dir = Path(args.delta_audit_dir) if args.delta_audit_dir else None
    if delta_audit_dir is not None:
        _copy_dir_if_exists(delta_audit_dir / "tables", pack_delta_audit / "tables", copied, missing)
        _copy_dir_if_exists(delta_audit_dir / "figures", pack_delta_audit / "figures", copied, missing)
        _copy_file_if_exists(delta_audit_dir / "report.md", pack_delta_audit / "report.md", copied, missing)
        _copy_file_if_exists(delta_audit_dir / "snapshot.json", pack_delta_audit / "snapshot.json", copied, missing)

    signal_uplift_dir = Path(args.signal_uplift_dir) if args.signal_uplift_dir else None
    if signal_uplift_dir is not None:
        _copy_dir_if_exists(signal_uplift_dir / "tables", pack_signal_uplift / "tables", copied, missing)
        _copy_dir_if_exists(signal_uplift_dir / "figures", pack_signal_uplift / "figures", copied, missing)
        _copy_file_if_exists(signal_uplift_dir / "report.md", pack_signal_uplift / "report.md", copied, missing)
        _copy_file_if_exists(signal_uplift_dir / "snapshot.json", pack_signal_uplift / "snapshot.json", copied, missing)

    query_bank_compare_dir = Path(args.query_bank_compare_dir) if args.query_bank_compare_dir else None
    if query_bank_compare_dir is not None:
        _copy_dir_if_exists(query_bank_compare_dir / "tables", pack_query_bank_compare / "tables", copied, missing)
        _copy_dir_if_exists(query_bank_compare_dir / "figures", pack_query_bank_compare / "figures", copied, missing)
        _copy_file_if_exists(query_bank_compare_dir / "compare_summary.json", pack_query_bank_compare / "compare_summary.json", copied, missing)
        _copy_file_if_exists(query_bank_compare_dir / "snapshot.json", pack_query_bank_compare / "snapshot.json", copied, missing)
        _copy_file_if_exists(query_bank_compare_dir / "commands.sh", pack_query_bank_compare / "commands.sh", copied, missing)
        _copy_file_if_exists(query_bank_compare_dir / "README.md", pack_query_bank_compare / "README.md", copied, missing)

    query_bank_promotion_decision_dir = (
        Path(args.query_bank_promotion_decision_dir) if args.query_bank_promotion_decision_dir else None
    )
    if query_bank_promotion_decision_dir is not None:
        _copy_dir_if_exists(
            query_bank_promotion_decision_dir / "tables",
            pack_query_bank_promotion_decision / "tables",
            copied,
            missing,
        )
        _copy_file_if_exists(
            query_bank_promotion_decision_dir / "report.md",
            pack_query_bank_promotion_decision / "report.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            query_bank_promotion_decision_dir / "snapshot.json",
            pack_query_bank_promotion_decision / "snapshot.json",
            copied,
            missing,
        )

    persistent_memory_main_compare_dir = (
        Path(args.persistent_memory_main_compare_dir) if args.persistent_memory_main_compare_dir else None
    )
    if persistent_memory_main_compare_dir is not None:
        _copy_dir_if_exists(
            persistent_memory_main_compare_dir / "tables",
            pack_persistent_memory_main_compare / "tables",
            copied,
            missing,
        )
        _copy_dir_if_exists(
            persistent_memory_main_compare_dir / "figures",
            pack_persistent_memory_main_compare / "figures",
            copied,
            missing,
        )
        _copy_file_if_exists(
            persistent_memory_main_compare_dir / "compare_summary.json",
            pack_persistent_memory_main_compare / "compare_summary.json",
            copied,
            missing,
        )
        _copy_file_if_exists(
            persistent_memory_main_compare_dir / "snapshot.json",
            pack_persistent_memory_main_compare / "snapshot.json",
            copied,
            missing,
        )
        _copy_file_if_exists(
            persistent_memory_main_compare_dir / "commands.sh",
            pack_persistent_memory_main_compare / "commands.sh",
            copied,
            missing,
        )
        _copy_file_if_exists(
            persistent_memory_main_compare_dir / "README.md",
            pack_persistent_memory_main_compare / "README.md",
            copied,
            missing,
        )

    persistent_memory_main_decision_dir = (
        Path(args.persistent_memory_main_decision_dir) if args.persistent_memory_main_decision_dir else None
    )
    if persistent_memory_main_decision_dir is not None:
        _copy_dir_if_exists(
            persistent_memory_main_decision_dir / "tables",
            pack_persistent_memory_main_decision / "tables",
            copied,
            missing,
        )
        _copy_file_if_exists(
            persistent_memory_main_decision_dir / "report.md",
            pack_persistent_memory_main_decision / "report.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            persistent_memory_main_decision_dir / "snapshot.json",
            pack_persistent_memory_main_decision / "snapshot.json",
            copied,
            missing,
        )

    mainline_admission_cleanup_dir = (
        Path(args.mainline_admission_cleanup_dir) if args.mainline_admission_cleanup_dir else None
    )
    if mainline_admission_cleanup_dir is not None:
        _copy_dir_if_exists(
            mainline_admission_cleanup_dir / "tables",
            pack_mainline_admission_cleanup / "tables",
            copied,
            missing,
        )
        _copy_file_if_exists(
            mainline_admission_cleanup_dir / "report.md",
            pack_mainline_admission_cleanup / "report.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            mainline_admission_cleanup_dir / "snapshot.json",
            pack_mainline_admission_cleanup / "snapshot.json",
            copied,
            missing,
        )
        _copy_dir_if_exists(
            mainline_admission_cleanup_dir,
            pack_paper_ready / "mainline_admission_cleanup",
            copied,
            missing,
        )

    sample_contract_dir = Path(args.sample_contract_dir) if args.sample_contract_dir else None
    if sample_contract_dir is not None:
        _copy_dir_if_exists(
            sample_contract_dir / "tables",
            pack_sample_contract / "tables",
            copied,
            missing,
        )
        _copy_file_if_exists(
            sample_contract_dir / "report.md",
            pack_sample_contract / "report.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            sample_contract_dir / "snapshot.json",
            pack_sample_contract / "snapshot.json",
            copied,
            missing,
        )
        _copy_dir_if_exists(
            sample_contract_dir,
            pack_paper_ready / "sample_contract",
            copied,
            missing,
        )

    harder_sample_contract_dir = (
        Path(args.harder_sample_contract_dir) if args.harder_sample_contract_dir else None
    )
    if harder_sample_contract_dir is not None:
        _copy_file_if_exists(
            harder_sample_contract_dir / "tables" / "table_harder_sample_contract_summary.csv",
            pack_harder_sample_contract / "tables" / "table_harder_sample_contract_summary.csv",
            copied,
            missing,
        )
        _copy_file_if_exists(
            harder_sample_contract_dir / "tables" / "table_harder_sample_contract_summary.md",
            pack_harder_sample_contract / "tables" / "table_harder_sample_contract_summary.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            harder_sample_contract_dir / "report.md",
            pack_harder_sample_contract / "report.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            harder_sample_contract_dir / "snapshot.json",
            pack_harder_sample_contract / "snapshot.json",
            copied,
            missing,
        )
        _copy_dir_if_exists(
            harder_sample_contract_dir,
            pack_paper_ready / "harder_sample_contract",
            copied,
            missing,
        )

    mainline_admission_closure_dir = (
        Path(args.mainline_admission_closure_dir) if args.mainline_admission_closure_dir else None
    )
    if mainline_admission_closure_dir is not None:
        _copy_file_if_exists(
            mainline_admission_closure_dir / "tables" / "table_mainline_admission_closure.csv",
            pack_mainline_admission_closure / "tables" / "table_mainline_admission_closure.csv",
            copied,
            missing,
        )
        _copy_file_if_exists(
            mainline_admission_closure_dir / "tables" / "table_mainline_admission_closure.md",
            pack_mainline_admission_closure / "tables" / "table_mainline_admission_closure.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            mainline_admission_closure_dir / "report.md",
            pack_mainline_admission_closure / "report.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            mainline_admission_closure_dir / "snapshot.json",
            pack_mainline_admission_closure / "snapshot.json",
            copied,
            missing,
        )
        _copy_dir_if_exists(
            mainline_admission_closure_dir,
            pack_paper_ready / "mainline_admission_closure",
            copied,
            missing,
        )

    query_strength_audit_dir = Path(args.query_strength_audit_dir) if args.query_strength_audit_dir else None
    if query_strength_audit_dir is not None:
        _copy_dir_if_exists(query_strength_audit_dir / "tables", pack_query_strength_audit / "tables", copied, missing)
        _copy_dir_if_exists(query_strength_audit_dir / "figures", pack_query_strength_audit / "figures", copied, missing)
        _copy_file_if_exists(query_strength_audit_dir / "report.md", pack_query_strength_audit / "report.md", copied, missing)
        _copy_file_if_exists(query_strength_audit_dir / "snapshot.json", pack_query_strength_audit / "snapshot.json", copied, missing)

    provider_telemetry_dir = Path(args.provider_telemetry_dir) if args.provider_telemetry_dir else None
    if provider_telemetry_dir is not None:
        _copy_file_if_exists(provider_telemetry_dir / "by_variant.csv", pack_provider_telemetry / "by_variant.csv", copied, missing)
        _copy_file_if_exists(provider_telemetry_dir / "summary.json", pack_provider_telemetry / "summary.json", copied, missing)

    provider_reachability_dir = Path(args.provider_reachability_dir) if args.provider_reachability_dir else None
    if provider_reachability_dir is not None:
        _copy_file_if_exists(provider_reachability_dir / "summary.json", pack_provider_reachability / "summary.json", copied, missing)
        _copy_file_if_exists(provider_reachability_dir / "report.md", pack_provider_reachability / "report.md", copied, missing)
        _copy_file_if_exists(provider_reachability_dir / "snapshot.json", pack_provider_reachability / "snapshot.json", copied, missing)

    provider_normalization_dir = Path(args.provider_normalization_dir) if args.provider_normalization_dir else None
    if provider_normalization_dir is not None:
        _copy_dir_if_exists(provider_normalization_dir / "tables", pack_provider_normalization / "tables", copied, missing)
        _copy_file_if_exists(provider_normalization_dir / "report.md", pack_provider_normalization / "report.md", copied, missing)
        _copy_file_if_exists(provider_normalization_dir / "snapshot.json", pack_provider_normalization / "snapshot.json", copied, missing)

    query_promotion_pack_dir = Path(args.query_promotion_pack_dir) if args.query_promotion_pack_dir else None
    if query_promotion_pack_dir is not None:
        _copy_dir_if_exists(query_promotion_pack_dir / "query_pack", pack_query_promotion_pack / "query_pack", copied, missing)
        _copy_file_if_exists(query_promotion_pack_dir / "report.md", pack_query_promotion_pack / "report.md", copied, missing)

    repeatability_audit_dir = Path(args.repeatability_audit_dir) if args.repeatability_audit_dir else None
    if repeatability_audit_dir is not None:
        _copy_dir_if_exists(repeatability_audit_dir / "tables", pack_repeatability_audit / "tables", copied, missing)
        _copy_dir_if_exists(repeatability_audit_dir / "figures", pack_repeatability_audit / "figures", copied, missing)
        _copy_file_if_exists(repeatability_audit_dir / "report.md", pack_repeatability_audit / "report.md", copied, missing)
        _copy_file_if_exists(repeatability_audit_dir / "snapshot.json", pack_repeatability_audit / "snapshot.json", copied, missing)

    sample_size_recommendation_dir = (
        Path(args.sample_size_recommendation_dir) if args.sample_size_recommendation_dir else None
    )
    if sample_size_recommendation_dir is not None:
        _copy_dir_if_exists(
            sample_size_recommendation_dir / "tables",
            pack_sample_size_recommendation / "tables",
            copied,
            missing,
        )
        _copy_file_if_exists(
            sample_size_recommendation_dir / "report.md",
            pack_sample_size_recommendation / "report.md",
            copied,
            missing,
        )
        _copy_file_if_exists(
            sample_size_recommendation_dir / "snapshot.json",
            pack_sample_size_recommendation / "snapshot.json",
            copied,
            missing,
        )

    query_uplift_candidates_dir = Path(args.query_uplift_candidates_dir) if args.query_uplift_candidates_dir else None
    if query_uplift_candidates_dir is not None:
        _copy_dir_if_exists(
            query_uplift_candidates_dir / "query_uplift_candidates",
            pack_query_uplift_candidates / "query_uplift_candidates",
            copied,
            missing,
        )
        _copy_file_if_exists(
            query_uplift_candidates_dir / "report.md",
            pack_query_uplift_candidates / "report.md",
            copied,
            missing,
        )

    golden_real_sample_dir = Path(args.golden_real_sample_dir) if args.golden_real_sample_dir else None
    if golden_real_sample_dir is not None:
        _copy_file_if_exists(golden_real_sample_dir / "sample_manifest.json", pack_golden_real_sample / "sample_manifest.json", copied, missing)
        _copy_file_if_exists(golden_real_sample_dir / "query_set.yaml", pack_golden_real_sample / "query_set.yaml", copied, missing)
        _copy_file_if_exists(golden_real_sample_dir / "expected_outputs.json", pack_golden_real_sample / "expected_outputs.json", copied, missing)
        _copy_file_if_exists(golden_real_sample_dir / "report.md", pack_golden_real_sample / "report.md", copied, missing)
        _copy_file_if_exists(golden_real_sample_dir / "snapshot.json", pack_golden_real_sample / "snapshot.json", copied, missing)

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
        f"- suite_dir: `{_display_optional(suite_dir)}`",
        f"- compare_dir: `{_display_optional(compare_dir)}`",
        f"- significance_dir: `{_display_optional(significance_dir)}`",
        f"- result_health_dir: `{_display_optional(result_health_dir)}`",
        f"- admission_dir: `{_display_optional(admission_dir)}`",
        f"- admission_calibration_dir: `{_display_optional(admission_calibration_dir)}`",
        f"- result_diagnosis_dir: `{_display_optional(result_diagnosis_dir)}`",
        f"- delta_audit_dir: `{_display_optional(delta_audit_dir)}`",
        f"- query_strength_audit_dir: `{_display_optional(query_strength_audit_dir)}`",
        f"- persistent_memory_main_compare_dir: `{_display_optional(persistent_memory_main_compare_dir)}`",
        f"- persistent_memory_main_decision_dir: `{_display_optional(persistent_memory_main_decision_dir)}`",
        f"- mainline_admission_cleanup_dir: `{_display_optional(mainline_admission_cleanup_dir)}`",
        f"- sample_contract_dir: `{_display_optional(sample_contract_dir)}`",
        f"- harder_sample_contract_dir: `{_display_optional(harder_sample_contract_dir)}`",
        f"- mainline_admission_closure_dir: `{_display_optional(mainline_admission_closure_dir)}`",
        f"- provider_telemetry_dir: `{_display_optional(provider_telemetry_dir)}`",
        f"- provider_reachability_dir: `{_display_optional(provider_reachability_dir)}`",
        f"- provider_normalization_dir: `{_display_optional(provider_normalization_dir)}`",
        f"- query_promotion_pack_dir: `{_display_optional(query_promotion_pack_dir)}`",
        f"- repeatability_audit_dir: `{_display_optional(repeatability_audit_dir)}`",
        f"- sample_size_recommendation_dir: `{_display_optional(sample_size_recommendation_dir)}`",
        f"- query_uplift_candidates_dir: `{_display_optional(query_uplift_candidates_dir)}`",
        f"- golden_real_sample_dir: `{_display_optional(golden_real_sample_dir)}`",
        f"- benchmark_freeze_dir: `{_display_optional(freeze_dir)}`",
        f"- paper_freeze_dir: `{_display_optional(paper_freeze_dir)}`",
        f"- paper_map: `{_display_optional(paper_map_path)}`",
        f"- prompt_registry: `{_display_optional(prompt_registry_path)}`",
        f"- prompt_lock: `{_display_optional(prompt_lock_path)}`",
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
        "- `admission_control/`: admission decision snapshot and report",
        "- `admission_calibration/`: suggested admission thresholds for the next real pilot",
        "- `result_diagnosis/`: diagnosis tables, figures, report, snapshot",
        "- `delta_audit/`: per-budget delta audit and next-step action suggestions",
        "- `signal_uplift/`: before/after perception signal gain summary for local YOLO26n pilots",
        "- `query_bank_compare/`: aligned v1-vs-v2 main-result compare tables, figures, and provenance",
        "- `query_bank_promotion_decision/`: formal recommendation on whether v2 should replace v1",
        "- `persistent_memory_main_compare/`: aligned large-sample baseline-vs-persistent compare tables, figures, and provenance",
        "- `persistent_memory_main_decision/`: formal recommendation on whether persistent memory should enter mainline",
        "- `mainline_admission_cleanup/`: explanation of why promotion can coexist with partial admission and what still needs cleanup",
        "- `sample_contract/`: explicit sample/coverage/freeze evidence for large-sample wording and mainline claims",
        "- `harder_sample_contract/`: stricter sample/coverage/freeze evidence used for v1.60 admission closure",
        "- `mainline_admission_closure/`: final closure verdict on whether mainline admission is now clean enough",
        "- `query_strength_audit/`: query-group strength audit for main-paper inclusion decisions",
        "- `provider_telemetry/`: provider/cost/latency/parse-fail sidecar summary",
        "- `provider_reachability/`: live-call reachability proof for the chosen provider/server",
        "- `provider_normalization/`: normalized provider telemetry with explicit missing-field semantics",
        "- `query_promotion_pack/`: candidate packs for promoting or keeping queries outside the main bank",
        "- `repeatability_audit/`: repeated-run variance audit for provider/sample stability",
        "- `sample_size_recommendation/`: next-round minimum sample-size guidance",
        "- `query_uplift_candidates/`: queries worth strengthening before any promotion decision",
        "- `golden_real_sample/`: reusable live-smoke sample contract with expected non-empty outputs",
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
    readme_lines.extend(
        [
            "",
            "## Repository Docs Entry",
            "",
            "- English docs entry: `docs/README.en.md`",
            "- Chinese docs entry: `docs/README.zh-CN.md`",
            "- Current mainline status: `docs/en/current_mainline_status.md` / `docs/zh-CN/current_mainline_status.md`",
            "- Experiment history: `docs/en/experiment_history.md` / `docs/zh-CN/experiment_history.md`",
            "",
        "## Reading Order",
        "",
        "- Read `docs/en/current_mainline_status.md` or `docs/zh-CN/current_mainline_status.md` first.",
        "- Then read `docs/en/experiment_history.md` or `docs/zh-CN/experiment_history.md`.",
        "- Then read `persistent_memory_main_compare/`.",
        "- Then read `persistent_memory_main_decision/`.",
        "- Then read `mainline_admission_cleanup/`.",
        "- Then read `harder_sample_contract/`.",
        "- Then read `mainline_admission_closure/`.",
        "- Then read `provider_reachability/`.",
        "- Then read `repeatability_audit/`.",
        "- Then read `sample_size_recommendation/`.",
        "- Then read `admission_control/`.",
            "- If admission is blocked, read `admission_calibration/` next.",
            "- Then read `result_health/` and `result_diagnosis/`.",
            "- Then read `delta_audit/`.",
            "- Then read `signal_uplift/` to decide whether weak results are really perception-signal limited.",
                "- Then read `query_bank_compare/` before deciding whether v2 is actually better than v1.",
                "- Then read `query_bank_promotion_decision/` before widening the next real main run on v2.",
            "- Then read `provider_normalization/` before interpreting cross-provider cost or usage.",
            "- If main figures still look weak, read `query_strength_audit/` before interpreting them.",
            "- Then read `query_uplift_candidates/` before deciding whether weak queries deserve more signal or sample.",
            "- Then read `query_promotion_pack/` before changing the frozen main query bank.",
            "- Then read `golden_real_sample/` to confirm the reusable live-smoke contract.",
            "- Only then cite the canonical main tables and figures under `paper_ready/canonical/`.",
        ]
    )
    if admission_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Admission First",
                "",
                "- Read `admission_control/` first.",
                "- Read `admission_control/snapshot.json` first to decide whether this pilot is worth expanding.",
            ]
        )
    if result_diagnosis_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Diagnosis First",
                "",
                "- After admission, read `result_diagnosis/report.md` to understand no-data, significance, and provider-noise explanations.",
            ]
        )
    if delta_audit_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Delta Audit",
                "",
                "- After diagnosis, read `delta_audit/report.md` to decide whether to tune signal coverage, query bank, planner, repo policy, or stop for no-effect.",
            ]
        )
    if signal_uplift_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Signal Uplift",
                "",
                "- Read `signal_uplift/report.md` before deciding whether the current weak result is still perception-limited.",
                "- If `should_try_sam3_next=true`, treat SAM3 as the next segmentation candidate instead of widening the current YOLO-only pilot immediately.",
            ]
        )
    if admission_calibration_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Calibration First",
                "",
                "- If admission is partial or blocked, read `admission_calibration/report.md` before loosening thresholds.",
            ]
        )
    if query_strength_audit_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Query Strength Audit",
                "",
                "- If main figures look weak, read `query_strength_audit/report.md` before blaming the algorithm.",
            ]
        )
    if provider_telemetry_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Provider Noise",
                "",
                "- If main results look noisy, read `provider_telemetry/summary.json` before trusting weak deltas.",
            ]
        )
    if provider_reachability_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Provider Reachability",
                "",
                "- Read `provider_reachability/report.md` first to confirm that a real or local OpenAI-compatible live call actually happened.",
            ]
        )
    if provider_normalization_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Normalized Telemetry",
                "",
                "- Read `provider_normalization/snapshot.json` before comparing provider cost, usage, or latency across backends.",
            ]
        )
    if repeatability_audit_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Repeatability Audit",
                "",
                "- Read `repeatability_audit/report.md` immediately after provider reachability to tell provider drift from sample drift.",
            ]
        )
    if sample_size_recommendation_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Sample Size Recommendation",
                "",
                "- Read `sample_size_recommendation/report.md` before widening the next real main pilot.",
            ]
        )
    if query_uplift_candidates_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Query Uplift Candidates",
                "",
                "- Read `query_uplift_candidates/report.md` before deciding whether a weak query needs more signal/sample or should stay out of the main bank.",
            ]
        )
    if query_promotion_pack_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Query Promotion",
                "",
                "- Only read `query_promotion_pack/report.md` after reviewing repeatability, sample size, and uplift evidence.",
            ]
        )
    if golden_real_sample_dir is not None:
        readme_lines.extend(
            [
                "",
                "## Golden Real Sample",
                "",
                "- Read `golden_real_sample/report.md` after promotion decisions to find the reusable non-empty live sample contract.",
            ]
        )
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper_ready_dir": str(paper_ready_dir),
        "compare_dir": str(compare_dir) if compare_dir is not None else None,
        "suite_dir": str(suite_dir) if suite_dir is not None else None,
        "significance_dir": str(significance_dir) if significance_dir is not None else None,
        "result_health_dir": str(result_health_dir) if result_health_dir is not None else None,
        "admission_dir": str(admission_dir) if admission_dir is not None else None,
        "admission_calibration_dir": str(admission_calibration_dir) if admission_calibration_dir is not None else None,
        "result_diagnosis_dir": str(result_diagnosis_dir) if result_diagnosis_dir is not None else None,
        "delta_audit_dir": str(delta_audit_dir) if delta_audit_dir is not None else None,
        "signal_uplift_dir": str(signal_uplift_dir) if signal_uplift_dir is not None else None,
        "query_bank_compare_dir": str(query_bank_compare_dir) if query_bank_compare_dir is not None else None,
        "query_bank_promotion_decision_dir": str(query_bank_promotion_decision_dir)
        if query_bank_promotion_decision_dir is not None
        else None,
        "persistent_memory_main_compare_dir": str(persistent_memory_main_compare_dir)
        if persistent_memory_main_compare_dir is not None
        else None,
        "persistent_memory_main_decision_dir": str(persistent_memory_main_decision_dir)
        if persistent_memory_main_decision_dir is not None
        else None,
        "mainline_admission_cleanup_dir": str(mainline_admission_cleanup_dir)
        if mainline_admission_cleanup_dir is not None
        else None,
        "sample_contract_dir": str(sample_contract_dir) if sample_contract_dir is not None else None,
        "harder_sample_contract_dir": str(harder_sample_contract_dir) if harder_sample_contract_dir is not None else None,
        "mainline_admission_closure_dir": str(mainline_admission_closure_dir)
        if mainline_admission_closure_dir is not None
        else None,
        "query_strength_audit_dir": str(query_strength_audit_dir) if query_strength_audit_dir is not None else None,
        "provider_telemetry_dir": str(provider_telemetry_dir) if provider_telemetry_dir is not None else None,
        "provider_reachability_dir": str(provider_reachability_dir) if provider_reachability_dir is not None else None,
        "provider_normalization_dir": str(provider_normalization_dir) if provider_normalization_dir is not None else None,
        "query_promotion_pack_dir": str(query_promotion_pack_dir) if query_promotion_pack_dir is not None else None,
        "repeatability_audit_dir": str(repeatability_audit_dir) if repeatability_audit_dir is not None else None,
        "sample_size_recommendation_dir": str(sample_size_recommendation_dir) if sample_size_recommendation_dir is not None else None,
        "query_uplift_candidates_dir": str(query_uplift_candidates_dir) if query_uplift_candidates_dir is not None else None,
        "golden_real_sample_dir": str(golden_real_sample_dir) if golden_real_sample_dir is not None else None,
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
