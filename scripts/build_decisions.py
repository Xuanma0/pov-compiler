from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.l3_decisions.decision_compiler import DecisionCompiler
from pov_compiler.schemas import Output


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict):
        return data
    return {}


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _as_output(path: Path) -> Output:
    data = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        return Output.model_validate(data)  # type: ignore[attr-defined]
    return Output.parse_obj(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild decision_points from existing output JSON")
    parser.add_argument("--json", required=True, help="Input output JSON")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Config YAML path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    decision_cfg = dict(cfg.get("decisions", {}))

    output = _as_output(Path(args.json))
    compiler = DecisionCompiler(config=decision_cfg)
    output.decision_points = compiler.compile(output)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_model_dump(output), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"decision_points_total={len(output.decision_points)}")
    if output.decision_points:
        alts_min = min(len(dp.alternatives) for dp in output.decision_points)
        print(f"alternatives_min={alts_min}")
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
