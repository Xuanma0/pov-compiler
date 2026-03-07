from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.prompts.registry import PromptRegistry, write_prompt_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prompt registry and optionally export a prompt lock.")
    parser.add_argument("--registry", required=True, help="Prompt registry YAML path")
    parser.add_argument("--profile", default=None, help="Optional prompt profile to resolve")
    parser.add_argument("--out-lock", default=None, help="Optional prompt lock JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    registry = PromptRegistry.from_path(registry_path)
    rows = registry.validate_paths(registry_path)
    missing = [row for row in rows if not bool(row.get("exists", False))]
    profile = args.profile or (registry.available_profiles()[0] if registry.available_profiles() else None)

    print(f"registry_id={registry.registry_id}")
    print(f"registry_version={registry.version}")
    print(f"entries_total={len(registry.entries)}")
    print(f"profiles_total={len(registry.profiles)}")
    print(f"missing_paths={len(missing)}")
    if profile:
        lock = registry.build_prompt_lock(registry_path, profile)
        print(f"profile={profile}")
        print(f"profile_prompts={len(lock.get('prompts', []))}")
        if args.out_lock:
            out_path = write_prompt_lock(lock, args.out_lock)
            print(f"saved_lock={out_path}")
    if missing:
        print("missing_detail=" + json.dumps(missing, ensure_ascii=False, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
