from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository.policy import (
    PolicyConfig,
    build_read_policy,
    build_write_policy,
    dump_policy_yaml,
    load_policy_yaml,
    policy_cfg_hash,
)


def test_repo_policy_io_and_hash_stable(tmp_path: Path) -> None:
    payload = {
        "repo": {
            "write_policy": {"name": "novelty", "novelty_threshold": 0.4, "max_reference": 6},
            "read_policy": {"name": "diverse", "max_chunks": 8, "max_tokens": 160, "diversity_threshold": 0.85},
        }
    }
    path = tmp_path / "repo_policy.json"
    dump_policy_yaml(path, payload)
    loaded = load_policy_yaml(path)
    assert loaded == payload

    repo_cfg = dict(loaded["repo"])
    h1 = policy_cfg_hash(repo_cfg)
    h2 = policy_cfg_hash(json.loads(json.dumps(repo_cfg)))
    assert h1 == h2

    wp = build_write_policy(repo_cfg["write_policy"])
    rp = build_read_policy(repo_cfg["read_policy"])
    assert wp.name == "novelty"
    assert rp.name == "diverse"

    cfg = PolicyConfig(name="novelty", params={"novelty_threshold": 0.4})
    assert cfg.stable_hash() == PolicyConfig(name="novelty", params={"novelty_threshold": 0.4}).stable_hash()

