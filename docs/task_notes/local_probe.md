# Local Probe Note

## Scope

- Task type:
  - local inventory + provider adaptation probe
- Boundaries:
  - no runtime changes
  - no script logic changes
  - no secrets written into repo

## Probe Commands

### Repo/document/provider code scan

```powershell
Get-Content docs/working_memory.md -TotalCount 120
Get-Content docs/repo_review.md -TotalCount 120
Get-Content docs/active_plan.md -TotalCount 140
rg -n "openai_compat|PARATERA|OPENAI|base_url|api_key|responses|chat/completions|chat.completions" src scripts configs tests
```

### Local directory inventory

```powershell
Get-ChildItem D:\BYES -Force
Get-ChildItem D:\BYES\repos -Force
Get-ChildItem D:\BYES\models -Force
Get-ChildItem D:\Ego4D_Dataset -Force
Get-ChildItem D:\ILSVRC2017_DET_test_new -Force
```

### Structured inventory extraction

```powershell
@'
# inline python used to enumerate repo roots, model weights, and dataset summaries
'@ | python -
```

### Env presence check

```powershell
@'
import os, json
keys = ["PARATERA_API_KEY","PARATERA_BASE_URL","PARATERA_MODEL","OPENAI_API_KEY","OPENAI_BASE_URL"]
print(json.dumps({k: bool(os.environ.get(k, "")) for k in keys}, indent=2))
'@ | python -
```

## Key Outputs

### BYES repo roots

- `D:\BYES\repos\Depth-Anything-3`
  - git repo: yes
  - `README.md`: yes
  - `requirements.txt`: yes
  - `pyproject.toml`: yes
- `D:\BYES\repos\sam3`
  - git repo: yes
  - `README.md`: yes
  - `requirements.txt`: not found at repo root
  - `pyproject.toml`: yes

### Key model checkpoints

- `D:\BYES\models\yolo26\yolo26n.pt`
- `D:\BYES\models\yolo26\yolo26s.pt`
- `D:\BYES\models\sam3\sam3\sam3.pt`
- `D:\BYES\models\sam3\sam3\model.safetensors`
- `D:\BYES\models\da3\DA3-LARGE-1.1\model.safetensors`
- `D:\BYES\models\da3\DA3NESTED-GIANT-LARGE-1.1\model.safetensors`

### Dataset summary

- Ego4D:
  - `D:\Ego4D_Dataset\v2_packed\annotations`
    - 34 files
  - `D:\Ego4D_Dataset\v2_packed\full_scale_0000\full_scale`
    - 79 mp4 files
    - about 48.04 GB
- ILSVRC:
  - `D:\ILSVRC2017_DET_test_new\ILSVRC\Data\DET\test`
    - 5500 JPEG files
  - `D:\ILSVRC2017_DET_test_new\ILSVRC\ImageSets\DET\test.txt`
    - present

### Provider env presence

```json
{
  "PARATERA_API_KEY": false,
  "PARATERA_BASE_URL": false,
  "PARATERA_MODEL": false,
  "OPENAI_API_KEY": false,
  "OPENAI_BASE_URL": false
}
```

### Provider compatibility read

- Current repo already supports:
  - `openai_compat`
  - `base_url` override
  - custom `api_key_env`
  - `responses -> chat/completions` fallback
- Current conclusion:
  - Paratera is likely adaptable through `openai_compat`
  - but not yet proven, because no live env values are present and no live probe was executed

## Risks

1. Provider evidence gap
   - No current `PARATERA_*` env values in this shell, so compatibility remains inferred, not proven.
2. Query / sample evidence gap
   - Real provider closure exists in repo history, but not specifically against the parallel-science platform.
3. Model integration risk
   - `sam3` and `DA3` are strong candidates, but they are heavier and operationally riskier than `yolo26n/s`.

## Next-Step Recommendations

1. For v1.52 provider probe, start with:
   - provider: `openai_compat`
   - envs:
     - `PARATERA_API_KEY`
     - `PARATERA_BASE_URL`
     - `PARATERA_MODEL`
2. First live smoke should use:
   - `scripts/model_health_check.py`
   - dry-run first, then real
3. First local model to integrate should be:
   - `YOLO26n`
   - because it is the lightest direct perception checkpoint already present locally
4. Second model family should be:
   - `SAM3`
   - because it adds segmentation/tracking and already shows Ego4D-specific signals in local repo config
