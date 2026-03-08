# Provider Probe

## Goal

- Determine whether the current "parallel science" provider is likely compatible with the repo's existing `openai_compat` path.
- Do not print or store any secret values.

## Current Environment Probe

Checked env presence only:

- `PARATERA_API_KEY=false`
- `PARATERA_BASE_URL=false`
- `PARATERA_MODEL=false`
- `OPENAI_API_KEY=false`
- `OPENAI_BASE_URL=false`

No live provider env is currently present in this shell, so no real health request was attempted against the parallel-science platform.

## What the Repo Already Supports

The current codebase already supports an OpenAI-compatible path:

- provider preset:
  - `openai_compat`
- default env names:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
- override points already supported:
  - config fields `api_key_env`, `base_url`, `base_url_env`
  - CLI flags such as `--model-base-url` and `--model-api-key-env`

Key evidence in repo:

- `src/pov_compiler/models/openai_compat.py`
  - calls `POST <base_url>/responses`
  - falls back to `POST <base_url>/chat/completions`
- `src/pov_compiler/models/presets.py`
  - `openai_compat` default preset uses base URL override and bearer auth
- `scripts/model_health_check.py`
  - supports dry-run / real probe with custom `--base-url` and `--api-key-env`

## Compatibility Judgment

### Most likely compatibility mode

- `OpenAI-compatible with base_url override`

### Most likely route behavior

- first guess:
  - `responses`
- fallback:
  - `chat/completions`

### Current judgment

- `possible_direct_openai_compat=true`
- confidence:
  - medium, not high

Reason:

- The repo's `openai_compat` client is generic and already handles OpenAI-like endpoints with:
  - custom `base_url`
  - custom `api_key_env`
  - `responses -> chat/completions` fallback
- That is exactly the shape needed for most "one key, many models" OpenAI-compatible gateways.

## What Is Missing

The following evidence is still missing before calling the platform "confirmed compatible":

1. A real `PARATERA_BASE_URL`
   - redacted value is enough; secret value is not needed in docs.
2. A real `PARATERA_MODEL`
   - at least one known working model ID on that platform.
3. One health-style probe result
   - does `POST /v1/responses` work?
   - if not, does `POST /v1/chat/completions` work?
4. Usage / cost semantics
   - whether the platform returns OpenAI-like `usage` fields, partial usage, or none.

If any of the above fails, the provider may still be usable, but only as:

- `chat/completions` only
- partial OpenAI-compatible
- or gateway-specific, which would need one more adapter layer outside runtime core

## Recommended Env Naming

### Preferred non-invasive path

Keep the repo runtime unchanged and set:

- `PARATERA_API_KEY`
- `PARATERA_BASE_URL`
- `PARATERA_MODEL`

Then pass them into the existing model path via config / CLI:

- provider:
  - `openai_compat`
- `api_key_env=PARATERA_API_KEY`
- `base_url=${PARATERA_BASE_URL}`
- model:
  - `${PARATERA_MODEL}`

### Fallback aliasing path

If you want zero new config keys for an initial smoke:

- map the same values into:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`

This is operationally simpler, but less explicit than keeping `PARATERA_*`.

## Suggested Provider Preset for v1.52

Recommended first preset:

- provider:
  - `openai_compat`
- api mode:
  - `auto`
- route expectation:
  - try `responses`, tolerate fallback to `chat/completions`

If the platform is known to be chat-only, safer first probe:

- provider:
  - `openai_compat`
- api mode:
  - `chat`

## Dry-Run Readiness Conclusion

- Continue using `openai_compat`:
  - `yes`
- Is the parallel-science platform already proven compatible:
  - `no`
- Is it likely directly adaptable:
  - `yes`, if it exposes an OpenAI-style bearer-auth base URL and at least one model ID

## Next Step for v1.52

Minimal next live probe once env is set:

```powershell
python scripts/model_health_check.py --provider openai_compat --model $env:PARATERA_MODEL --base-url $env:PARATERA_BASE_URL --api-key-env PARATERA_API_KEY --dry-run
```

Then, if dry-run looks correct:

```powershell
python scripts/model_health_check.py --provider openai_compat --model $env:PARATERA_MODEL --base-url $env:PARATERA_BASE_URL --api-key-env PARATERA_API_KEY --real
```

If `responses` fails but `chat/completions` works, keep using `openai_compat`, but pin:

- `api_mode=chat`
