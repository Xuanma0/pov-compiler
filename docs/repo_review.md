# Repo Review

## Overview

`pov-compiler` is an egocentric-video compilation and retrieval workbench. The center of gravity is `src/pov_compiler/pipeline.py:OfflinePipeline.run()`, which converts a raw first-person video into layered artifacts such as events, highlights, tokens, decisions, retrieval-friendly `events_v1`, optional perception traces, optional repository chunks, and downstream evaluation/export outputs.

The repo is not just a library. In practice it is a research and experiment platform with three modes of use:

- single-video offline compilation via `scripts/run_offline.py`
- retrieval, context building, and NLQ evaluation via `scripts/build_index.py`, `scripts/retrieve.py`, `scripts/build_context.py`, `scripts/eval_cross.py`, and `scripts/eval_nlq.py`
- large experiment orchestration via scripts such as `scripts/ego4d_smoke.py`, `scripts/run_ab_bye_compare.py`, `scripts/run_model_stack_compare.py`, and multiple sweep/compare utilities

Current repo shape:

- source files under `src/pov_compiler`: about 95 files
- tests under `tests`: 155 files
- runtime config lives in `configs/*.yaml`
- there is no packaging metadata such as `pyproject.toml`; most scripts bootstrap `src` onto `sys.path` directly

The repo currently looks most like a retrieval-centric multimodal experiment harness, not a polished reusable SDK.

## What the repo currently does

Primary capabilities already implemented:

- Offline video compilation:
  `scripts/run_offline.py` drives `pov_compiler.pipeline.OfflinePipeline`.
  The pipeline samples frames, embeds them, segments events, mines anchors/highlights, emits token sequences, compiles heuristic or model-backed decisions, builds `events_v1`, optionally runs perception, and optionally writes repository chunks.
- Retrieval and context assembly:
  `scripts/build_index.py` builds an embedding index through `pov_compiler.memory.index_builder.IndexBuilder`.
  `scripts/retrieve.py` executes structured or free-text retrieval through `pov_compiler.retrieval.retriever.Retriever`.
  `scripts/build_context.py` assembles timeline/highlight/decision/full/repo-aware contexts through `pov_compiler.context.context_builder.build_context`.
- Query planning and hard-constraint retrieval:
  `pov_compiler.retrieval.query_parser`, `query_planner`, `constraints`, `reranker`, and `trace` support structured query parsing, candidate generation, chain retrieval, hard constraints, reranking, and debug traces.
- Model-backed planning/decision/summary layers:
  `pov_compiler.l3_decisions.model_compiler`, `pov_compiler.retrieval.model_planner`, and `pov_compiler.repository.summarizer` all sit on the shared model client layer in `src/pov_compiler/models/`.
- Perception and object memory:
  `pov_compiler.perception.runner` can use `StubPerceptionBackend` or `RealPerceptionBackend`.
  `pov_compiler.perception.object_memory_v0` derives object memory from perception and event evidence.
- Repository memory:
  `pov_compiler.repository.writer` emits multi-scale repo chunks.
  `policy.py`, `reader.py`, `dedup.py`, and `summarizer.py` implement write policy, read policy, deduplication, and optional summary chunks.
- Streaming and budgeted online evaluation:
  `pov_compiler.streaming.runner` supports incremental indexing, budget-aware retrieval, chain backoff, and intervention policies.
- Evaluation and paper/export tooling:
  `scripts/eval_cross.py`, `scripts/eval_nlq.py`, `scripts/export_paper_ready.py`, `scripts/make_paper_figures.py`, and many compare/sweep scripts turn outputs into reports, tables, and paper-ready panels.
- BYE integration:
  `src/pov_compiler/integrations/bye/` exports events and orchestrates BYE-side runs and metric parsing.

Important entry points:

- CLI-like scripts:
  `scripts/run_offline.py`, `build_index.py`, `retrieve.py`, `build_context.py`, `gen_queries.py`, `eval_cross.py`, `eval_nlq.py`
- experiment runners:
  `scripts/ego4d_smoke.py`, `run_ab_bye_compare.py`, `run_model_stack_compare.py`
- key library modules:
  `src/pov_compiler/pipeline.py`, `schemas.py`, `retrieval/retriever.py`, `streaming/runner.py`, `repository/writer.py`
- tests:
  `tests/` mostly exercise behavior through public functions and scripts rather than through one large end-to-end pipeline test

## Architecture map

Compact repo map:

```text
configs/
  default.yaml
  repo_default.yaml
  rerank_default.yaml
docs/
  ARCHITECTURE.md
  CLI.md
  OUTPUTS.md
scripts/
  run_offline.py
  build_index.py
  retrieve.py
  build_context.py
  gen_queries.py
  eval_cross.py
  eval_nlq.py
  ego4d_smoke.py
  run_ab_bye_compare.py
  run_model_stack_compare.py
  export_paper_ready.py
src/pov_compiler/
  pipeline.py
  schemas.py
  l1_events/
  l2_tokens/
  l3_decisions/
  ir/
  perception/
  memory/
  retrieval/
  repository/
  context/
  streaming/
  eval/
  bench/nlq/
  models/
  integrations/bye/
tests/
```

Layer responsibilities:

- `l1_events/`
  Event segmentation from frame embeddings and motion signals.
  Core entry: `event_segmenter.segment_events`.
- `l2_tokens/`
  Tokenization of event windows into retrieval-friendly discrete signals.
  Core entry: `token_codec.compile_token_codec`.
- `l3_decisions/`
  Heuristic and model-backed decision compilation.
  Core entries: `decision_compiler.compile_decisions`, `model_compiler.compile_decisions_with_model_and_meta`.
- `ir/`
  Contract enrichment from raw events/highlights/tokens/decisions into `events_v1`.
  Core entry: `events_v1.build_events_v1`.
- `perception/`
  Stub or real vision backends, contact scoring, and object memory.
- `memory/`
  Vector index building and storage.
- `retrieval/`
  Query parsing, planning, constraint application, reranking, chain logic, and tracing.
- `repository/`
  RepoV1 chunk writing, deduplication, summary synthesis, and query-aware read policy.
- `context/`
  Builds bounded contexts for prompts or downstream consumers.
- `streaming/`
  Online budget policies, codecs, interventions, and runner.
- `eval/` and `bench/nlq/`
  Query generation, evaluation, ablation, sweep logic, safety gating, and reporting.
- `models/`
  Provider presets, client transport, capability inference, structured output strategy selection, cost estimation, and cache.
- `integrations/bye/`
  Export and metric plumbing for external BYE tooling.

Critical call chain:

`scripts/run_offline.py`
-> `pov_compiler.pipeline.OfflinePipeline.run()`
-> `l1_events`, `l2_tokens`, `l3_decisions`, `perception`, `ir.events_v1`, `repository.writer`
-> `Output` from `src/pov_compiler/schemas.py`
-> index/retrieval/context/eval/export scripts

## Key execution flows

### 1. Offline compile flow

Real path:

- `scripts/run_offline.py`
- loads `configs/default.yaml` plus CLI overrides
- instantiates `OfflinePipeline`
- `OfflinePipeline.run(video_path)` does:
  - sample video frames with `utils.video.VideoReader`
  - embed frames with `features.embedder.Embedder`
  - compute motion and embedding-change signals
  - segment `events`
  - mine anchors and highlights
  - optionally run perception through `perception.runner.run_perception`
  - optionally create perception-aware `events_v0` with `l1_events.event_segmentation_v0.segment_events_v0`
  - compile `token_codec`
  - compile heuristic decisions
  - optionally compile `decisions_model_v1`
  - build `events_v1`
  - build `object_memory_v0`
  - optionally build and deduplicate repository chunks
  - serialize via `output_to_dict`

### 2. Retrieval flow

Real path:

- `scripts/build_index.py`
  builds `.index.npz` and `.index_meta.json`
- `scripts/retrieve.py`
  parses query DSL, runs `Retriever.retrieve()`, and returns ranked hits
- `scripts/build_context.py`
  optionally retrieves first, then turns hits and/or repo chunks into bounded prompt context

Retrieval behavior details:

- `retrieval/query_parser.py` parses structured fields such as `anchor=`, `token=`, `decision=`, `place=`, `interaction_object=`, `lost_object=`, and chain fields
- `retrieval/query_planner.py` can produce heuristic candidate queries and chain plans
- `retrieval/model_planner.py` can replace the heuristic planner with a structured-output model planner
- `retrieval/constraints.py` applies hard filters and relaxation/backoff
- `retrieval/reranker.py` scores semantic match, intent bonus, place/object match, scene penalties, and distractor penalties
- `retrieval/trace.py` produces debug reports for one query

### 3. Repository-aware retrieval flow

When repo mode is enabled:

- `repository/writer.py` emits multiscale chunks: `event`, `decision`, `place`, `window`, `segment`, and optional `summary`
- `repository/dedup.py` removes overlapping or redundant chunks
- `repository/policy.py:QueryAwareReadPolicyV0` selects chunks using intent, time/place/object hints, recency, dedup, and per-level caps
- `context/context_builder.py` can return `repo_chunks` plus `repo_trace` for repo-only or mixed contexts

### 4. Evaluation and compare flow

- `scripts/gen_queries.py` builds fixed query sets from one output JSON
- `scripts/eval_cross.py` evaluates cross-variant retrieval under budget sweeps
- `scripts/eval_nlq.py` runs pseudo, hard-pseudo, chain, or Ego4D-style NLQ evaluation, plus safety reporting
- `scripts/ego4d_smoke.py` orchestrates batch compile/index/query/eval/NLQ/BYE/perception runs
- `scripts/run_ab_bye_compare.py` and `scripts/run_model_stack_compare.py` are higher-level experiment harnesses for comparing perception backends, planner backends, decision backends, and model stacks

### 5. Export flow

- `integrations/bye/exporter.py` exports POV outputs to BYE event JSONL
- `scripts/export_paper_ready.py` aggregates compare outputs into a unified budget panel with copied tables, figures, and a `snapshot.json`

## Model stack design

The model layer is relatively well-factored compared with the rest of the repo.

Core stack:

- presets:
  `src/pov_compiler/models/presets.py`
  normalizes providers and default env var names
- config and transport:
  `src/pov_compiler/models/client.py`
  defines `ModelClientConfig`, base URL normalization, env-only API key lookup, and cached-client wrapping
- provider implementation:
  `openai_compat.py`, `gemini.py`, `fake.py`
- capability inference:
  `capabilities.py`
  supports static/probed capability detection and probe caching
- structured output:
  `structured_output.py`
  auto-selects among `json_schema`, `json_object`, `tool`, `prompted_json`, and parse-repair fallbacks
- cost:
  `cost.py`
  best-effort cost estimation via `litellm`
- cache:
  `cache.py`
  file-based model call cache under `data/outputs/model_cache`

Task consumers built on that stack:

- decision generation:
  `l3_decisions/model_compiler.py`
- retrieval planning:
  `retrieval/model_planner.py`
- repo summaries:
  `repository/summarizer.py`

Design observations:

- API keys are intentionally env-only. Config stores env var names such as `OPENAI_API_KEY` and `GEMINI_API_KEY`, not literal secrets.
- The fake provider is a first-class path, not a toy stub. It is used to keep tests and CI deterministic.
- Structured output is a cross-cutting concern. Reliability depends on `structured_output.py`, not on any one task-specific caller.
- Multiple providers are normalized behind an OpenAI-compatible abstraction, but Gemini also has a native path.

## Data / artifact contract

### Core in-memory/output contract

The main contract is `src/pov_compiler/schemas.py:Output`.

Top-level fields with real downstream consumers:

- `video_id`
- `meta`
- `stats`
- `events`
- `events_v0`
- `events_v1`
- `highlights`
- `token_codec`
- `decision_points`
- `decisions_model_v1`
- `perception`
- `object_memory_v0`
- `repository`
- `debug.signals`

Key schema meanings:

- `events`
  baseline event segmentation from frame-level signals
- `events_v0`
  perception-aware event layer
- `events_v1`
  retrieval-oriented enriched event layer with evidence, place segments, and interaction signature
- `token_codec`
  discrete symbolic timeline representation
- `decision_points`
  heuristic S-A-C-O style decisions
- `decisions_model_v1`
  model-generated decision layer, preferred by some consumers when present
- `object_memory_v0`
  last-seen / last-contact object state
- `repository`
  multiscale repo chunks plus summary/trace data

### Repository chunk contract

`src/pov_compiler/repository/schema.py:RepoChunk` defines:

- identity:
  `id`, `chunk_id`
- temporal span:
  `t0`, `t1`, `t0_ms`, `t1_ms`
- level/scale:
  `level`, `scale`
- content:
  `text`, `importance`, `source_ids`, `tags`
- metadata:
  `score_fields`, `payload`, `meta`

Supported levels are normalized to:

- `event`
- `decision`
- `place`
- `segment`
- `window`
- `summary`

### Summary chunk contract

`src/pov_compiler/repository/summary_schema.py:RepoSummaryV0` stores:

- goals
- step summaries with time spans and evidence ids
- entities
- places
- interactions
- decisions
- provenance

### Index and retrieval artifacts

Per `docs/OUTPUTS.md` and `memory/index_builder.py`:

- index files:
  `data/cache/<video_uid>.index.npz`
  `data/cache/<video_uid>.index_meta.json`
- batch output roots:
  `data/outputs/<run>/json/`
  `data/outputs/<run>/cache/`
  `data/outputs/<run>/eval/`
  `data/outputs/<run>/nlq/`
  `data/outputs/<run>/perception/`
  `data/outputs/<run>/event/`

### Config, environment, cache, and snapshot conventions

- default runtime config:
  `configs/default.yaml`
- repo-focused config:
  `configs/repo_default.yaml`
- reranker config:
  `configs/rerank_default.yaml`
- model cache default:
  `data/outputs/model_cache`
- perception cache default:
  `data/cache/perception` or output-adjacent `perception_cache`
- paper-ready exporter snapshot:
  `scripts/export_paper_ready.py` writes `snapshot.json`

Important invariant:

many downstream tools assume these output keys and directory names already exist. Refactors that rename fields like `events_v1`, `decision_points`, `decisions_model_v1`, `repo_chunks`, or `repo_trace` will cascade widely.

## Test coverage map

Coverage is broad by subsystem and script behavior, but not centered on a single end-to-end golden pipeline test.

### Retrieval, constraints, reranking, and trace

Representative tests:

- `tests/test_retriever.py`
- `tests/test_retriever_chain.py`
- `tests/test_retrieve_decisions.py`
- `tests/test_constraints_filtering.py`
- `tests/test_constraints_chain_time_place_object.py`
- `tests/test_reranker_constraints.py`
- `tests/test_reranker_decision_features.py`
- `tests/test_trace_one_query_chain.py`
- `tests/test_trace_query.py`

What this protects:

- query parsing and planner candidate generation
- hard constraint semantics and backoff
- chain-derived time/place/object constraints
- reranker weighting and intent bonuses
- trace/debug payload stability

### Repository, context, place/object memory, and summary-first retrieval

Representative tests:

- `tests/test_repo_writer.py`
- `tests/test_repo_multiscale_writer_v1.py`
- `tests/test_repo_query_aware_read_policy.py`
- `tests/test_repo_dedup_cross_scale_v1.py`
- `tests/test_repo_reader.py`
- `tests/test_context_builder.py`
- `tests/test_context_builder_includes_summary_first.py`
- `tests/test_object_memory_v0.py`
- `tests/test_events_v1_converter.py`

What this protects:

- repo chunk shapes and multiscale writing
- query-aware read policy scoring and selection
- dedup behavior across chunk levels
- context modes and repo inclusion
- `events_v1` enrichment and object-memory derivation

### Model stack and structured output

Representative tests:

- `tests/test_model_clients_smoke.py`
- `tests/test_model_call_cache.py`
- `tests/test_model_capabilities_probe_cache.py`
- `tests/test_model_planner_fallback.py`
- `tests/test_structured_output_auto_strategy.py`
- `tests/test_structured_output_fallback_parse.py`
- `tests/test_structured_output_meta_redaction.py`
- `tests/test_provider_presets.py`

What this protects:

- provider normalization
- env-only secret handling
- model cache behavior
- planner fallback when model path is unavailable
- structured-output strategy selection and parse fallback
- metadata redaction

### Streaming and budget policies

Representative tests:

- `tests/test_streaming_runner.py`
- `tests/test_streaming_budget_smoke.py`
- `tests/test_streaming_repo_integration.py`
- `tests/test_streaming_chain_backoff_policy.py`
- `tests/test_streaming_chain_budget_split.py`
- `tests/test_budget_policy.py`
- `tests/test_recommend_budget.py`
- `tests/test_intervention_action_selection.py`

What this protects:

- online indexing/retrieval loops
- codec selection and fixed-K behavior
- budget recommendation and safety-latency policies
- chain backoff and intervention logic
- repo-aware streaming integration

### Perception, segmentation, tokenization, and heuristic decisions

Representative tests:

- `tests/test_segmenter.py`
- `tests/test_event_segmentation_v0.py`
- `tests/test_token_codec.py`
- `tests/test_decision_compiler.py`
- `tests/test_perception_stub.py`
- `tests/test_perception_real_fallback.py`
- `tests/test_contact_heuristic.py`
- `tests/test_highlights.py`

What this protects:

- event boundary generation
- perception-aware event segmentation
- token emission
- heuristic decision windows and merge behavior
- stub/real perception fallback

### Evaluation, reporting, BYE, and experiment harnesses

Representative tests:

- `tests/test_eval_metrics.py`
- `tests/test_eval_nlq_chain_metrics.py`
- `tests/test_safety_gate.py`
- `tests/test_cross_variant_eval.py`
- `tests/test_bye_exporter.py`
- `tests/test_bye_report_parse.py`
- `tests/test_run_model_stack_compare_smoke.py`
- `tests/test_run_ab_bye_compare_smoke.py`
- `tests/test_ego4d_smoke_plan.py`
- `tests/test_export_paper_ready_smoke.py`

What this protects:

- NLQ metrics and safety reporting
- BYE export/report parsing
- compare harness plumbing
- batch smoke orchestration and resume behavior
- paper-ready export artifacts

### Coverage gaps

- There is no strong direct test layer around `OfflinePipeline.run()` as one contract-bearing public workflow.
- Very large orchestration scripts are mostly smoke-tested, which catches wiring regressions but not deep semantic drift.
- Artifact compatibility is spread across many tests rather than enforced by a small set of explicit schema/fixture goldens.

## Technical debt and risks

### Structural debt

1. Orchestrator concentration.
   `src/pov_compiler/pipeline.py` is the central orchestrator and already spans about 470 lines. It owns too many decisions: perception toggles, decision backend selection, `events_v1` assembly, object memory, and repository writing.

2. Oversized high-coupling modules.
   `src/pov_compiler/retrieval/retriever.py` is about 1615 lines.
   `src/pov_compiler/streaming/runner.py` is about 1820 lines.
   `scripts/eval_nlq.py` is about 997 lines.
   These modules combine parsing, policy, orchestration, metrics, fallback, and reporting in one file, making safe edits expensive.

3. Repeated helper logic.
   `_load_yaml`, `_as_output`, and related normalization helpers are duplicated across many scripts such as `build_context.py`, `build_index.py`, `eval_cross.py`, `eval_nlq.py`, `gen_queries.py`, `trace_one_query.py`, and others.

4. Script-first import model.
   Many scripts manually inject `src` into `sys.path` instead of relying on an installed package. This keeps local iteration easy, but it increases duplication and makes packaging/tooling less coherent.

### Contract risk

1. Implicit cross-module contracts.
   Retrieval, repo read policy, chain derivation, `events_v1`, object memory, and context assembly share many field names and assumptions that are not centralized in one compatibility layer.

2. Decision-pool ambiguity.
   Several consumers prefer `decisions_model_v1` if present and otherwise fall back to `decision_points`. That is convenient, but it makes behavior sensitive to optional fields and backend selection.

3. Artifact shape dependence.
   Evaluation and export scripts rely on stable filenames, directory layouts, and CSV column names. These contracts exist mostly by convention plus tests, not by one formal manifest schema.

### Concrete repo hygiene issues

1. Version mismatch.
   `README.md` and `docs/ARCHITECTURE.md` say `v0.2.0`, but `src/pov_compiler/__init__.py` exposes `__version__ = "0.1.0"`.

2. Mojibake in query planner literals.
   `src/pov_compiler/retrieval/query_planner.py` contains visibly corrupted non-ASCII keyword literals. That is a real retrieval-quality risk for multilingual query planning.

3. Checked-in/generated cache artifacts.
   The repo tree contains many `__pycache__` directories under `scripts/`, `src/pov_compiler/`, and `tests/`. That is low severity functionally, but it is a hygiene and review-noise problem.

### Most fragile chains

- query parsing -> constraint derivation -> reranker bonuses -> strict metrics
- `events_v1` generation -> object memory -> place/object-aware retrieval
- repository chunk writing -> dedup -> query-aware read policy -> context assembly
- model structured-output reliability -> planner/decision/summary quality -> evaluation deltas
- large compare scripts stitching together many output directories and optional panels

### Most easily forgotten context

- `events`, `events_v0`, and `events_v1` are three different layers with different downstream roles
- repo mode is optional, but several context and retrieval paths become materially different when it is on
- the fake provider is an intentional first-class execution path and should not be treated as test-only scaffolding
- streaming has its own budget/intervention logic and is not just "offline retrieval in a loop"

## Recommended next refactors

### 1. Extract a stable contract layer and shared IO utilities

Priority: highest

Refactor target:

- centralize YAML loading, `Output` coercion, output-path conventions, and common CLI helper code
- create one canonical compatibility module for artifact loading and field preference logic such as heuristic vs model decisions

Why first:

- it reduces duplicate risk across almost every script
- it creates a single place to stabilize contracts before deeper refactors
- it lowers the cost of adding new tasks without further copy/paste growth

### 2. Split retrieval into explicit phases with typed boundaries

Priority: high

Refactor target:

- break `retrieval/retriever.py` into query normalization, candidate generation, candidate collection, constraint filtering, reranking, and trace assembly
- isolate chain retrieval as its own submodule with typed inputs/outputs

Why second:

- retrieval is the most central runtime path after offline compilation
- many metrics and failure modes originate here
- current file size makes behavioral changes hard to reason about

### 3. Separate experiment orchestration from reusable library behavior

Priority: high

Refactor target:

- shrink `streaming/runner.py`, `scripts/eval_nlq.py`, `scripts/ego4d_smoke.py`, and compare scripts into smaller reusable services plus thin CLIs
- unify compare/sweep manifests so output collection is less ad hoc

Why third:

- the repo already behaves like an experiment platform
- orchestration sprawl is where context is most often lost between sessions
- this would make future Codex work smaller, safer, and easier to validate

## Glossary of project-specific terms

- `events`
  Primary event segments from motion/embedding boundary signals.
- `events_v0`
  Perception-aware event segmentation layer.
- `events_v1`
  Retrieval-oriented enriched event layer with evidence, place segments, and interaction summaries.
- `highlight`
  A compact key clip around an anchor, usually used as the highest-priority retrieval unit.
- `token_codec`
  Discrete symbolic representation of the video timeline, including items like `SCENE_CHANGE`, `MOTION_*`, and `INTERACTION`.
- `decision_points`
  Heuristic S-A-C-O decision windows compiled from highlights/anchors.
- `decisions_model_v1`
  Model-generated decision layer that some consumers prefer over heuristic decisions.
- `object_memory_v0`
  Per-object last-seen / last-contact memory derived from perception and events.
- `repository`
  Multi-scale memory store of text chunks about the video, used for repo-aware retrieval and context assembly.
- `repo summary`
  A `summary`-level repo chunk, optionally model-written, that compresses a larger time window.
- `summary-first retrieval`
  Retrieval plan that first narrows using summary chunks, then drills into finer-grained artifacts.
- `chain retrieval`
  Two-step retrieval where step 1 derives constraints for step 2, such as time/place/object hints.
- `BYE`
  External integration target with export, run-package, and metric/report plumbing under `integrations/bye/`.

## Bottom line

This repo currently behaves like a layered egocentric-video retrieval research system with strong experiment tooling. What it lacks most is a tighter contract boundary between reusable library code and growing orchestration code, especially around retrieval, artifact loading, and experiment runners.
