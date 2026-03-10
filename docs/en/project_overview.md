[中文对照](../zh-CN/project_overview.md)

# Project Overview

- Purpose: Explain what POV Compiler is and what problem it solves.
- Version scope: `v1.60+`
- Reading order: current mainline status -> experiment history -> this document -> repo map.
- Related docs: [current_mainline_status.md](./current_mainline_status.md), [repo_map.md](./repo_map.md), [experiment_history.md](./experiment_history.md)

## What POV Compiler Does

POV Compiler converts long first-person videos into layered artifacts that can be searched, evaluated, compared, and exported. The stack supports:

- offline compilation of per-video artifacts
- retrieval and NLQ-style evaluation
- budgeted and compare-driven reporting
- paper-ready and submission-ready evidence export

## Current Technical Story

The main technical story through `v1.60` is:

1. `YOLO26n` improved base object signal.
2. `YOLO26n + SAM3` improved object persistence and chain grounding.
3. object-memory logic uplift and then persistent object memory v2 turned those signals into stronger memory evidence.
4. large-sample real main compare showed persistent memory is worth promoting to mainline.
5. mainline admission is not fully closed yet because provider semantics still add residual caveat.

## What This Repo Is Not

- It is not a polished SDK-first package.
- It is not a single evaluation script.
- It is not a generic video perception framework.

It is best understood as a research and experiment workbench with strong provenance and reporting layers.
