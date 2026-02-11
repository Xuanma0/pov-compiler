# Changelog

All notable changes to this project are documented in this file.

## v0.2.0

### Added

- Perception v0 with pluggable backends:
  - `stub` backend for deterministic CI and fast smoke
  - `real` backend using YOLO + MediaPipe Tasks hand landmarker
- Event Segmentation v0 (`events_v0`) using visual change + contact fusion.
- Decision-centric highlights and top-level compression stats.
- Token codec (`token_codec.version=0.2`) and minimal context builder.
- Decision compiler with S-A-C-O structure and counterfactual alternatives.
- Vector index and retrieval stack:
  - query parser
  - query planner cascade
  - retriever merge
  - reranker with hard-constraint support
- NLQ evaluation modes:
  - `mock`
  - `pseudo_nlq`
  - `hard_pseudo_nlq` with distractor-aware strict metrics
- Cross-variant fixed-query evaluation and budget sweeps.
- Paper reporting pipeline:
  - figure/table generation
  - compare mode (run vs baseline)
  - snapshot metadata for reproducibility
- New docs set:
  - architecture
  - CLI reference
  - outputs schema guide
  - end-to-end reproducibility guide
  - FAQ

### Changed

- `make_paper_figures.py` now supports compare inputs via `--compare_dir` (snapshot-driven) or explicit compare paths.
- Added macro average alias support for both `--macro_avg` and `--macro-avg`.
- README reorganized for professor-friendly quick understanding and reproducible entry points.

### Notes

- Model and dataset artifacts are intentionally excluded from git history.
- Weight files are only downloaded on demand when missing.
