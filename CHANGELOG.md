# Changelog

All notable changes to this project are documented in this file.

## [0.3.0] - 2026-08-08

### Added

- `audio-embedding-v1`: a Core ML neural embedding model, run on-device with Apple Neural Engine
  acceleration, that augments the transition ranker with a learned notion of sonic similarity.
  Ships as a bundled `.mlmodel`; requires the optional `pip install "tunnel[ml]"` extra
  (`coremltools`) to run, with a clean fallback to `playlist-audio-ml-v1` when it is not installed.
- `tools/train_embedding_model.py`: the reproducible training script for `audio-embedding-v1`,
  trained locally on procedurally generated acoustic archetypes.
- `scripts/benchmark_ordering.py`: a model-independent benchmark comparing every ordering engine
  against the source playlist order and against each other.
- 4-band spectral energy features in `AudioFeatures`, used as embedding model input.

### Changed

- CI now includes a dedicated job that installs the `ml` extra and runs the Core ML model against
  the shipped `.mlmodel` file on every push.

## [0.2.0] - 2026-06-06

### Added

- Local ML ordering engine (`playlist-ml-v1`, `playlist-audio-ml-v1`) that trains a transition
  ranker directly from each playlist instead of relying on a single fixed heuristic.
- Local audio feature extraction (`audio-hybrid-v1`) using macOS `afconvert`, with a persistent
  on-disk cache.

### Fixed

- Disabled the destructive in-place playlist reorder path. Tunnel now only ever creates new
  playlists and verifies the resulting track count before finishing.

## [0.1.1] - 2026-06-06

### Fixed

- Fixed the release workflow's asset upload step.

## [0.1.0] - 2026-06-06

### Added

- Initial release: interactive terminal app and `list` / `export` / `order` / `restore` CLI
  commands, metadata-based ordering (`metadata-flow-v0`), playlist snapshotting, and the
  `tunnel.pyz` zipapp build/release pipeline.
