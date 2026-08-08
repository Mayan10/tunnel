# Changelog

All notable changes to this project are documented in this file.

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
