# Tunnel

[![CI](https://github.com/Mayan10/tunnel/actions/workflows/ci.yml/badge.svg)](https://github.com/Mayan10/tunnel/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Mayan10/tunnel?sort=semver)](https://github.com/Mayan10/tunnel/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey)](README.md#requirements)

Tunnel is a terminal app that reorders Apple Music playlists into smoother listening flows. It trains a
lightweight transition model on your playlist and, when local audio files are available, blends in
tempo, energy, brightness, and dynamics extracted directly from the audio. Everything runs locally: no
network calls, no accounts, no cloud ML.

## Features

- Reorders playlists to minimize jarring tempo, energy, and genre transitions
- Learns transition weights from each playlist instead of using one fixed heuristic
- Extracts audio features locally via macOS `afconvert`, no external audio libraries
- Optional Core ML neural embedding layer (`audio-embedding-v1`) for a learned notion of sonic similarity, accelerated by the Apple Neural Engine
- Never mutates the source playlist; always writes to a new playlist and verifies the track count
- Snapshots the source playlist to `~/.tunnel/snapshots/` before every reorder, and can restore it
- Works as both an interactive terminal app and a scriptable CLI

## Requirements

- macOS with the Music app
- Python 3.11 or newer
- Apple Music automation permission for Terminal or your terminal app
- Downloaded music files for audio analysis. Cloud-only Apple Music tracks can still be ordered, but Tunnel cannot analyze their waveform.

## Install

Download `tunnel.pyz` from the [latest release](https://github.com/Mayan10/tunnel/releases/latest).

```bash
chmod +x tunnel.pyz
./tunnel.pyz
```

To install it as a normal command:

```bash
mkdir -p "$HOME/.local/bin"
cp tunnel.pyz "$HOME/.local/bin/tunnel"
chmod +x "$HOME/.local/bin/tunnel"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
tunnel
```

## Run Locally

From this repository:

```bash
./bin/tunnel
```

Install the local build as `tunnel`:

```bash
./scripts/install.sh
tunnel
```

Build the release artifact:

```bash
python3 scripts/build_zipapp.py
```

The build output is `dist/tunnel.pyz`.

## Use

Open the interactive app:

```bash
tunnel
```

The app lets you:

- choose an Apple Music playlist
- analyze the tracks
- preview the new order
- create a new ordered playlist
- save the ordered result as JSON

Power-user commands:

```bash
tunnel list
tunnel order "My Playlist" --create-playlist "My Playlist - Flow"
tunnel export "My Playlist" --out exports/my-playlist.json
tunnel restore "My Playlist - Before Tunnel abc123" --to "My Playlist Restored"
```

Tunnel does not mutate original playlists. It creates new ordered playlists and verifies the track count after writing.
Before ordering a playlist from Apple Music, Tunnel also writes a local snapshot to `~/.tunnel/snapshots/`.

## Ordering Engine

Tunnel picks the strongest engine it can run for a given playlist, in this order:

- `audio-embedding-v1`: a compact neural network, packaged as a Core ML model, that turns each
  track's local audio features into an 8-dimensional embedding. It runs on-device through Core ML,
  which transparently uses the Apple Neural Engine, GPU, or CPU depending on hardware. The embedding
  augments the playlist's learned transition weights with a learned notion of "does this sound
  similar," on top of the handcrafted tempo/energy/genre features. Requires the optional `ml` extra
  (see [Neural embeddings](#neural-embeddings) below).
- `playlist-audio-ml-v1`: the trained transition ranker plus local audio features, without the
  Core ML embedding layer (used when the `ml` extra is not installed).
- `playlist-ml-v1`: trains a lightweight transition ranker from the source playlist order, then uses
  the learned weights to build the new order. No local audio required.
- `audio-hybrid-v1` / `metadata-flow-v0`: untrained fallbacks used when a playlist has fewer than 4
  tracks, with (`audio-hybrid-v1`) or without (`metadata-flow-v0`) local audio features.

For best results, download the playlist in Music before running Tunnel. If the output says `0 local files`, Tunnel cannot analyze audio for those songs.

### Neural embeddings

`audio-embedding-v1` ships as a bundled `.mlmodel` file, but running it requires `coremltools`,
which is not part of Tunnel's zero-dependency default install:

```bash
pip install "tunnel[ml]"
```

Without it, Tunnel falls back to `playlist-audio-ml-v1` and prints a note telling you how to enable
the embedding layer. The model itself is a 3-layer feed-forward network trained with a triplet loss
on procedurally generated acoustic archetypes (ambient, acoustic, classical, pop, dance, rock)
spanning realistic tempo, energy, brightness, dynamics, and spectral-balance ranges. There is no real
audio and no network access involved in training; it is a local, synthetic bootstrap for a
general-purpose "how similar does this sound" embedding, not a model trained on a music catalog. The
training script is at `tools/train_embedding_model.py` and is fully reproducible.

### Benchmarks

`scripts/benchmark_ordering.py` scores each engine with three metrics that do not depend on any
model's own internal cost function, so the comparison isn't self-graded: mean BPM jump, mean audio
energy jump, and mean genre-affinity distance between consecutive tracks. Lower is smoother. Run it
yourself with `python3 scripts/benchmark_ordering.py`.

Metadata only, using the bundled 10-track `tunnel/sample_playlist.json`, shuffled first:

| Engine                     | Tempo jump (BPM) | Energy jump | Genre distance |
| --------------------------- | ---------------: | ----------: | --------------: |
| source order (shuffled)    |             24.33 |       0.254 |            0.690 |
| metadata-flow-v0           |              9.00 |       0.108 |            0.591 |
| playlist-ml-v1             |             12.67 |       0.146 |            0.591 |

Audio-aware, using 14 procedurally generated synthetic tracks (there is no real audio in this repo),
shuffled first:

| Engine                     | Tempo jump (BPM) | Energy jump | Genre distance |
| --------------------------- | ---------------: | ----------: | --------------: |
| source order (shuffled)    |             21.54 |       0.020 |            0.586 |
| playlist-ml-v1 (no audio)  |             30.92 |       0.009 |            0.266 |
| audio-hybrid-v1            |             16.15 |       0.010 |            0.266 |
| audio-embedding-v1         |              9.77 |       0.011 |            0.320 |

Two honest caveats: these are small, synthetic benchmarks, not a study on real listening data, and
`playlist-ml-v1` can do worse than the shuffled input on tempo alone when it has no BPM or audio
signal to learn from and only genre/artist text to go on, exactly the gap `audio-hybrid-v1` and
`audio-embedding-v1` are meant to close. In this run, `audio-embedding-v1` produced the smoothest
tempo curve of any engine tested.

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

To also work on the Core ML embedding model:

```bash
pip install -e ".[dev,ml]"
python3 tools/train_embedding_model.py
```

Run tests:

```bash
python3 -m unittest -v
```

Lint and type-check:

```bash
ruff check .
mypy tunnel
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development
environment, coding conventions, and how to submit a pull request.

## License

Tunnel is licensed under the [MIT License](LICENSE).
