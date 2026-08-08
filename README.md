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

Tunnel currently uses local ordering engines:

- `playlist-ml-v1`: trains a lightweight transition ranker from the source playlist order, then uses the learned weights to build the new order.
- `playlist-audio-ml-v1`: same trainer, plus local audio features when Apple Music exposes downloaded file paths.
- `audio-hybrid-v1`: local audio feature extractor used by the trainer. It decodes audio with macOS `afconvert`, estimates tempo, energy, brightness, and dynamics, and caches the features.
- `metadata-flow-v0`: fallback when tracks are cloud-only or Music does not expose local file paths.

For best results, download the playlist in Music before running Tunnel. If the output says `0 local files`, Tunnel cannot analyze audio for those songs.

The current ML trainer runs locally with no dependency downloads. The next production ML milestone is `audio-embedding-v1`: a packaged Core ML model using Apple hardware acceleration for neural audio embeddings. That model is not included in this release yet.

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
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
