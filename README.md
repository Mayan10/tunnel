# Tunnel

Tunnel is a terminal app for ordering Apple Music playlists into smoother listening flows.

## Requirements

- macOS with the Music app
- Python 3.11 or newer
- Apple Music automation permission for Terminal or your terminal app
- Downloaded music files for audio analysis. Cloud-only Apple Music tracks can still be ordered, but Tunnel cannot analyze their waveform.

## Install From GitHub

Download `tunnel.pyz` from the latest GitHub release.

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

Run tests:

```bash
python3 -m unittest
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

## Publish On GitHub

1. Create a new empty repository on GitHub named `tunnel`.

2. Initialize and push this repo:

```bash
cd /Users/mayan/mzk
git init
git add .
git commit -m "Initial Tunnel release"
git branch -M main
git remote add origin git@github.com:YOUR-USERNAME/tunnel.git
git push -u origin main
```

3. Create a release on GitHub:

- Go to your repo on GitHub.
- Open Releases.
- Click Draft a new release.
- Tag: `v0.2.0`
- Title: `Tunnel v0.2.0`
- Publish the release.

The included GitHub Actions release workflow builds `dist/tunnel.pyz` and attaches it to the release automatically.

Manual fallback:

```bash
python3 scripts/build_zipapp.py
```

Then upload `dist/tunnel.pyz` to the GitHub release assets.
