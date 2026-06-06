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
- replace the original playlist order after confirmation
- save the ordered result as JSON

Power-user commands:

```bash
tunnel list
tunnel order "My Playlist" --create-playlist "My Playlist - Flow"
tunnel order "My Playlist" --replace-original
tunnel export "My Playlist" --out exports/my-playlist.json
```

Tunnel does not silently drop tracks. Before writing to Apple Music it stages the full ordered playlist and verifies the track count. Replacing the original playlist also creates a backup playlist first.

## Model

Tunnel currently uses two on-device models:

- `audio-hybrid-v1`: runs when Apple Music exposes local audio file paths. It decodes audio with macOS `afconvert`, estimates tempo, energy, brightness, and dynamics, caches the features, and combines them with metadata.
- `metadata-flow-v0`: fallback when tracks are cloud-only or Music does not expose local file paths.

For best results, download the playlist in Music before running Tunnel. If the output says `0 local files`, Tunnel cannot analyze audio for those songs.

The next production ML milestone is `audio-embedding-v1`: a packaged Core ML model using Apple hardware acceleration for neural audio embeddings.

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
- Tag: `v0.1.0`
- Title: `Tunnel v0.1.0`
- Publish the release.

The included GitHub Actions release workflow builds `dist/tunnel.pyz` and attaches it to the release automatically.

Manual fallback:

```bash
python3 scripts/build_zipapp.py
```

Then upload `dist/tunnel.pyz` to the GitHub release assets.
