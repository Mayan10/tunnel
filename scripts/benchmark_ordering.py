#!/usr/bin/env python3
"""Benchmarks Tunnel's ordering engines against a source (as-is) order.

Two benchmarks:

1. Metadata only, using the bundled sample playlist (`tunnel/sample_playlist.json`):
   compares the untrained baseline (metadata-flow-v0) against the locally
   trained ranker (playlist-ml-v1).
2. Audio-aware, using procedurally generated synthetic audio (there is no real
   audio in this repository): compares metadata-only, audio-hybrid-v1, and
   audio-embedding-v1 on the same synthetic playlist.

Every engine is scored with the same three model-independent metrics, so the
comparison is not biased by any one model's own internal cost function:

- mean tempo jump: average |BPM(i+1) - BPM(i)| between consecutive tracks
- mean energy jump: average |energy(i+1) - energy(i)|, using measured audio
  energy where available, otherwise a fixed genre-energy heuristic
- mean genre distance: average cosine distance between consecutive tracks'
  text-affinity vectors (genre/artist/album), ignoring audio and embeddings

Run with: python3 scripts/benchmark_ordering.py
Run with the ml extra installed for the audio-embedding-v1 row:
    pip install ".[ml]" && python3 scripts/benchmark_ordering.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import wave
from array import array
from importlib import resources
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tunnel.audio import analyze_audio_for_tracks  # noqa: E402
from tunnel.embedding import coreml_available, embed_tracks  # noqa: E402
from tunnel.model import LocalFlowModel, _genre_energy, train_playlist_model  # noqa: E402
from tunnel.ordering import order_tracks  # noqa: E402
from tunnel.types import Track  # noqa: E402

SYNTHETIC_TRACKS = [
    # name, genre, freq_hz, beat_period_samples (smaller = faster)
    ("Quiet Room", "Ambient", 220, 9000),
    ("Slow Fog", "Ambient", 200, 9400),
    ("Open Field", "Ambient", 235, 8800),
    ("Nylon Morning", "Acoustic", 260, 7000),
    ("Porch Light", "Acoustic", 250, 7200),
    ("Warm Strings", "Classical", 280, 6600),
    ("Evening Set", "Classical", 300, 6400),
    ("Neon Pop", "Pop", 340, 4200),
    ("Radio Friend", "Pop", 360, 4000),
    ("Skyline Drive", "Dance", 400, 2500),
    ("Club Signal", "Dance", 420, 2400),
    ("Voltage", "Dance", 410, 2450),
    ("Heavy Gear", "Rock", 300, 3800),
    ("Static Wall", "Rock", 320, 3600),
]


def make_synthetic_wav(freq: float, beat_period: int, seconds: int = 6, sample_rate: int = 11025) -> Path:
    samples = array("h")
    for i in range(sample_rate * seconds):
        beat = 1.0 if (i % beat_period) < (beat_period // 6) else 0.3
        tone = math.sin(2 * math.pi * freq * i / sample_rate)
        samples.append(int(beat * tone * 14000))
    handle = tempfile.NamedTemporaryFile(prefix="tunnel-bench-", suffix=".wav", delete=False)
    path = Path(handle.name)
    handle.close()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return path


def evaluate(tracks: list[Track], audio_features: dict) -> dict[str, float]:
    tempo_jumps = []
    energy_jumps = []
    genre_distances = []

    def track_bpm(track: Track) -> float | None:
        audio = audio_features.get(track.id)
        if audio and audio.bpm:
            return audio.bpm
        return track.bpm

    def track_energy(track: Track) -> float:
        audio = audio_features.get(track.id)
        if audio:
            return audio.energy
        return _genre_energy(track.genre)

    model = LocalFlowModel()
    features = {track.id: model.features_for(track) for track in tracks}

    for left, right in zip(tracks, tracks[1:]):
        left_bpm, right_bpm = track_bpm(left), track_bpm(right)
        if left_bpm and right_bpm:
            tempo_jumps.append(abs(left_bpm - right_bpm))
        energy_jumps.append(abs(track_energy(left) - track_energy(right)))
        left_affinity = features[left.id].affinity[:48]
        right_affinity = features[right.id].affinity[:48]
        dot = sum(a * b for a, b in zip(left_affinity, right_affinity, strict=True))
        genre_distances.append(1.0 - dot)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "tempo_jump": mean(tempo_jumps),
        "energy_jump": mean(energy_jumps),
        "genre_distance": mean(genre_distances),
    }


def print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    print(f"{'Engine':<22}{'Tempo jump (BPM)':>18}{'Energy jump':>14}{'Genre distance':>16}")
    print("-" * 70)
    for name, metrics in rows:
        print(
            f"{name:<22}{metrics['tempo_jump']:>18.2f}"
            f"{metrics['energy_jump']:>14.3f}{metrics['genre_distance']:>16.3f}"
        )


def benchmark_metadata_only() -> None:
    print("\n== Benchmark 1: metadata only (bundled sample_playlist.json) ==\n")
    sample_text = resources.files("tunnel").joinpath("sample_playlist.json").read_text(encoding="utf-8")
    payload = json.loads(sample_text)
    tracks = [Track.from_dict(item, index + 1) for index, item in enumerate(payload.get("tracks", []))]

    rng = random.Random(7)
    shuffled = tracks[:]
    rng.shuffle(shuffled)

    rows = [
        ("source order (shuffled)", evaluate(shuffled, {})),
        ("metadata-flow-v0", evaluate(order_tracks(shuffled, model=LocalFlowModel()).tracks, {})),
        ("playlist-ml-v1", evaluate(order_tracks(shuffled, model=train_playlist_model(shuffled)).tracks, {})),
    ]
    print_table(rows)


def benchmark_audio_aware() -> None:
    print("\n== Benchmark 2: synthetic audio (procedurally generated, not real music) ==\n")
    tracks = []
    paths = []
    for index, (name, genre, freq, beat_period) in enumerate(SYNTHETIC_TRACKS):
        path = make_synthetic_wav(freq, beat_period)
        paths.append(path)
        tracks.append(
            Track(
                id=f"bench-{index}",
                name=name,
                genre=genre,
                artist="Benchmark",
                location=f"file://{path}",
                source_index=index + 1,
            )
        )

    try:
        rng = random.Random(11)
        shuffled = tracks[:]
        rng.shuffle(shuffled)

        audio_features = analyze_audio_for_tracks(shuffled)
        embeddings = embed_tracks(shuffled, audio_features)

        rows = [
            ("source order (shuffled)", evaluate(shuffled, audio_features)),
            (
                "playlist-ml-v1 (no audio)",
                evaluate(order_tracks(shuffled, model=train_playlist_model(shuffled)).tracks, audio_features),
            ),
            (
                "audio-hybrid-v1",
                evaluate(
                    order_tracks(shuffled, model=train_playlist_model(shuffled, audio_features=audio_features)).tracks,
                    audio_features,
                ),
            ),
        ]
        if embeddings:
            rows.append(
                (
                    "audio-embedding-v1",
                    evaluate(
                        order_tracks(
                            shuffled,
                            model=train_playlist_model(shuffled, audio_features=audio_features, embeddings=embeddings),
                        ).tracks,
                        audio_features,
                    ),
                )
            )
        else:
            print("(coremltools not installed; skipping audio-embedding-v1. Install with: pip install \".[ml]\")\n")

        print_table(rows)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def main() -> int:
    print(f"coremltools available: {coreml_available()}")
    benchmark_metadata_only()
    benchmark_audio_aware()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
