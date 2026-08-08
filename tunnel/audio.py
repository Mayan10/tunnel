from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .types import Track


@dataclass(frozen=True)
class AudioFeatures:
    bpm: float | None
    energy: float
    brightness: float
    dynamics: float
    analyzed_seconds: float


ProgressCallback = Callable[[int, int, Track], None]


def analyze_audio_for_tracks(
    tracks: list[Track],
    progress: ProgressCallback | None = None,
    max_seconds: int = 120,
) -> dict[str, AudioFeatures]:
    local_tracks = [(track, _location_to_path(track.location)) for track in tracks if track.location]
    local_tracks = [(track, path) for track, path in local_tracks if path and path.exists()]
    if not local_tracks:
        return {}

    cache = _AudioFeatureCache.load()
    features: dict[str, AudioFeatures] = {}
    total = len(local_tracks)

    for index, (track, path) in enumerate(local_tracks, 1):
        if progress:
            progress(index, total, track)
        key = _cache_key(path)
        cached = cache.get(key)
        if cached is None:
            try:
                cached = analyze_audio_file(path, max_seconds=max_seconds)
            except (OSError, subprocess.SubprocessError, wave.Error, ValueError):
                continue
            cache.set(key, cached)
        features[track.id] = cached

    cache.save()
    return features


def analyze_audio_file(path: Path, max_seconds: int = 120) -> AudioFeatures:
    wav_path, cleanup_path = _as_wav(path)
    try:
        samples, sample_rate, analyzed_seconds = _read_wav_mono(wav_path, max_seconds)
    finally:
        if cleanup_path:
            cleanup_path.unlink(missing_ok=True)

    if len(samples) < sample_rate:
        raise ValueError("Audio file is too short to analyze.")

    frame_size = 1024
    hop_size = 512
    energies = _frame_energies(samples, frame_size, hop_size)
    mean_rms = sum(energies) / len(energies) if energies else 0.0
    energy = _clamp((mean_rms - 0.015) / 0.18)
    dynamics = _dynamics(energies)
    brightness = _brightness(samples)
    bpm = _estimate_bpm(energies, sample_rate / hop_size)

    return AudioFeatures(
        bpm=bpm,
        energy=energy,
        brightness=brightness,
        dynamics=dynamics,
        analyzed_seconds=analyzed_seconds,
    )


class _AudioFeatureCache:
    version = 1

    def __init__(self, path: Path, values: dict[str, AudioFeatures]) -> None:
        self.path = path
        self.values = values

    @classmethod
    def load(cls) -> _AudioFeatureCache:
        path = Path.home() / ".cache" / "tunnel" / "audio-features-v1.json"
        if not path.exists():
            return cls(path, {})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(path, {})
        if payload.get("version") != cls.version:
            return cls(path, {})
        values = {}
        for key, item in payload.get("features", {}).items():
            try:
                values[key] = AudioFeatures(
                    bpm=item.get("bpm"),
                    energy=float(item["energy"]),
                    brightness=float(item["brightness"]),
                    dynamics=float(item["dynamics"]),
                    analyzed_seconds=float(item["analyzed_seconds"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
        return cls(path, values)

    def get(self, key: str) -> AudioFeatures | None:
        return self.values.get(key)

    def set(self, key: str, features: AudioFeatures) -> None:
        self.values[key] = features

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "features": {key: asdict(value) for key, value in self.values.items()},
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _as_wav(path: Path) -> tuple[Path, Path | None]:
    if path.suffix.casefold() in {".wav", ".wave"}:
        return path, None
    afconvert = shutil.which("afconvert")
    if afconvert is None:
        raise OSError("afconvert is required to analyze non-WAV audio on macOS.")

    handle = tempfile.NamedTemporaryFile(prefix="tunnel-", suffix=".wav", delete=False)
    handle.close()
    wav_path = Path(handle.name)
    command = [afconvert, "-f", "WAVE", "-d", "LEI16@11025", "-c", "1", str(path), str(wav_path)]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        wav_path.unlink(missing_ok=True)
        raise subprocess.SubprocessError(process.stderr.strip() or process.stdout.strip())
    return wav_path, wav_path


def _read_wav_mono(path: Path, max_seconds: int) -> tuple[list[float], int, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames_to_read = min(wav.getnframes(), sample_rate * max_seconds)
        raw = wav.readframes(frames_to_read)

    if sample_width == 1:
        values = [(byte - 128) / 128.0 for byte in raw]
    elif sample_width == 2:
        ints = array("h")
        ints.frombytes(raw)
        if sys.byteorder != "little":
            ints.byteswap()
        values = [sample / 32768.0 for sample in ints]
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        mono = []
        for index in range(0, len(values) - channels + 1, channels):
            mono.append(sum(values[index : index + channels]) / channels)
        values = mono

    return values, sample_rate, len(values) / sample_rate


def _frame_energies(samples: list[float], frame_size: int, hop_size: int) -> list[float]:
    energies: list[float] = []
    if len(samples) < frame_size:
        return [math.sqrt(sum(sample * sample for sample in samples) / len(samples))]
    for start in range(0, len(samples) - frame_size, hop_size):
        frame = samples[start : start + frame_size]
        rms = math.sqrt(sum(sample * sample for sample in frame) / frame_size)
        energies.append(rms)
    return energies


def _estimate_bpm(energies: list[float], envelope_rate: float) -> float | None:
    if len(energies) < 16:
        return None
    onsets = [0.0]
    for previous, current in zip(energies, energies[1:]):
        onsets.append(max(0.0, current - previous))

    onset_total = sum(onsets)
    if onset_total <= 1e-6:
        return None
    onsets = [value / onset_total for value in onsets]

    best_bpm = None
    best_score = 0.0
    for bpm in range(70, 181):
        lag = round((60.0 / bpm) * envelope_rate)
        if lag <= 0 or lag >= len(onsets):
            continue
        score = sum(onsets[index] * onsets[index - lag] for index in range(lag, len(onsets)))
        if score > best_score:
            best_score = score
            best_bpm = float(bpm)

    if best_score <= 1e-6:
        return None
    return best_bpm


def _brightness(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.5
    step = max(1, len(samples) // 250_000)
    previous = samples[0]
    crossings = 0
    checked = 0
    diff_energy = 0.0
    signal_energy = 0.0
    for sample in samples[step::step]:
        if (sample >= 0) != (previous >= 0):
            crossings += 1
        difference = sample - previous
        diff_energy += difference * difference
        signal_energy += sample * sample
        previous = sample
        checked += 1
    zcr = crossings / checked if checked else 0.0
    roughness = math.sqrt(diff_energy / max(signal_energy, 1e-12)) if checked else 0.0
    return _clamp(0.55 * ((zcr - 0.02) / 0.18) + 0.45 * (roughness / 2.5))


def _dynamics(energies: list[float]) -> float:
    if not energies:
        return 0.5
    mean = sum(energies) / len(energies)
    if mean <= 1e-9:
        return 0.0
    variance = sum((value - mean) ** 2 for value in energies) / len(energies)
    return _clamp(math.sqrt(variance) / (mean * 1.8))


def _cache_key(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"


def _location_to_path(location: str) -> Path:
    if location.startswith("file://"):
        parsed = urlparse(location)
        return Path(unquote(parsed.path))
    return Path(location)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))
