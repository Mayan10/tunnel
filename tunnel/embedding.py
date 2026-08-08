from __future__ import annotations

from importlib import resources

import coremltools as ct
import numpy as np

from .audio import AudioFeatures
from .types import Track

EMBEDDING_DIMS = 8

_MODEL_RESOURCE = "models/audio_embedding_v1.mlmodel"
_MODEL_INPUT_NAME = "audio_features"
_MODEL_OUTPUT_NAME = "embedding"

_model_cache: ct.models.MLModel | None = None


class EmbeddingError(RuntimeError):
    """Raised when the bundled Core ML model cannot be loaded or run."""


def embed_tracks(
    tracks: list[Track],
    audio_features: dict[str, AudioFeatures],
) -> dict[str, tuple[float, ...]]:
    """Runs audio-embedding-v1 over every track with local audio features.

    Embeds the whole playlist in a single Core ML call (the model has a
    flexible batch dimension) rather than one call per track, since per-call
    dispatch overhead dominates at batch size 1 on the Neural Engine.
    """
    model = _load_model()

    usable_tracks = [track for track in tracks if track.id in audio_features]
    if not usable_tracks:
        return {}

    vectors = [_input_vector(track, audio_features[track.id]) for track in usable_tracks]
    raw = _predict_batch(model, vectors)

    embeddings: dict[str, tuple[float, ...]] = {}
    for track, row in zip(usable_tracks, raw, strict=True):
        norm = float(np.linalg.norm(row))
        embeddings[track.id] = tuple(0.0 for _ in row) if norm <= 1e-8 else tuple(float(v / norm) for v in row)
    return embeddings


def _load_model() -> ct.models.MLModel:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    try:
        model_path = resources.files("tunnel").joinpath(_MODEL_RESOURCE)
        with resources.as_file(model_path) as path:
            # compute_units=ALL lets Core ML dispatch across the Neural Engine,
            # GPU, and CPU, whichever it judges fastest for this model and
            # hardware. This is the default, but is set explicitly so a future
            # coremltools version changing its default doesn't silently
            # regress performance.
            _model_cache = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
    except Exception as exc:
        raise EmbeddingError(f"Could not load the Core ML embedding model: {exc}") from exc
    return _model_cache


def _predict_batch(model: ct.models.MLModel, vectors: list[list[float]]) -> np.ndarray:
    try:
        input_array = np.asarray(vectors, dtype=np.float32)
        result = model.predict({_MODEL_INPUT_NAME: input_array})
        return np.asarray(result[_MODEL_OUTPUT_NAME])
    except Exception as exc:
        raise EmbeddingError(f"Core ML prediction failed: {exc}") from exc


def _input_vector(track: Track, audio: AudioFeatures) -> list[float]:
    tempo = _tempo_norm(audio.bpm if audio.bpm else track.bpm)
    duration = _duration_norm(track.duration)
    return [tempo, audio.energy, audio.brightness, audio.dynamics, duration, *audio.bands]


def _tempo_norm(bpm: float | None) -> float:
    if bpm is None or bpm <= 0:
        return 0.48
    tempo = float(bpm)
    while tempo > 180:
        tempo /= 2
    while tempo < 70:
        tempo *= 2
    return _clamp((tempo - 70) / 90)


def _duration_norm(duration: float | None) -> float:
    if duration is None or duration <= 0:
        return 0.5
    return _clamp((duration - 120) / 300)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))
