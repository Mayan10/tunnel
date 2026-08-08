from __future__ import annotations

from importlib import resources

from .audio import AudioFeatures
from .types import Track

EMBEDDING_DIMS = 8

_MODEL_RESOURCE = "models/audio_embedding_v1.mlmodel"
_MODEL_INPUT_NAME = "audio_features"
_MODEL_OUTPUT_NAME = "embedding"

_model_loaded = False
_model_cache: object | None = None


def coreml_available() -> bool:
    """Whether the optional Core ML runtime is installed (`pip install tunnel[ml]`)."""
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def embed_tracks(
    tracks: list[Track],
    audio_features: dict[str, AudioFeatures],
) -> dict[str, tuple[float, ...]]:
    """Runs audio-embedding-v1 over every track with local audio features.

    Returns an empty dict if the optional `coremltools` dependency is not
    installed, or if the bundled model fails to load. Callers should treat
    this as an optional enhancement layered on top of `audio_features`, not
    a required input.
    """
    model = _load_model()
    if model is None:
        return {}

    embeddings: dict[str, tuple[float, ...]] = {}
    for track in tracks:
        audio = audio_features.get(track.id)
        if audio is None:
            continue
        vector = _input_vector(track, audio)
        embedding = _predict(model, vector)
        if embedding is not None:
            embeddings[track.id] = embedding
    return embeddings


def _load_model() -> object | None:
    global _model_loaded, _model_cache
    if _model_loaded:
        return _model_cache
    _model_loaded = True

    try:
        import coremltools as ct
    except ImportError:
        return None

    try:
        model_path = resources.files("tunnel").joinpath(_MODEL_RESOURCE)
        with resources.as_file(model_path) as path:
            _model_cache = ct.models.MLModel(str(path))
    except Exception:
        # Core ML's Python bridge can raise a wide range of native/runtime
        # errors depending on the OS version and installed toolchain. Any
        # failure here should just disable the embedding layer, not crash
        # the ordering pipeline.
        _model_cache = None
    return _model_cache


def _predict(model: object, vector: list[float]) -> tuple[float, ...] | None:
    try:
        import numpy as np
    except ImportError:
        return None

    try:
        input_array = np.asarray([vector], dtype=np.float32)
        result = model.predict({_MODEL_INPUT_NAME: input_array})  # type: ignore[attr-defined]
        raw = np.asarray(result[_MODEL_OUTPUT_NAME]).reshape(-1)
    except Exception:
        return None

    norm = float(np.linalg.norm(raw))
    if norm <= 1e-8:
        return None
    return tuple(float(value / norm) for value in raw)


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
