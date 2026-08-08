from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import sqrt
from re import findall

from .audio import AudioFeatures
from .embedding import EMBEDDING_DIMS
from .types import Track

VECTOR_DIMS = 48
DEFAULT_TRANSITION_WEIGHTS = (1.15, 1.35, 0.35, 0.25, 1.05, 0.25, 0.15)

LOW_ENERGY_WORDS = {
    "acoustic": 0.25,
    "ambient": 0.16,
    "ballad": 0.22,
    "chill": 0.22,
    "classical": 0.2,
    "downtempo": 0.2,
    "folk": 0.33,
    "indie folk": 0.3,
    "jazz": 0.38,
    "lofi": 0.24,
    "piano": 0.18,
    "singer songwriter": 0.28,
    "sleep": 0.08,
    "soul": 0.42,
}

HIGH_ENERGY_WORDS = {
    "afrobeats": 0.72,
    "dance": 0.82,
    "disco": 0.76,
    "drill": 0.74,
    "dubstep": 0.9,
    "edm": 0.88,
    "electronic": 0.76,
    "garage": 0.68,
    "hard rock": 0.78,
    "hip hop": 0.68,
    "house": 0.82,
    "hyperpop": 0.84,
    "metal": 0.92,
    "pop": 0.62,
    "punk": 0.82,
    "rap": 0.7,
    "reggaeton": 0.78,
    "rock": 0.7,
    "techno": 0.86,
    "trance": 0.84,
    "trap": 0.72,
}


@dataclass(frozen=True)
class TrackFeatures:
    tempo: float
    energy: float
    brightness: float
    dynamics: float
    era: float
    duration: float
    affinity: tuple[float, ...]


@dataclass(frozen=True)
class TransitionBreakdown:
    tempo: float
    energy: float
    brightness: float
    dynamics: float
    genre: float
    era: float
    duration: float
    repetition: float
    weights: tuple[float, float, float, float, float, float, float] = DEFAULT_TRANSITION_WEIGHTS

    @property
    def total(self) -> float:
        tempo_weight, energy_weight, brightness_weight, dynamics_weight, genre_weight, era_weight, duration_weight = self.weights
        return max(
            0.0,
            tempo_weight * self.tempo
            + energy_weight * self.energy
            + brightness_weight * self.brightness
            + dynamics_weight * self.dynamics
            + genre_weight * self.genre
            + era_weight * self.era
            + duration_weight * self.duration
            + self.repetition,
        )


class LocalFlowModel:
    """Local transition model used by the playlist orderer.

    Tunnel has exactly one ordering engine: audio-embedding-v1. It always runs
    Core ML locally through `tunnel.embedding`. Per-playlist transition weights
    are trained on top of it by `train_playlist_model`; that training step
    tunes the same model, it does not select a different one.
    """

    name = "audio-embedding-v1"

    def __init__(
        self,
        audio_features: dict[str, AudioFeatures] | None = None,
        embeddings: dict[str, tuple[float, ...]] | None = None,
        weights: tuple[float, float, float, float, float, float, float] = DEFAULT_TRANSITION_WEIGHTS,
    ) -> None:
        self.audio_features = audio_features or {}
        self.embeddings = embeddings or {}
        self.weights = weights

    @property
    def audio_feature_count(self) -> int:
        return len(self.audio_features)

    @property
    def embedding_count(self) -> int:
        return len(self.embeddings)

    def features_for(self, track: Track) -> TrackFeatures:
        audio = self.audio_features.get(track.id)
        tempo = _tempo_norm(audio.bpm if audio and audio.bpm else track.bpm)
        genre_energy = _genre_energy(track.genre)
        rating_energy = _rating_norm(track.rating, track.loved)
        duration = _duration_norm(track.duration)
        era = _era_norm(track.year)
        if audio:
            brightness = audio.brightness
            dynamics = audio.dynamics
            energy = _clamp(
                0.52 * audio.energy
                + 0.22 * tempo
                + 0.12 * genre_energy
                + 0.08 * dynamics
                + 0.06 * rating_energy
            )
        else:
            brightness = 0.5
            dynamics = 0.5
            energy = _clamp(0.52 * tempo + 0.35 * genre_energy + 0.08 * rating_energy + 0.05 * duration)
        return TrackFeatures(
            tempo=tempo,
            energy=energy,
            brightness=brightness,
            dynamics=dynamics,
            era=era,
            duration=duration,
            affinity=_affinity_vector(track, self.embeddings.get(track.id)),
        )

    def transition(self, left: Track, right: Track, left_features: TrackFeatures, right_features: TrackFeatures) -> TransitionBreakdown:
        genre_distance = _cosine_distance(left_features.affinity, right_features.affinity)
        repetition = 0.0
        if left.artist and right.artist and left.artist.casefold() == right.artist.casefold():
            repetition -= 0.06
        if left.album and right.album and left.album.casefold() == right.album.casefold():
            repetition -= 0.05
        if left.name.casefold() == right.name.casefold() and left.artist.casefold() == right.artist.casefold():
            repetition += 0.35

        return TransitionBreakdown(
            tempo=abs(left_features.tempo - right_features.tempo),
            energy=abs(left_features.energy - right_features.energy),
            brightness=abs(left_features.brightness - right_features.brightness),
            dynamics=abs(left_features.dynamics - right_features.dynamics),
            genre=genre_distance,
            era=abs(left_features.era - right_features.era),
            duration=abs(left_features.duration - right_features.duration),
            repetition=repetition,
            weights=self.weights,
        )

    def target_energy(self, position: int, total: int) -> float:
        if total <= 1:
            return 0.5
        progress = position / (total - 1)
        if progress <= 0.72:
            return 0.18 + 0.72 * _ease(progress / 0.72)
        release = _ease((progress - 0.72) / 0.28)
        return 0.9 - 0.17 * release

    def placement_cost(self, features: TrackFeatures, position: int, total: int) -> float:
        return 0.68 * abs(features.energy - self.target_energy(position, total))


def train_playlist_model(
    tracks: list[Track],
    audio_features: dict[str, AudioFeatures] | None = None,
    embeddings: dict[str, tuple[float, ...]] | None = None,
    epochs: int = 20,
    learning_rate: float = 0.08,
    margin: float = 0.08,
) -> LocalFlowModel:
    """Train a lightweight transition ranker from the source playlist order.

    The model learns which feature differences matter for this playlist by
    treating existing adjacent pairs as positive examples and deterministic
    non-adjacent pairs as negatives. It is intentionally small so it runs
    locally and ships without dependency downloads.
    """

    base = LocalFlowModel(audio_features=audio_features, embeddings=embeddings)
    if len(tracks) < 4:
        return base

    features = [base.features_for(track) for track in tracks]
    weights = list(DEFAULT_TRANSITION_WEIGHTS)
    examples = _training_examples(tracks, features, base)
    if not examples:
        return base

    for _ in range(max(1, epochs)):
        for positive, negative in examples:
            positive_score = _weighted_score(weights, positive)
            negative_score = _weighted_score(weights, negative)
            if positive_score + margin > negative_score:
                for index, (pos_value, neg_value) in enumerate(zip(positive, negative)):
                    weights[index] -= learning_rate * (pos_value - neg_value)
                    weights[index] = _clamp(weights[index], 0.05, 4.0)

    return LocalFlowModel(
        audio_features=audio_features,
        embeddings=embeddings,
        weights=tuple(weights),  # type: ignore[arg-type]
    )


def _training_examples(
    tracks: list[Track],
    features: list[TrackFeatures],
    model: LocalFlowModel,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    examples = []
    total = len(tracks)
    for left_index in range(total - 1):
        right_index = left_index + 1
        negative_index = (left_index * 7 + 3) % total
        if negative_index in {left_index, right_index}:
            negative_index = (negative_index + 2) % total
        if negative_index == left_index:
            continue
        positive = model.transition(
            tracks[left_index],
            tracks[right_index],
            features[left_index],
            features[right_index],
        )
        negative = model.transition(
            tracks[left_index],
            tracks[negative_index],
            features[left_index],
            features[negative_index],
        )
        examples.append((_transition_vector(positive), _transition_vector(negative)))
    return examples


def _transition_vector(breakdown: TransitionBreakdown) -> tuple[float, ...]:
    return (
        breakdown.tempo,
        breakdown.energy,
        breakdown.brightness,
        breakdown.dynamics,
        breakdown.genre,
        breakdown.era,
        breakdown.duration,
    )


def _weighted_score(weights: list[float], values: tuple[float, ...]) -> float:
    return sum(weight * value for weight, value in zip(weights, values))


def _tokenize(text: str) -> list[str]:
    return findall(r"[a-z0-9]+", text.casefold())


def _hash_index(token: str) -> int:
    digest = blake2b(token.encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, "big") % VECTOR_DIMS


def _affinity_vector(track: Track, embedding: tuple[float, ...] | None) -> tuple[float, ...]:
    values = [0.0] * VECTOR_DIMS
    weighted_sources = [
        (track.genre, 1.5),
        (track.artist, 1.0),
        (track.album_artist, 0.8),
        (track.album, 0.65),
    ]
    for source, weight in weighted_sources:
        for token in _tokenize(source):
            values[_hash_index(token)] += weight

    norm = sqrt(sum(value * value for value in values))
    text_vector = tuple(values) if norm == 0 else tuple(value / norm for value in values)
    embedding_vector = embedding if embedding is not None else (0.0,) * EMBEDDING_DIMS
    return text_vector + embedding_vector


def _cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.35
    similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return _clamp(1.0 - similarity)


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


def _era_norm(year: int | None) -> float:
    if year is None or year <= 0:
        return 0.5
    return _clamp((year - 1970) / 60)


def _rating_norm(rating: int | None, loved: bool) -> float:
    rating_value = 0.5 if rating is None else _clamp(rating / 100)
    if loved:
        rating_value = min(1.0, rating_value + 0.18)
    return rating_value


def _genre_energy(genre: str) -> float:
    normalized = " ".join(_tokenize(genre))
    if not normalized:
        return 0.52

    matches: list[float] = []
    for phrase, energy in LOW_ENERGY_WORDS.items():
        if phrase in normalized:
            matches.append(energy)
    for phrase, energy in HIGH_ENERGY_WORDS.items():
        if phrase in normalized:
            matches.append(energy)
    if not matches:
        return 0.52
    return sum(matches) / len(matches)


def _ease(value: float) -> float:
    value = _clamp(value)
    return value * value * (3 - 2 * value)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))
