from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def _positive_float(value: Any) -> float | None:
    number = _optional_float(value)
    if number is None or number <= 0:
        return None
    return number


def _positive_int(value: Any) -> int | None:
    number = _positive_float(value)
    if number is None:
        return None
    return int(number)


@dataclass(frozen=True)
class Track:
    """A normalized Music track record used by the local flow model."""

    id: str
    name: str
    artist: str = ""
    album: str = ""
    album_artist: str = ""
    genre: str = ""
    duration: float | None = None
    bpm: float | None = None
    year: int | None = None
    rating: int | None = None
    played_count: int | None = None
    loved: bool = False
    location: str = ""
    source_index: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_index: int | None = None) -> Track:
        name = _clean_text(data.get("name")) or "Unknown Title"
        artist = _clean_text(data.get("artist"))
        album = _clean_text(data.get("album"))
        genre = _clean_text(data.get("genre"))
        source_index = _optional_int(_first_present(data, "source_index", "index"))
        if source_index is None:
            source_index = fallback_index

        raw_id = _clean_text(
            data.get("id")
            or data.get("persistent_id")
            or data.get("persistentID")
            or data.get("database_id")
            or data.get("databaseID")
        )
        if not raw_id:
            stable = "|".join([name.lower(), artist.lower(), album.lower(), str(source_index or "")])
            raw_id = sha1(stable.encode("utf-8")).hexdigest()[:16]

        return cls(
            id=raw_id,
            name=name,
            artist=artist,
            album=album,
            album_artist=_clean_text(_first_present(data, "album_artist", "albumArtist")),
            genre=genre,
            duration=_positive_float(data.get("duration")),
            bpm=_positive_float(data.get("bpm")),
            year=_positive_int(data.get("year")),
            rating=_optional_int(data.get("rating")),
            played_count=_optional_int(_first_present(data, "played_count", "playedCount")),
            loved=bool(_first_present(data, "loved", "favorited") or False),
            location=_clean_text(data.get("location")),
            source_index=source_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "artist": self.artist,
            "album": self.album,
            "album_artist": self.album_artist,
            "genre": self.genre,
            "duration": self.duration,
            "bpm": self.bpm,
            "year": self.year,
            "rating": self.rating,
            "played_count": self.played_count,
            "loved": self.loved,
            "location": self.location,
            "source_index": self.source_index,
        }

    @property
    def display_name(self) -> str:
        if self.artist:
            return f"{self.artist} - {self.name}"
        return self.name
