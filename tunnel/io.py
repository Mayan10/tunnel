from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from .ordering import OrderedPlaylist
from .types import Track


def load_tracks_json(path: Path) -> tuple[str, list[Track]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return path.stem, [Track.from_dict(item, index + 1) for index, item in enumerate(payload)]
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object or array.")
    raw_tracks = payload.get("tracks")
    if not isinstance(raw_tracks, list):
        raise ValueError("Expected a 'tracks' array in the JSON file.")
    playlist = str(payload.get("playlist") or path.stem)
    return playlist, [Track.from_dict(item, index + 1) for index, item in enumerate(raw_tracks)]


def write_order_json(path: Path, playlist_name: str, ordered: OrderedPlaylist) -> None:
    payload = {
        "playlist": playlist_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": ordered.model_name,
        "score": round(ordered.score, 6),
        "tracks": [track.to_dict() for track in ordered.tracks],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def write_m3u(path: Path, tracks: list[Track]) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as file:
        file.write("#EXTM3U\n")
        for track in tracks:
            location = _location_to_path(track.location)
            if not location:
                continue
            duration = int(track.duration or -1)
            file.write(f"#EXTINF:{duration},{track.display_name}\n")
            file.write(f"{location}\n")
            written += 1
    return written


def _location_to_path(location: str) -> str:
    if not location:
        return ""
    if location.startswith("file://"):
        parsed = urlparse(location)
        return unquote(parsed.path)
    return location
