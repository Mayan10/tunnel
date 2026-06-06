from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .types import Track


def write_playlist_snapshot(playlist_name: str, tracks: list[Track]) -> Path:
    snapshot_dir = Path.home() / ".tunnel" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}-{_safe_filename(playlist_name)}.json"
    path = snapshot_dir / filename
    payload = {
        "playlist": playlist_name,
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "track_count": len(tracks),
        "tracks": [track.to_dict() for track in tracks],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe.casefold() or "playlist"
