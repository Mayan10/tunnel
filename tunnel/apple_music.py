from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .types import Track


class AppleMusicError(RuntimeError):
    """Raised when the macOS Music app cannot provide playlist data."""


@dataclass(frozen=True)
class MusicPlaylist:
    name: str
    track_count: int
    special_kind: str = ""


LIST_PLAYLISTS_JXA = r"""
ObjC.import('stdlib');
const Music = Application('/System/Applications/Music.app');

function safe(fn, fallback) {
  try {
    const value = fn();
    return value === null || value === undefined ? fallback : value;
  } catch (error) {
    return fallback;
  }
}

const playlists = Music.playlists().map((playlist) => {
  return {
    name: String(safe(() => playlist.name(), "")),
    track_count: Number(safe(() => playlist.tracks().length, 0)),
    special_kind: String(safe(() => playlist.specialKind(), "")),
  };
}).filter((playlist) => playlist.name.length > 0);

JSON.stringify(playlists);
"""


EXPORT_PLAYLIST_JXA = r"""
ObjC.import('stdlib');
const Music = Application('/System/Applications/Music.app');
const playlistName = $.getenv('TUNNEL_PLAYLIST');

function safe(fn, fallback) {
  try {
    const value = fn();
    return value === null || value === undefined ? fallback : value;
  } catch (error) {
    return fallback;
  }
}

function text(fn) {
  const value = safe(fn, "");
  return value === null || value === undefined ? "" : String(value);
}

function number(fn) {
  const value = safe(fn, null);
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

if (!playlistName) {
  throw new Error("TUNNEL_PLAYLIST is required.");
}

const matches = Music.playlists.whose({ name: playlistName })();
if (matches.length === 0) {
  throw new Error(`Playlist not found: ${playlistName}`);
}

const playlist = matches[0];
const tracks = playlist.tracks().map((track, index) => {
  return {
    index: index + 1,
    id: text(() => track.persistentID()),
    database_id: number(() => track.databaseID()),
    name: text(() => track.name()) || "Unknown Title",
    artist: text(() => track.artist()),
    album: text(() => track.album()),
    album_artist: text(() => track.albumArtist()),
    genre: text(() => track.genre()),
    duration: number(() => track.duration()),
    bpm: number(() => track.bpm()),
    year: number(() => track.year()),
    rating: number(() => track.rating()),
    played_count: number(() => track.playedCount()),
    loved: Boolean(safe(() => track.loved(), false)),
    location: text(() => track.location()),
  };
});

JSON.stringify({
  source: "apple-music",
  playlist: playlistName,
  exported_at: new Date().toISOString(),
  tracks,
});
"""


CREATE_PLAYLIST_APPLESCRIPT = r"""
on run argv
  set sourceName to item 1 of argv
  set targetName to item 2 of argv
  set tempName to item 3 of argv
  set orderedIndexes to items 4 thru -1 of argv

  tell application "/System/Applications/Music.app"
    if exists user playlist targetName then
      error "A playlist named \"" & targetName & "\" already exists."
    end if
    if exists user playlist tempName then
      delete user playlist tempName
    end if

    set sourcePlaylist to first playlist whose name is sourceName
    set sourceCount to count tracks of sourcePlaylist

    if sourceCount is not (count orderedIndexes) then
      error "Refusing to create an incomplete playlist. Source has " & sourceCount & " tracks, but the computed order has " & (count orderedIndexes) & "."
    end if

    set tempPlaylist to make new user playlist with properties {name:tempName}

    try
      repeat with sourceIndex in orderedIndexes
        set sourcePosition to sourceIndex as integer
        if sourcePosition < 1 or sourcePosition > sourceCount then
          error "Ordered track index " & sourcePosition & " is outside the source playlist."
        end if
        duplicate track sourcePosition of sourcePlaylist to tempPlaylist
      end repeat

      if (count tracks of tempPlaylist) is not sourceCount then
        error "Copy count mismatch. Source has " & sourceCount & " tracks, but the staged playlist has " & (count tracks of tempPlaylist) & "."
      end if

      set name of tempPlaylist to targetName
    on error errorMessage
      try
        delete tempPlaylist
      end try
      error errorMessage
    end try
  end tell
end run
"""


COPY_PLAYLIST_APPLESCRIPT = r"""
on run argv
  set sourceName to item 1 of argv
  set targetName to item 2 of argv
  set tempName to item 3 of argv

  tell application "/System/Applications/Music.app"
    if exists user playlist targetName then
      error "A playlist named \"" & targetName & "\" already exists."
    end if
    if exists user playlist tempName then
      delete user playlist tempName
    end if

    set sourcePlaylist to first playlist whose name is sourceName
    set sourceCount to count tracks of sourcePlaylist
    set tempPlaylist to make new user playlist with properties {name:tempName}

    try
      repeat with sourceTrack in tracks of sourcePlaylist
        duplicate sourceTrack to tempPlaylist
      end repeat

      if (count tracks of tempPlaylist) is not sourceCount then
        error "Copy count mismatch. Source has " & sourceCount & " tracks, but the staged playlist has " & (count tracks of tempPlaylist) & "."
      end if

      set name of tempPlaylist to targetName
    on error errorMessage
      try
        delete tempPlaylist
      end try
      error errorMessage
    end try
  end tell
end run
"""


def list_playlists() -> list[MusicPlaylist]:
    data = _run_jxa(LIST_PLAYLISTS_JXA)
    try:
        raw_playlists = json.loads(data)
    except json.JSONDecodeError as exc:
        raise AppleMusicError(f"Music returned invalid playlist data: {exc}") from exc
    return [
        MusicPlaylist(
            name=str(item.get("name", "")),
            track_count=int(item.get("track_count") or 0),
            special_kind=str(item.get("special_kind", "")),
        )
        for item in raw_playlists
    ]


def export_playlist(playlist_name: str) -> dict[str, Any]:
    data = _run_jxa(EXPORT_PLAYLIST_JXA, {"TUNNEL_PLAYLIST": playlist_name})
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise AppleMusicError(f"Music returned invalid track data: {exc}") from exc


def export_tracks(playlist_name: str) -> list[Track]:
    payload = export_playlist(playlist_name)
    tracks = payload.get("tracks", [])
    if not isinstance(tracks, list):
        raise AppleMusicError("Music returned a playlist without a tracks array.")
    return [Track.from_dict(track, index + 1) for index, track in enumerate(tracks)]


def create_playlist_from_order(source_playlist: str, target_playlist: str, ordered_tracks: list[Track]) -> None:
    ordered_indexes = _ordered_source_indexes(ordered_tracks)
    temp_playlist = f"{target_playlist} - Tunnel Staging {uuid4().hex[:8]}"

    process = subprocess.run(
        ["osascript", "-e", CREATE_PLAYLIST_APPLESCRIPT, source_playlist, target_playlist, temp_playlist, *ordered_indexes],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise AppleMusicError(_friendly_music_error(process.stderr.strip() or process.stdout.strip()))


def copy_playlist(source_playlist: str, target_playlist: str) -> None:
    temp_playlist = f"{target_playlist} - Tunnel Staging {uuid4().hex[:8]}"
    process = subprocess.run(
        ["osascript", "-e", COPY_PLAYLIST_APPLESCRIPT, source_playlist, target_playlist, temp_playlist],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise AppleMusicError(_friendly_music_error(process.stderr.strip() or process.stdout.strip()))


def _ordered_source_indexes(ordered_tracks: list[Track]) -> list[str]:
    indexes = []
    for position, track in enumerate(ordered_tracks, 1):
        if track.source_index is None:
            raise AppleMusicError(
                f"Track {position} ({track.display_name}) has no source playlist index. "
                "Re-export the playlist from Apple Music and try again."
            )
        indexes.append(str(track.source_index))
    if not indexes:
        raise AppleMusicError("No tracks are available to write to Music.")
    expected = list(range(1, len(ordered_tracks) + 1))
    actual = sorted(int(index) for index in indexes)
    if actual != expected:
        raise AppleMusicError(
            "Refusing to write a partial playlist. The computed order does not contain every "
            "source playlist position exactly once."
        )
    return indexes


def _run_jxa(script: str, extra_env: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    process = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if process.returncode != 0:
        message = _friendly_music_error(process.stderr.strip() or process.stdout.strip())
        raise AppleMusicError(message)
    return process.stdout.strip()


def _friendly_music_error(raw: str) -> str:
    if not raw:
        return "Unknown Music app error."
    noisy_markers = [
        "Application can't be found",
        "Parameter is missing",
        "Connection Invalid",
        "Connection invalid",
        "-10827",
        "-2700",
        "-1701",
    ]
    if any(marker in raw for marker in noisy_markers):
        return (
            "Could not read the macOS Music library. Open Music once, then allow this "
            "terminal app to control Music in System Settings > Privacy & Security > "
            f"Automation. Raw error: {raw}"
        )
    return raw
