from __future__ import annotations

from pathlib import Path

from .apple_music import (
    AppleMusicError,
    create_playlist_from_order,
    export_tracks,
    list_playlists,
    rebuild_playlist_in_order,
)
from .audio import analyze_audio_for_tracks
from .io import write_order_json
from .model import LocalFlowModel
from .ordering import order_tracks
from .types import Track


def run_app() -> int:
    print()
    print("Tunnel")
    print()

    playlists = [playlist for playlist in list_playlists() if playlist.track_count > 0]
    if not playlists:
        print("No playlists with tracks were found in Apple Music.")
        return 1

    playlist_name = _choose_playlist(playlists)
    if not playlist_name:
        print("Cancelled.")
        return 0

    print()
    print(f"Reading \"{playlist_name}\" from Apple Music...")
    tracks = export_tracks(playlist_name)
    if not tracks:
        print("That playlist has no tracks.")
        return 1

    model = _build_model(tracks)
    print(f"Ordering {len(tracks)} tracks with {model.name}...")
    ordered = order_tracks(tracks, model=model)

    print()
    print(f"Preview: {playlist_name}")
    _print_preview(ordered.tracks)
    print()
    print(
        "Inputs: "
        f"{len(ordered.tracks)} tracks, "
        f"{ordered.missing_bpm} missing BPM, "
        f"{ordered.missing_year} missing year, "
        f"{ordered.local_files} local files, "
        f"{ordered.audio_features} audio-analyzed"
    )
    if ordered.audio_features == 0 and ordered.local_files == 0:
        print("Audio model unavailable: Apple Music did not expose local audio file paths.")
    print(f"Flow score: {ordered.score:.3f} lower is smoother")

    while True:
        print()
        print("What should I do with this order?")
        print("  1. Create a new ordered playlist (recommended)")
        print("  2. Replace the order of the original playlist")
        print("  3. Save ordered JSON only")
        print("  4. Cancel")
        choice = input("Choose 1-4: ").strip()

        if choice == "1" or choice == "":
            target_name = _available_playlist_name(f"{playlist_name} - Flow")
            custom_name = input(f"New playlist name [{target_name}]: ").strip()
            if custom_name:
                target_name = custom_name
            create_playlist_from_order(playlist_name, target_name, ordered.tracks)
            print(f"Created Apple Music playlist: {target_name}")
            return 0

        if choice == "2":
            print()
            print(f"This will rebuild the order of \"{playlist_name}\" in Apple Music.")
            print("It does not delete songs from your library, but it changes this playlist.")
            confirmation = input("Type REORDER to continue: ").strip()
            if confirmation != "REORDER":
                print("Cancelled original-playlist reorder.")
                continue
            backup_name = rebuild_playlist_in_order(playlist_name, ordered.tracks)
            print(f"Reordered Apple Music playlist: {playlist_name}")
            print(f"Backup playlist kept: {backup_name}")
            return 0

        if choice == "3":
            default_path = Path("exports") / _safe_filename(f"{playlist_name}-flow.json")
            path_text = input(f"Output path [{default_path}]: ").strip()
            path = Path(path_text) if path_text else default_path
            path.parent.mkdir(parents=True, exist_ok=True)
            write_order_json(path, playlist_name, ordered)
            print(f"Wrote ordered JSON: {path}")
            return 0

        if choice == "4":
            print("Cancelled.")
            return 0

        print("Choose 1, 2, 3, or 4.")


def _choose_playlist(playlists) -> str | None:
    playlists = sorted(playlists, key=lambda playlist: playlist.name.casefold())
    filtered = playlists
    while True:
        print("Choose a playlist:")
        for index, playlist in enumerate(filtered, 1):
            print(f"  {index:>2}. {playlist.name} ({playlist.track_count} tracks)")
        print()
        answer = input("Number, search text, or q to quit: ").strip()
        if answer.casefold() in {"q", "quit", "exit"}:
            return None
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(filtered):
                return filtered[index - 1].name
            print("That number is not in the list.")
            continue
        if answer:
            matches = [
                playlist
                for playlist in playlists
                if answer.casefold() in playlist.name.casefold()
            ]
            if not matches:
                print(f"No playlists matched \"{answer}\".")
                continue
            filtered = matches
            continue
        print("Choose a playlist number, or type part of a playlist name.")


def _print_preview(tracks: list[Track], limit: int = 20) -> None:
    visible = tracks[:limit]
    for index, track in enumerate(visible, 1):
        bpm = f"{track.bpm:.0f}" if track.bpm else "-"
        year = str(track.year) if track.year else "-"
        print(f"  {index:>2}. {bpm:>4} BPM  {year:>4}  {track.display_name}")
    hidden = len(tracks) - len(visible)
    if hidden > 0:
        print(f"  ... {hidden} more tracks")


def _build_model(tracks: list[Track]) -> LocalFlowModel:
    local_files = sum(1 for track in tracks if track.location)
    if local_files == 0:
        return LocalFlowModel()

    print(f"Analyzing local audio for {local_files} tracks...")

    def progress(index: int, total: int, track: Track) -> None:
        print(f"  [{index}/{total}] {track.display_name}")

    audio_features = analyze_audio_for_tracks(tracks, progress=progress)
    if not audio_features:
        print("No audio files could be analyzed; using metadata fallback.")
        return LocalFlowModel()
    print(f"Audio features ready for {len(audio_features)} tracks.")
    return LocalFlowModel(audio_features=audio_features)


def _available_playlist_name(base_name: str) -> str:
    try:
        existing = {playlist.name.casefold() for playlist in list_playlists()}
    except AppleMusicError:
        return base_name
    if base_name.casefold() not in existing:
        return base_name
    counter = 2
    while True:
        candidate = f"{base_name} {counter}"
        if candidate.casefold() not in existing:
            return candidate
        counter += 1


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe.casefold() or "playlist-flow"
