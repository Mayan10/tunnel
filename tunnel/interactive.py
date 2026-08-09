from __future__ import annotations

import shutil
from pathlib import Path

from . import __version__, ui
from .apple_music import (
    AppleMusicError,
    create_playlist_from_order,
    export_tracks,
    list_playlists,
)
from .audio import analyze_audio_for_tracks
from .embedding import embed_tracks
from .io import write_order_json
from .model import LocalFlowModel, train_playlist_model
from .ordering import order_tracks
from .snapshot import write_playlist_snapshot
from .types import Track


def run_app() -> int:
    print()
    print(ui.banner("Tunnel", f"v{__version__} — smoother Apple Music playlists, on-device"))
    print()

    playlists = [playlist for playlist in list_playlists() if playlist.track_count > 0]
    if not playlists:
        print(ui.warn("No playlists with tracks were found in Apple Music."))
        return 1

    playlist_name = _choose_playlist(playlists)
    if not playlist_name:
        print(ui.muted("Cancelled."))
        return 0

    print()
    print(ui.section(f"Reading \"{playlist_name}\" from Apple Music..."))
    tracks = export_tracks(playlist_name)
    if not tracks:
        print(ui.warn("That playlist has no tracks."))
        return 1
    snapshot_path = write_playlist_snapshot(playlist_name, tracks)
    print(f"  {ui.muted(f'Snapshot saved: {snapshot_path}')}")

    model = _build_model(tracks)
    print()
    print(ui.section(f"Ordering {len(tracks)} tracks with {model.name}..."))
    ordered = order_tracks(tracks, model=model)

    print()
    print(ui.section(f"Preview: {playlist_name}"))
    _print_preview(ordered.tracks)
    print()
    _print_diagnostics(ordered)

    while True:
        print()
        print(ui.section("What should I do with this order?"))
        print(ui.numbered(1, f"Create a new ordered playlist {ui.muted('(recommended)')}"))
        print(ui.numbered(2, "Save ordered JSON only"))
        print(ui.numbered(3, "Cancel"))
        choice = input(ui.prompt("Choose 1-3: ")).strip()

        if choice == "1" or choice == "":
            target_name = _available_playlist_name(f"{playlist_name} - Flow")
            custom_name = input(ui.prompt(f"New playlist name [{target_name}]: ")).strip()
            if custom_name:
                target_name = custom_name
            create_playlist_from_order(playlist_name, target_name, ordered.tracks)
            print(f"Created Apple Music playlist: {ui.accent(target_name)}")
            return 0

        if choice == "2":
            default_path = Path("exports") / _safe_filename(f"{playlist_name}-flow.json")
            path_text = input(ui.prompt(f"Output path [{default_path}]: ")).strip()
            path = Path(path_text) if path_text else default_path
            path.parent.mkdir(parents=True, exist_ok=True)
            write_order_json(path, playlist_name, ordered)
            print(f"Wrote ordered JSON: {path}")
            return 0

        if choice == "3":
            print(ui.muted("Cancelled."))
            return 0

        print(ui.warn("Choose 1, 2, or 3."))


def _choose_playlist(playlists) -> str | None:
    playlists = sorted(playlists, key=lambda playlist: playlist.name.casefold())
    filtered = playlists
    while True:
        print(ui.section("Choose a playlist:"))
        width = len(str(len(filtered)))
        for index, playlist in enumerate(filtered, 1):
            count = ui.muted(f"({playlist.track_count} tracks)")
            print(ui.numbered(index, f"{playlist.name} {count}", width=width))
        print()
        answer = input(ui.prompt("Number, search text, or q to quit: ")).strip()
        if answer.casefold() in {"q", "quit", "exit"}:
            return None
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(filtered):
                return filtered[index - 1].name
            print(ui.warn("That number is not in the list."))
            continue
        if answer:
            matches = [
                playlist
                for playlist in playlists
                if answer.casefold() in playlist.name.casefold()
            ]
            if not matches:
                print(ui.warn(f"No playlists matched \"{answer}\"."))
                continue
            filtered = matches
            continue
        print(ui.warn("Choose a playlist number, or type part of a playlist name."))


def _print_preview(tracks: list[Track], limit: int = 20) -> None:
    visible = tracks[:limit]
    terminal_width = shutil.get_terminal_size((100, 24)).columns
    title_width = max(20, min(terminal_width - 24, 60))
    widths = [len(str(len(visible))), 4, 4, title_width]

    rows = []
    codes = []
    for index, track in enumerate(visible, 1):
        bpm = f"{track.bpm:.0f}" if track.bpm else "-"
        year = str(track.year) if track.year else "-"
        title = _clip(track.display_name, title_width)
        rows.append([str(index), bpm, year, title])
        codes.append([("gray",), (), (), ()])

    for line in ui.table(["#", "BPM", "Year", "Track"], rows, widths, ["r", "r", "r", "l"], codes):
        print(line)

    hidden = len(tracks) - len(visible)
    if hidden > 0:
        print(f"  {ui.muted(f'... {hidden} more tracks')}")


def _print_diagnostics(ordered) -> None:
    print(
        ui.kv(
            [
                ("tracks", str(len(ordered.tracks))),
                ("missing BPM", str(ordered.missing_bpm)),
                ("missing year", str(ordered.missing_year)),
                ("local files", str(ordered.local_files)),
                ("audio-analyzed", str(ordered.audio_features)),
                ("neural-embedded", str(ordered.embeddings)),
            ]
        )
    )
    if ordered.audio_features == 0 and ordered.local_files == 0:
        print(ui.muted("No local audio to analyze: Apple Music did not expose local file paths for these tracks."))
    print(f"{ui.muted('Flow score')} {ui.accent(f'{ordered.score:.3f}')} {ui.muted('(lower is smoother)')}")


def _build_model(tracks: list[Track]) -> LocalFlowModel:
    local_files = sum(1 for track in tracks if track.location)
    if local_files == 0:
        return train_playlist_model(tracks)

    print()
    print(ui.section(f"Analyzing local audio for {local_files} tracks..."))

    spinner = ui.Spinner()

    def progress(index: int, total: int, track: Track) -> None:
        spinner.update(f"{ui.muted(f'[{index}/{total}]')} {track.display_name}")

    result = analyze_audio_for_tracks(tracks, progress=progress)

    if not result.features:
        if result.protected:
            print(
                ui.warn(
                    f"{result.protected} of {local_files} tracks are DRM-protected Apple Music "
                    "downloads (.movpkg) — encrypted streaming bundles, not audio files — and "
                    "can't be decoded on-device. Using metadata only."
                )
            )
        else:
            print(ui.warn("No audio files could be analyzed; using metadata only."))
        return train_playlist_model(tracks)

    spinner.finish(f"Audio features ready for {len(result.features)} of {local_files} tracks.")
    if result.protected:
        print(ui.muted(f"  {result.protected} tracks are DRM-protected and were skipped."))
    if result.failed:
        print(ui.muted(f"  {result.failed} tracks could not be decoded and were skipped."))

    embeddings = embed_tracks(tracks, result.features)
    print(f"  Core ML embeddings ready for {len(embeddings)} tracks (audio-embedding-v1).")
    return train_playlist_model(tracks, audio_features=result.features, embeddings=embeddings)


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


def _clip(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe.casefold() or "playlist-flow"
