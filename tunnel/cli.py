from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import resources
from pathlib import Path

from . import __version__, ui
from .apple_music import (
    AppleMusicError,
    copy_playlist,
    create_playlist_from_order,
    export_playlist,
    export_tracks,
    list_playlists,
)
from .audio import analyze_audio_for_tracks
from .embedding import EmbeddingError, embed_tracks
from .interactive import run_app
from .io import load_tracks_json, write_m3u, write_order_json
from .model import LocalFlowModel, train_playlist_model
from .ordering import OrderingConfig, order_tracks
from .snapshot import write_playlist_snapshot
from .types import Track


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        try:
            return run_app()
        except (AppleMusicError, EmbeddingError, OSError, ValueError) as exc:
            print(f"tunnel: {exc}", file=sys.stderr)
            return 1

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (AppleMusicError, EmbeddingError, OSError, ValueError) as exc:
        print(f"tunnel: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunnel",
        description="Order an Apple Music playlist into a smoother listening flow.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(required=True)

    app_parser = subparsers.add_parser("app", help="Open the interactive terminal app.")
    app_parser.set_defaults(handler=cmd_app)

    list_parser = subparsers.add_parser("list", help="List playlists from the macOS Music app.")
    list_parser.set_defaults(handler=cmd_list)

    export_parser = subparsers.add_parser("export", help="Export a Music playlist to JSON.")
    export_parser.add_argument("playlist", help="Exact Apple Music playlist name.")
    export_parser.add_argument("--out", type=Path, required=True, help="JSON file to write.")
    export_parser.set_defaults(handler=cmd_export)

    order_parser = subparsers.add_parser("order", help="Compute a smooth playlist order.")
    order_parser.add_argument("playlist", nargs="?", help="Exact Apple Music playlist name.")
    order_parser.add_argument("--from-json", type=Path, help="Use exported JSON instead of reading Music.")
    order_parser.add_argument("--out", type=Path, help="Write ordered JSON.")
    order_parser.add_argument("--m3u", type=Path, help="Write an M3U playlist for tracks with local files.")
    order_parser.add_argument("--create-playlist", help="Create a new Apple Music playlist with the ordered tracks.")
    order_parser.add_argument("--limit", type=int, help="Use only the first N tracks while testing.")
    order_parser.add_argument("--max-print", type=int, default=120, help="Maximum rows to print in the terminal.")
    order_parser.add_argument("--no-polish", action="store_true", help="Skip local swap polishing for very fast runs.")
    order_parser.set_defaults(handler=cmd_order)

    demo_parser = subparsers.add_parser("demo", help="Run the model against the bundled sample playlist.")
    demo_parser.set_defaults(handler=cmd_demo)

    restore_parser = subparsers.add_parser("restore", help="Copy a backup playlist into a new restored playlist.")
    restore_parser.add_argument("backup_playlist", help="Backup playlist name, usually containing 'Before Tunnel'.")
    restore_parser.add_argument("--to", required=True, help="New playlist name to create from the backup.")
    restore_parser.set_defaults(handler=cmd_restore)

    return parser


def cmd_app(_: argparse.Namespace) -> int:
    return run_app()


def cmd_list(_: argparse.Namespace) -> int:
    playlists = sorted(list_playlists(), key=lambda item: item.name.casefold())
    if not playlists:
        print(ui.muted("No playlists found."))
        return 0
    name_width = min(max(len(playlist.name) for playlist in playlists), 56)
    widths = [name_width, 6]
    print(ui.table_top(widths))
    print(ui.table_row(["Playlist", "Tracks"], widths, ["l", "r"], [("bold",), ("bold",)]))
    print(ui.table_mid(widths))
    for playlist in playlists:
        cells = [_clip(playlist.name, name_width), str(playlist.track_count)]
        print(ui.table_row(cells, widths, ["l", "r"]))
    print(ui.table_bottom(widths))
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    copy_playlist(args.backup_playlist, args.to)
    print(f"Created restored playlist: {args.to}")
    print(f"Source backup kept unchanged: {args.backup_playlist}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    payload = export_playlist(args.playlist)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_json_dumps(payload), encoding="utf-8")
    print(f"Exported {len(payload.get('tracks', []))} tracks to {args.out}")
    return 0


def cmd_order(args: argparse.Namespace) -> int:
    playlist_name, tracks = _load_tracks_for_order(args)
    if args.limit is not None:
        if args.create_playlist:
            raise ValueError("--limit cannot be used with --create-playlist.")
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero.")
        tracks = tracks[: args.limit]

    config = OrderingConfig(swap_passes=0 if args.no_polish else 3)
    model = _build_model(tracks)
    ordered = order_tracks(tracks, model=model, config=config)
    _print_order(playlist_name, ordered.tracks, ordered.score, args.max_print)
    _print_diagnostics(ordered)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_order_json(args.out, playlist_name, ordered)
        print(f"\nWrote ordered JSON: {args.out}")

    if args.m3u:
        args.m3u.parent.mkdir(parents=True, exist_ok=True)
        written = write_m3u(args.m3u, ordered.tracks)
        print(f"Wrote M3U: {args.m3u} ({written} local files)")

    if args.create_playlist:
        if args.from_json and not args.playlist:
            raise ValueError("--create-playlist needs the source Apple Music playlist name.")
        create_playlist_from_order(args.playlist or playlist_name, args.create_playlist, ordered.tracks)
        print(f"Created Apple Music playlist: {args.create_playlist}")

    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    sample_text = resources.files("tunnel").joinpath("sample_playlist.json").read_text(encoding="utf-8")
    payload = json.loads(sample_text)
    playlist_name = str(payload.get("playlist") or "sample")
    tracks = [Track.from_dict(item, index + 1) for index, item in enumerate(payload.get("tracks", []))]
    ordered = order_tracks(tracks, model=train_playlist_model(tracks))
    _print_order(playlist_name, ordered.tracks, ordered.score, max_print=80)
    _print_diagnostics(ordered)
    return 0


def _load_tracks_for_order(args: argparse.Namespace) -> tuple[str, list[Track]]:
    if args.from_json:
        playlist_name, tracks = load_tracks_json(args.from_json)
        if args.playlist:
            playlist_name = args.playlist
        return playlist_name, tracks
    if not args.playlist:
        raise ValueError("Provide a playlist name, or use --from-json.")
    tracks = export_tracks(args.playlist)
    snapshot_path = write_playlist_snapshot(args.playlist, tracks)
    print(f"Snapshot saved: {snapshot_path}")
    return args.playlist, tracks


def _print_order(playlist_name: str, tracks: list[Track], score: float, max_print: int) -> None:
    print()
    print(ui.section(playlist_name))
    print(f"{ui.muted('Model score')} {ui.accent(f'{score:.3f}')} {ui.muted('(lower is smoother)')}")
    print()
    if not tracks:
        print(ui.muted("No tracks to order."))
        return

    terminal_width = shutil.get_terminal_size((100, 24)).columns
    title_width = max(30, min(terminal_width - 26, 60))
    visible_tracks = tracks[:max_print]
    widths = [max(3, len(str(len(visible_tracks)))), 5, 4, title_width]

    print(ui.table_top(widths))
    print(ui.table_row(["#", "BPM", "Year", "Track"], widths, ["r", "r", "r", "l"], [("bold",)] * 4))
    print(ui.table_mid(widths))
    for index, track in enumerate(visible_tracks, 1):
        bpm = f"{track.bpm:.0f}" if track.bpm else "-"
        year = str(track.year) if track.year else "-"
        title = _clip(track.display_name, title_width)
        codes = [("gray",), (), (), ()]
        print(ui.table_row([str(index), bpm, year, title], widths, ["r", "r", "r", "l"], codes))
    print(ui.table_bottom(widths))

    hidden = len(tracks) - len(visible_tracks)
    if hidden > 0:
        print(ui.muted(f"... {hidden} more tracks hidden by --max-print"))


def _print_diagnostics(ordered) -> None:
    print()
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
    print(f"{ui.muted('Model')} {ordered.model_name}")
    if ordered.audio_features == 0 and ordered.local_files == 0:
        print(ui.muted("No local audio to analyze: Apple Music did not expose local file paths for these tracks."))


def _build_model(tracks: list[Track]) -> LocalFlowModel:
    local_files = sum(1 for track in tracks if track.location)
    if local_files == 0:
        return train_playlist_model(tracks)

    print(ui.section(f"Analyzing local audio for {local_files} tracks..."))
    audio_features = analyze_audio_for_tracks(tracks)
    if not audio_features:
        print(ui.warn("No audio files could be analyzed; using metadata only."))
        return train_playlist_model(tracks)
    print(f"  {ui.style(ui.CHECK, 'green')} Audio features ready for {len(audio_features)} tracks.")

    embeddings = embed_tracks(tracks, audio_features)
    print(f"  {ui.style(ui.CHECK, 'green')} Core ML embeddings ready for {len(embeddings)} tracks (audio-embedding-v1).")
    return train_playlist_model(tracks, audio_features=audio_features, embeddings=embeddings)


def _clip(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _json_dumps(payload) -> str:
    import json

    return json.dumps(payload, indent=2) + "\n"
