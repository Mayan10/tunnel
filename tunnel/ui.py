"""Small ANSI styling helpers for Tunnel's terminal output.

No dependency is added for this: Tunnel stays network-free and dependency-light
(see README), so this hand-rolls the handful of primitives the interactive app
and CLI need instead of pulling in a TUI library.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable

_RESET = "\033[0m"
_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "gray": "\033[90m",
    "white": "\033[97m",
}


def _enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TUNNEL_FORCE_COLOR"):
        return True
    if not sys.stdout.isatty():
        return False
    return os.environ.get("TERM", "") != "dumb"


COLOR = _enabled()


def style(text: str, *codes: str) -> str:
    if not COLOR or not codes:
        return text
    prefix = "".join(_CODES[code] for code in codes)
    return f"{prefix}{text}{_RESET}"


def accent(text: str) -> str:
    return style(text, "magenta", "bold")


def muted(text: str) -> str:
    return style(text, "gray")


def ok(text: str) -> str:
    return style(text, "green")


def warn(text: str) -> str:
    return style(text, "yellow")


CHECK = "✓"
ARROW = "❯"
DOT = "·"
BULLET = "•"


def banner(title: str, subtitle: str = "") -> str:
    inner_width = max(len(title), len(subtitle)) + 4
    top = "╭" + "─" * inner_width + "╮"
    bottom = "╰" + "─" * inner_width + "╯"
    lines = [style(top, "magenta")]
    title_line = f"  {style(title, 'bold', 'white')}"
    pad = inner_width - len(title) - 2
    lines.append(style("│", "magenta") + title_line + " " * pad + style("│", "magenta"))
    if subtitle:
        subtitle_line = f"  {muted(subtitle)}"
        pad = inner_width - len(subtitle) - 2
        lines.append(style("│", "magenta") + subtitle_line + " " * pad + style("│", "magenta"))
    lines.append(style(bottom, "magenta"))
    return "\n".join(lines)


def section(text: str) -> str:
    return style(text, "bold", "cyan")


def prompt(text: str) -> str:
    return f"{style(ARROW, 'magenta', 'bold')} {text}"


def numbered(index: int, text: str, width: int = 2) -> str:
    return f"  {style(f'{index:>{width}}', 'cyan')}{muted('.')} {text}"


def kv(items: Iterable[tuple[str, str]]) -> str:
    return muted(", ").join(f"{muted(key)} {value}" for key, value in items)


def _pad(text: str, width: int, align: str) -> str:
    return text.rjust(width) if align == "r" else text.ljust(width)


def table_top(widths: list[int]) -> str:
    return style(_hline(widths, "╭", "┬", "╮"), "gray")


def table_mid(widths: list[int]) -> str:
    return style(_hline(widths, "├", "┼", "┤"), "gray")


def table_bottom(widths: list[int]) -> str:
    return style(_hline(widths, "╰", "┴", "╯"), "gray")


def table_row(
    cells: list[str],
    widths: list[int],
    aligns: list[str] | None = None,
    codes: list[tuple[str, ...]] | None = None,
) -> str:
    aligns = aligns or ["l"] * len(cells)
    codes = codes or [() for _ in cells]
    border = style("│", "gray")
    parts = [
        style(_pad(text, width, align), *cell_codes) if cell_codes else _pad(text, width, align)
        for text, width, align, cell_codes in zip(cells, widths, aligns, codes)
    ]
    return f"{border} " + f" {border} ".join(parts) + f" {border}"


def _hline(widths: list[int], left: str, mid: str, right: str) -> str:
    segments = ["─" * (width + 2) for width in widths]
    return left + mid.join(segments) + right


class Spinner:
    """A single-line, carriage-return-driven progress indicator.

    Falls back to one line per update when stdout is not a TTY (piped output,
    CI logs), so nothing is lost when color/animation is unavailable.
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        self._frame = 0
        self._live = COLOR
        self._last_len = 0

    def update(self, text: str) -> None:
        if not self._live:
            print(f"  {text}")
            return
        frame = self.FRAMES[self._frame % len(self.FRAMES)]
        self._frame += 1
        line = f"  {style(frame, 'magenta')} {text}"
        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def finish(self, text: str) -> None:
        if not self._live:
            print(f"  {style(CHECK, 'green')} {text}")
            return
        line = f"  {style(CHECK, 'green')} {text}"
        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad + "\n")
        sys.stdout.flush()
        self._last_len = 0
