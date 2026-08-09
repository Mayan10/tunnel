"""Small ANSI styling helpers for Tunnel's terminal output.

No dependency is added for this: Tunnel stays network-free and dependency-light
(see README), so this hand-rolls the handful of primitives the interactive app
and CLI need instead of pulling in a TUI library.

Palette is deliberately restrained: default foreground for body text, gray for
secondary/meta text, and a single blue accent for interactive or emphasized
elements (prompts, section labels, the flow score). Red is reserved for actual
warnings and errors.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable

_RESET = "\033[0m"
_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "gray": "\033[90m",
    "red": "\033[31m",
    "blue": "\033[38;2;46;27;223m",
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
    return style(text, "blue")


def muted(text: str) -> str:
    return style(text, "gray")


def warn(text: str) -> str:
    return style(text, "red")


ARROW = "›"


def banner(title: str, subtitle: str = "") -> str:
    lines = [style(title, "bold", "blue")]
    if subtitle:
        lines.append(muted(subtitle))
    return "\n".join(lines)


def section(text: str) -> str:
    return style(text, "bold")


def prompt(text: str) -> str:
    return f"{style(ARROW, 'blue')} {text}"


def numbered(index: int, text: str, width: int = 2) -> str:
    return f"  {muted(f'{index:>{width}}')}  {text}"


def kv(items: Iterable[tuple[str, str]]) -> str:
    return muted("  ").join(f"{muted(key)} {value}" for key, value in items)


def _pad(text: str, width: int, align: str) -> str:
    return text.rjust(width) if align == "r" else text.ljust(width)


def table(
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
    aligns: list[str] | None = None,
    row_codes: list[list[tuple[str, ...]]] | None = None,
) -> list[str]:
    """A plain, minimal table: bold header, a thin rule, left-aligned rows.

    No box-drawing grid — closer to `git log --stat` / `pip list` than a
    bordered widget, which reads calmer at a glance.
    """
    aligns = aligns or ["l"] * len(headers)
    lines = ["  " + "  ".join(style(_pad(h, w, a), "bold") for h, w, a in zip(headers, widths, aligns))]
    lines.append(muted("  " + "─" * (sum(widths) + 2 * (len(widths) - 1))))
    for row_index, row in enumerate(rows):
        codes = row_codes[row_index] if row_codes else [() for _ in row]
        cells = [
            style(_pad(text, width, align), *cell_codes) if cell_codes else _pad(text, width, align)
            for text, width, align, cell_codes in zip(row, widths, aligns, codes)
        ]
        lines.append("  " + "  ".join(cells))
    return lines


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
        line = f"  {muted(frame)} {text}"
        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def finish(self, text: str) -> None:
        if not self._live:
            print(f"  {text}")
            return
        line = f"  {text}"
        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad + "\n")
        sys.stdout.flush()
        self._last_len = 0
