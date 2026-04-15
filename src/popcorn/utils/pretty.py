"""Pretty-printing utilities — colored text + aligned tables.

Zero dependencies beyond stdlib. Uses ANSI escape codes for color
when the output is a terminal; degrades to plain text otherwise.

Usage:
    from popcorn.utils.pretty import Table, style, print_header

    print_header("Benchmark Results")

    t = Table()
    t.add_column("Kernel", align="left", min_width=20)
    t.add_column("Time (μs)", align="right")
    t.add_column("Status", align="center")

    t.add_row("gemm", "14.82", style("OK", "green"))
    t.add_row("attn", "16.44", style("FAIL", "red"))
    t.print()
"""

from __future__ import annotations

import os
import sys

# ── ANSI color support ──────────────────────────────────────────────

_COLOR_ENABLED: bool | None = None


def _colors_enabled() -> bool:
    global _COLOR_ENABLED
    if _COLOR_ENABLED is None:
        if os.environ.get("NO_COLOR"):
            _COLOR_ENABLED = False
        elif os.environ.get("FORCE_COLOR"):
            _COLOR_ENABLED = True
        else:
            _COLOR_ENABLED = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return _COLOR_ENABLED


_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bold_red": "\033[1;31m",
    "bold_green": "\033[1;32m",
    "bold_yellow": "\033[1;33m",
    "bold_cyan": "\033[1;36m",
}


def style(text: str, *styles: str) -> str:
    """Apply ANSI styles to text. No-op if colors are disabled.

    Styles: bold, dim, red, green, yellow, blue, magenta, cyan,
    bold_red, bold_green, bold_yellow, bold_cyan.
    """
    if not _colors_enabled() or not styles:
        return text
    prefix = "".join(_CODES.get(s, "") for s in styles)
    return f"{prefix}{text}{_CODES['reset']}"


def _visible_len(text: str) -> int:
    """Length of text excluding ANSI escape sequences."""
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", text))


# ── Printing helpers ────────────────────────────────────────────────


def print_header(title: str, *, char: str = "═", width: int = 60) -> None:
    """Print a prominent section header."""
    line = char * width
    print(f"\n{style(line, 'dim')}")
    print(f"  {style(title, 'bold')}")
    print(f"{style(line, 'dim')}")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  {style(title, 'bold', 'cyan')}")


def print_kv(key: str, value: str, *, indent: int = 2) -> None:
    """Print a key-value pair."""
    pad = " " * indent
    print(f"{pad}{style(key, 'dim')}: {value}")


def print_status(label: str, status: str, ok: bool) -> None:
    """Print a status line with colored pass/fail."""
    color = "green" if ok else "red"
    print(f"  {label:<40} {style(status, color)}")


# ── Table ───────────────────────────────────────────────────────────


class _Column:
    __slots__ = ("align", "header", "min_width")

    def __init__(self, header: str, align: str, min_width: int):
        self.header = header
        self.align = align
        self.min_width = min_width


class Table:
    """Aligned table with colored output and rich-style box borders.

    Computes column widths from content, respects ANSI escape codes
    in cell values (visible length only for alignment).

    Default render uses a rounded Unicode box outline with per-column
    separators — mirrors `rich.table.Table` without the dependency::

        ╭──────────┬────────────┬─────────╮
        │ Kernel   │ Time (μs)  │ Status  │
        ├──────────┼────────────┼─────────┤
        │ gemm     │    14.82   │   OK    │
        │ attn     │    16.44   │  FAIL   │
        ╰──────────┴────────────┴─────────╯

    Pass ``border=False`` for the unframed padding-separated layout
    used by older callers (header + dash rule, no sides).

    Supports: left/right/center alignment, min_width per column,
    header separator, row separators via ``add_separator()``.
    """

    def __init__(self, *, padding: int = 1, border: bool = True):
        self._columns: list[_Column] = []
        self._rows: list[list[str] | None] = []
        self._padding = padding
        self._border = border

    def add_column(self, header: str, *, align: str = "left", min_width: int = 0) -> None:
        self._columns.append(_Column(header, align, min_width))

    def add_row(self, *cells: str) -> None:
        row = list(cells)
        # Pad with empty strings if fewer cells than columns
        while len(row) < len(self._columns):
            row.append("")
        self._rows.append(row)

    def add_separator(self) -> None:
        """Add a visual separator row."""
        self._rows.append(None)

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _align(text: str, width: int, align: str) -> str:
        vlen = _visible_len(text)
        deficit = max(0, width - vlen)
        if align == "right":
            return " " * deficit + text
        if align == "center":
            left = deficit // 2
            right = deficit - left
            return " " * left + text + " " * right
        return text + " " * deficit

    def _widths(self) -> list[int]:
        return [
            max(
                col.min_width,
                _visible_len(col.header),
                *(_visible_len(row[i]) for row in self._rows if row is not None and i < len(row)),
            )
            for i, col in enumerate(self._columns)
        ]

    # ── render paths ───────────────────────────────────────────────

    def _render_bordered(self) -> str:
        """Rich-style outlined render with box-drawing borders."""
        widths = self._widths()
        pad = self._padding
        cell_widths = [w + 2 * pad for w in widths]

        def hline(left: str, mid: str, right: str, fill: str = "─") -> str:
            return left + mid.join(fill * w for w in cell_widths) + right

        def data_row(cells: list[str]) -> str:
            body = [
                " " * pad + self._align(cells[i], widths[i], self._columns[i].align) + " " * pad
                for i in range(len(self._columns))
            ]
            return style("│", "dim") + style("│", "dim").join(body) + style("│", "dim")

        lines: list[str] = []
        # Top border
        lines.append(style(hline("╭", "┬", "╮"), "dim"))
        # Header row
        header_cells = [style(col.header, "bold") for col in self._columns]
        lines.append(data_row(header_cells))
        # Header/body separator
        lines.append(style(hline("├", "┼", "┤"), "dim"))
        # Rows (None → mid-table separator line)
        for row in self._rows:
            if row is None:
                lines.append(style(hline("├", "┼", "┤"), "dim"))
                continue
            cells = [row[i] if i < len(row) else "" for i in range(len(self._columns))]
            lines.append(data_row(cells))
        # Bottom border
        lines.append(style(hline("╰", "┴", "╯"), "dim"))
        return "\n".join(lines)

    def _render_plain(self) -> str:
        """Legacy unframed render — header + dash rule, no side pipes."""
        widths = self._widths()
        pad = " " * 2
        lines: list[str] = []
        header_cells = [
            self._align(style(col.header, "bold"), widths[i], col.align)
            for i, col in enumerate(self._columns)
        ]
        lines.append(pad.join(header_cells))
        lines.append(pad.join("─" * w for w in widths))
        for row in self._rows:
            if row is None:
                lines.append(pad.join("─" * w for w in widths))
                continue
            cells = [
                self._align(row[i] if i < len(row) else "", widths[i], self._columns[i].align)
                for i in range(len(self._columns))
            ]
            lines.append(pad.join(cells))
        return "\n".join(lines)

    def render(self) -> str:
        """Render the table as a string."""
        if not self._columns:
            return ""
        return self._render_bordered() if self._border else self._render_plain()

    def print(self, *, indent: int = 2) -> None:
        """Print the rendered table with optional indent."""
        prefix = " " * indent
        for line in self.render().split("\n"):
            print(f"{prefix}{line}")
