"""Small shared helpers: duration parsing/formatting and table rendering."""

import argparse
import re

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_DURATION_RE = re.compile(r"(\d+)([smhd])")


def parse_duration(text: str) -> int:
    """Parse a duration like "3600", "90s", "15m", "2h", "1d" or "1h30m" into seconds."""
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    if text.isdigit():
        return int(text)
    total = 0
    pos = 0
    for match in _DURATION_RE.finditer(text):
        if match.start() != pos:
            raise ValueError(f"invalid duration: {text!r}")
        total += int(match.group(1)) * _UNIT_SECONDS[match.group(2)]
        pos = match.end()
    if pos != len(text):
        raise ValueError(f"invalid duration: {text!r}")
    return total


def duration_arg(text: str) -> int:
    """argparse type= adapter for parse_duration."""
    try:
        return parse_duration(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def format_duration(seconds: float | None) -> str:
    """Format seconds compactly, e.g. 5400 -> "1h30m". None -> "-"."""
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return "".join(parts)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render rows as a left-aligned, space-padded table."""
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    for row in rows:
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    return "\n".join(lines)
