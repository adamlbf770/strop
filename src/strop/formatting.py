"""Small text helpers shared by services and the terminal views."""

MAX_TEXT = 70


def fmt(value: float | None, digits: int = 4) -> str:
    """A number with fixed decimals, or ``n/a`` when it is unavailable."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def shorten(text: str, limit: int = MAX_TEXT) -> str:
    """Collapse whitespace and cut ``text`` to ``limit`` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."
