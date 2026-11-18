"""Shared helpers and type aliases for tests."""

from collections.abc import Callable
from pathlib import Path

CliRunner = Callable[..., tuple[int, str, str]]


def write_dataset(project_dir: Path, rows: int = 100, *, bad_row: bool = False) -> None:
    """Write a two-label CSV dataset; ``bad_row`` blanks the message of the second row."""
    lines = ["message,category"]
    for i in range(rows):
        label = "refund" if i % 2 else "billing"
        message = "" if bad_row and i == 1 else f"message {i}"
        lines.append(f"{message},{label}")
    (project_dir / "data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
