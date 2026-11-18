"""Shared type aliases for tests."""

from collections.abc import Callable

CliRunner = Callable[..., tuple[int, str, str]]
