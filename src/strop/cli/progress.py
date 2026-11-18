"""A progress line on stderr, shown only on an interactive terminal."""

import sys

from strop.evaluation.runner import Progress


def stderr_progress(label: str) -> Progress | None:
    """Return a callback that rewrites ``label: done/total`` on stderr, or None when piped."""
    if not sys.stderr.isatty():
        return None

    def update(done: int, total: int) -> None:
        end = "\n" if done >= total else ""
        sys.stderr.write(f"\r{label}: {done}/{total}{end}")
        sys.stderr.flush()

    return update
