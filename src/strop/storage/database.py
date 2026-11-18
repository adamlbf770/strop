"""SQLite connection handling."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from strop.errors import StorageError
from strop.storage.migrator import migrate
from strop.storage.paths import ProjectPaths


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection in autocommit mode with foreign keys enforced.

    Transactions are explicit: use ``transaction`` for multi-statement writes.
    """
    try:
        conn = sqlite3.connect(path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.Error as exc:
        raise StorageError(f"cannot open the project database at {path}: {exc}") from exc
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block atomically: commit on success, roll back on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def open_database(paths: ProjectPaths) -> sqlite3.Connection:
    """Create the state directories if needed, open the database, and migrate it."""
    paths.ensure()
    conn = connect(paths.database)
    migrate(conn)
    return conn
