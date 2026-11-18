"""A small versioned SQL migration runner for SQLite."""

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources

from strop.errors import StorageError

MIGRATIONS_PACKAGE = "strop.storage.migrations"
_FILENAME = re.compile(r"^(?P<version>\d{3,})_(?P<name>[a-z0-9_]+)\.sql$")

_CREATE_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


def discover_migrations(package: str = MIGRATIONS_PACKAGE) -> list[Migration]:
    """Load migrations shipped inside ``package``, ordered by version."""
    found = []
    for entry in resources.files(package).iterdir():
        match = _FILENAME.match(entry.name)
        if match:
            found.append(
                Migration(int(match["version"]), match["name"], entry.read_text(encoding="utf-8"))
            )
    return _validated(sorted(found, key=lambda m: m.version))


def current_version(conn: sqlite3.Connection) -> int:
    """The highest applied migration version, or 0 for a new database."""
    conn.execute(_CREATE_VERSION_TABLE)
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0] or 0)


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration] | None = None) -> list[int]:
    """Apply pending migrations in order and return the versions that were applied.

    Each migration runs in its own transaction together with its version record,
    so a failure leaves the database at the last good version.
    """
    available = _validated(list(migrations if migrations is not None else discover_migrations()))
    applied = current_version(conn)
    if available and applied > available[-1].version:
        raise StorageError(
            f"the project database is at schema version {applied}, newer than this "
            f"version of strop understands ({available[-1].version}); upgrade strop"
        )
    done = []
    for migration in (m for m in available if m.version > applied):
        _apply(conn, migration)
        done.append(migration.version)
    return done


def _apply(conn: sqlite3.Connection, migration: Migration) -> None:
    record = (
        f"INSERT INTO schema_version (version, name) "
        f"VALUES ({migration.version}, '{migration.name}');"
    )
    try:
        conn.executescript(f"BEGIN;\n{migration.sql}\n{record}\nCOMMIT;")
    except sqlite3.Error as exc:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise StorageError(
            f"migration {migration.version:03d}_{migration.name} failed: {exc}"
        ) from exc


def _validated(migrations: list[Migration]) -> list[Migration]:
    for expected, migration in enumerate(migrations, start=1):
        if migration.version != expected:
            raise StorageError(
                f"migration versions must be consecutive from 1; "
                f"expected {expected:03d}, found {migration.version:03d}"
            )
    return migrations
