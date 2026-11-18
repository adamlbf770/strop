"""Local project storage: paths, SQLite connection, and migrations."""

from strop.storage.database import connect, open_database, transaction
from strop.storage.paths import ProjectPaths

__all__ = ["ProjectPaths", "connect", "open_database", "transaction"]
