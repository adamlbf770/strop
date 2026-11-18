"""An opened Strop project: its directory, validated configuration, and database."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from strop.config import StropConfig, load_config
from strop.storage import ProjectPaths, open_database


@dataclass(frozen=True)
class Project:
    root: Path
    paths: ProjectPaths
    config: StropConfig

    def connect(self) -> sqlite3.Connection:
        """Open the project database, creating and migrating it if needed."""
        return open_database(self.paths)

    @contextmanager
    def database(self) -> Iterator[sqlite3.Connection]:
        """Open the project database for the duration of a ``with`` block."""
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()


def open_project(root: Path) -> Project:
    """Load strop.yaml from ``root``; raise ``ConfigError`` if it is missing or invalid."""
    paths = ProjectPaths(root)
    return Project(root, paths, load_config(paths.config_file))
