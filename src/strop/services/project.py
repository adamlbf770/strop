"""An opened Strop project: its directory, validated configuration, and database."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from strop.config import StropConfig, load_config
from strop.config.llm import ModelSpec
from strop.env import load_env_file
from strop.storage import ProjectPaths, open_database


@dataclass(frozen=True)
class Project:
    root: Path
    paths: ProjectPaths
    config: StropConfig

    def connect(self) -> sqlite3.Connection:
        """Open the project database, creating and migrating it if needed."""
        return open_database(self.paths)

    def with_models(self, overrides: dict[str, str | None]) -> "Project":
        """A copy that uses the given model names for some roles; None values are ignored."""
        models = self.config.models
        changes: dict[str, ModelSpec] = {}
        for role, name in overrides.items():
            if name:
                current: ModelSpec | None = getattr(models, role)
                changes[role] = (
                    current.model_copy(update={"model": name}) if current else ModelSpec(model=name)
                )
        if not changes:
            return self
        config = self.config.model_copy(update={"models": models.model_copy(update=changes)})
        return replace(self, config=config)

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
    config = load_config(paths.config_file)
    load_env_file(root / ".env")
    return Project(root, paths, config)
