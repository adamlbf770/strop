"""Persistence for prompt versions and the current best pointer."""

import sqlite3

from strop.errors import ConfigError
from strop.prompts.models import PromptTemplate
from strop.prompts.versions import PromptVersion, Source
from strop.storage.database import transaction

BEST_KEY = "best_prompt_version"

_COLUMNS = "number, parent_number, content_json, change_summary, reason, source, created_at"


class PromptRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        template: PromptTemplate,
        *,
        parent_number: int | None,
        change_summary: str = "",
        reason: str = "",
        source: Source = "user",
    ) -> PromptVersion:
        """Store a new version with the next number; existing versions are never touched."""
        with transaction(self._conn):
            row = self._conn.execute("SELECT COALESCE(MAX(number), 0) + 1 FROM prompt_versions")
            number = int(row.fetchone()[0])
            self._conn.execute(
                "INSERT INTO prompt_versions (number, parent_number, content_json, content_hash, "
                "change_summary, reason, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    number,
                    parent_number,
                    template.to_json(),
                    template.content_hash,
                    change_summary,
                    reason,
                    source,
                ),
            )
        return self.get(number)

    def get(self, number: int) -> PromptVersion:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM prompt_versions WHERE number = ?", (number,)
        ).fetchone()
        if row is None:
            raise ConfigError(f"prompt version v{number} does not exist")
        return _to_version(row)

    def list(self) -> list[PromptVersion]:
        rows = self._conn.execute(f"SELECT {_COLUMNS} FROM prompt_versions ORDER BY number")
        return [_to_version(row) for row in rows.fetchall()]

    def first(self) -> PromptVersion | None:
        rows = self.list()
        return rows[0] if rows else None

    def best_number(self) -> int | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (BEST_KEY,)).fetchone()
        return int(row["value"]) if row else None

    def best(self) -> PromptVersion | None:
        number = self.best_number()
        return self.get(number) if number is not None else None

    def set_best(self, number: int) -> None:
        """Point "best" at an existing version. This never deletes or alters any version."""
        self.get(number)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (BEST_KEY, str(number))
        )


def _to_version(row: sqlite3.Row) -> PromptVersion:
    return PromptVersion(
        number=row["number"],
        parent_number=row["parent_number"],
        template=PromptTemplate.from_json(row["content_json"]),
        change_summary=row["change_summary"],
        reason=row["reason"],
        source=row["source"],
        created_at=row["created_at"],
    )
