"""Persistence for dataset versions and split assignments."""

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from strop.data.models import Partition
from strop.storage.database import transaction


@dataclass(frozen=True, slots=True)
class SplitRecord:
    id: int
    dataset_id: int
    content_hash: str
    skip_invalid: bool
    seed: int
    strategy: str
    config_json: str


class DatasetRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_dataset(
        self,
        *,
        content_hash: str,
        source_path: str,
        row_count: int,
        skipped_rows: int,
        skip_invalid: bool,
    ) -> int:
        """Record a dataset version (idempotent) and return its id."""
        self._conn.execute(
            "INSERT OR IGNORE INTO datasets "
            "(content_hash, source_path, row_count, skipped_rows, skip_invalid) "
            "VALUES (?, ?, ?, ?, ?)",
            (content_hash, source_path, row_count, skipped_rows, int(skip_invalid)),
        )
        row = self._conn.execute(
            "SELECT id FROM datasets WHERE content_hash = ? AND skip_invalid = ?",
            (content_hash, int(skip_invalid)),
        ).fetchone()
        return int(row["id"])

    def create_split(
        self,
        *,
        dataset_id: int,
        seed: int,
        strategy: str,
        config_json: str,
        assignments: Mapping[str, Partition],
    ) -> int:
        """Store a split and all its row assignments atomically; return the split id."""
        with transaction(self._conn):
            cursor = self._conn.execute(
                "INSERT INTO splits (dataset_id, seed, strategy, config_json) VALUES (?, ?, ?, ?)",
                (dataset_id, seed, strategy, config_json),
            )
            split_id = int(cursor.lastrowid or 0)
            self._conn.executemany(
                "INSERT INTO split_assignments (split_id, row_id, partition) VALUES (?, ?, ?)",
                [(split_id, row_id, part.value) for row_id, part in assignments.items()],
            )
        return split_id

    def find_split(self, dataset_id: int, config_json: str) -> int | None:
        row = self._conn.execute(
            "SELECT id FROM splits WHERE dataset_id = ? AND config_json = ? ORDER BY id DESC",
            (dataset_id, config_json),
        ).fetchone()
        return int(row["id"]) if row else None

    def latest_split(self) -> SplitRecord | None:
        row = self._conn.execute(
            "SELECT s.id, s.dataset_id, d.content_hash, d.skip_invalid, s.seed, s.strategy, "
            "s.config_json FROM splits s JOIN datasets d ON d.id = s.dataset_id "
            "ORDER BY s.id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return SplitRecord(
            row["id"],
            row["dataset_id"],
            row["content_hash"],
            bool(row["skip_invalid"]),
            row["seed"],
            row["strategy"],
            row["config_json"],
        )

    def assignments(self, split_id: int) -> dict[str, Partition]:
        rows = self._conn.execute(
            "SELECT row_id, partition FROM split_assignments WHERE split_id = ?", (split_id,)
        ).fetchall()
        return {r["row_id"]: Partition(r["partition"]) for r in rows}
