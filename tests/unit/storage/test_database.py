import sqlite3
from pathlib import Path

import pytest

from strop.errors import StorageError
from strop.storage import ProjectPaths, connect, open_database, transaction


def test_paths_are_derived_from_the_root(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    assert paths.config_file == tmp_path / "strop.yaml"
    assert paths.database == tmp_path / ".strop" / "strop.db"


def test_ensure_creates_the_state_directory(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    paths.ensure()
    assert paths.state_dir.is_dir()


def test_connect_enforces_foreign_keys() -> None:
    conn = connect(":memory:")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_connect_reports_an_unusable_path(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="cannot open"):
        connect(tmp_path / "missing-dir" / "strop.db")


def test_transaction_commits_on_success() -> None:
    conn = connect(":memory:")
    conn.execute("CREATE TABLE t (x INTEGER)")
    with transaction(conn):
        conn.execute("INSERT INTO t VALUES (1)")
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


def test_transaction_rolls_back_on_error() -> None:
    conn = connect(":memory:")
    conn.execute("CREATE TABLE t (x INTEGER)")

    def failing_write() -> None:
        with transaction(conn):
            conn.execute("INSERT INTO t VALUES (1)")
            raise sqlite3.IntegrityError("boom")

    with pytest.raises(sqlite3.IntegrityError):
        failing_write()
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0


def test_open_database_creates_and_migrates(tmp_path: Path) -> None:
    conn = open_database(ProjectPaths(tmp_path))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"meta", "schema_version"} <= tables
    conn.close()
    reopened = open_database(ProjectPaths(tmp_path))
    assert reopened.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] >= 1
