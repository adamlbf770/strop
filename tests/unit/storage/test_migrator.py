import pytest

from strop.errors import StorageError
from strop.storage import connect
from strop.storage.migrator import Migration, current_version, discover_migrations, migrate

CREATE_A = Migration(1, "create_a", "CREATE TABLE a (id INTEGER PRIMARY KEY);")
CREATE_B = Migration(2, "create_b", "CREATE TABLE b (id INTEGER PRIMARY KEY);")


def test_shipped_migrations_are_consecutive_and_non_empty() -> None:
    migrations = discover_migrations()
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    assert all(m.sql.strip() for m in migrations)


def test_new_database_starts_at_version_zero() -> None:
    assert current_version(connect(":memory:")) == 0


def test_applies_pending_migrations_in_order() -> None:
    conn = connect(":memory:")
    assert migrate(conn, [CREATE_A, CREATE_B]) == [1, 2]
    assert current_version(conn) == 2


def test_migrate_is_idempotent_and_applies_only_new_ones() -> None:
    conn = connect(":memory:")
    migrate(conn, [CREATE_A])
    assert migrate(conn, [CREATE_A]) == []
    assert migrate(conn, [CREATE_A, CREATE_B]) == [2]


def test_failed_migration_rolls_back_and_keeps_last_good_version() -> None:
    conn = connect(":memory:")
    broken = Migration(2, "broken", "CREATE TABLE c (id INTEGER); SELECT * FROM missing_table;")
    with pytest.raises(StorageError, match="002_broken"):
        migrate(conn, [CREATE_A, broken])
    assert current_version(conn) == 1
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "c" not in tables


def test_gaps_in_versions_are_rejected() -> None:
    with pytest.raises(StorageError, match="consecutive"):
        migrate(connect(":memory:"), [CREATE_A, Migration(3, "skip", "SELECT 1;")])


def test_database_from_a_newer_version_is_rejected() -> None:
    conn = connect(":memory:")
    migrate(conn, [CREATE_A, CREATE_B])
    with pytest.raises(StorageError, match="upgrade strop"):
        migrate(conn, [CREATE_A])
