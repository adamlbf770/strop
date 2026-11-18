import sqlite3

import pytest

from strop.errors import ConfigError
from strop.prompts import PromptMessage, PromptTemplate
from strop.prompts.versions import parse_version_ref
from strop.storage import connect
from strop.storage.migrator import migrate
from strop.storage.prompts import PromptRepository


def template(text: str) -> PromptTemplate:
    return PromptTemplate((PromptMessage("system", text),), frozenset({"system"}))


@pytest.fixture
def repo() -> PromptRepository:
    conn = connect(":memory:")
    migrate(conn)
    return PromptRepository(conn)


def test_versions_are_numbered_and_linked(repo: PromptRepository) -> None:
    first = repo.add(template("a"), parent_number=None)
    second = repo.add(
        template("b"), parent_number=1, change_summary="tweak", reason="r", source="optimizer"
    )
    assert (first.ref, second.ref) == ("v1", "v2")
    assert second.parent_number == 1
    assert second.change_summary == "tweak"
    assert second.source == "optimizer"
    assert second.template == template("b")
    assert [v.number for v in repo.list()] == [1, 2]
    assert repo.first() == first


def test_unknown_version_is_reported(repo: PromptRepository) -> None:
    with pytest.raises(ConfigError, match="v9 does not exist"):
        repo.get(9)


def test_best_pointer_moves_without_touching_versions(repo: PromptRepository) -> None:
    repo.add(template("a"), parent_number=None)
    repo.add(template("b"), parent_number=1)
    assert repo.best() is None
    repo.set_best(2)
    assert repo.best_number() == 2
    repo.set_best(1)
    assert repo.best() is not None
    assert len(repo.list()) == 2
    with pytest.raises(ConfigError):
        repo.set_best(5)


def test_versions_cannot_be_updated_or_deleted() -> None:
    conn = connect(":memory:")
    migrate(conn)
    PromptRepository(conn).add(template("a"), parent_number=None)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE prompt_versions SET change_summary = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM prompt_versions")


@pytest.mark.parametrize(("text", "number"), [("v3", 3), ("3", 3), (" V12 ", 12)])
def test_parse_version_ref(text: str, number: int) -> None:
    assert parse_version_ref(text) == number


@pytest.mark.parametrize("text", ["v", "vx", "v0", "-1", ""])
def test_parse_version_ref_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ConfigError, match="invalid prompt version"):
        parse_version_ref(text)
