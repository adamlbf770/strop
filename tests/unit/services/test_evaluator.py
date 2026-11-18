from pathlib import Path

import pytest

from strop.data.models import Partition
from strop.errors import ConfigError, DataError
from strop.llm import MockLLMClient
from strop.services.data import create_split, effective_split_config
from strop.services.evaluate import sample_rows
from strop.services.evaluator import RunTag, build_evaluator
from strop.services.project import open_project
from strop.services.prompts import ensure_initial_version
from strop.storage.prompts import PromptRepository
from tests.helpers import CliRunner


@pytest.fixture
def opened(run_cli: CliRunner, project: Path) -> Path:
    assert run_cli("data", "split", "--dir", str(project))[0] == 0
    return project


async def test_test_partition_is_locked_unless_allowed(
    opened: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("strop.services.evaluator.build_client", lambda *_: MockLLMClient("x"))
    project = open_project(opened)
    with project.database() as conn:
        evaluator = build_evaluator(project, conn)
        with pytest.raises(DataError, match="locked"):
            evaluator.rows(Partition.TEST)
        version = ensure_initial_version(project, PromptRepository(conn))
        with pytest.raises(DataError, match="locked"):
            await evaluator.evaluate(version, Partition.TEST, RunTag("test"))
        unlocked = build_evaluator(project, conn, allow_test=True)
        assert len(unlocked.rows(Partition.TEST)) == 20


def test_partitions_require_a_split(project: Path) -> None:
    opened_project = open_project(project)
    with opened_project.database() as conn, pytest.raises(ConfigError, match="data split"):
        build_evaluator(opened_project, conn)


def test_split_rows_never_leak_between_partitions(opened: Path) -> None:
    project = open_project(opened)
    with project.database() as conn:
        evaluator = build_evaluator(project, conn)
    ids = {p: {r.id for r in evaluator.partitions.get(p)} for p in Partition}
    assert not ids[Partition.TRAIN] & ids[Partition.VALIDATION]
    assert not ids[Partition.TRAIN] & ids[Partition.TEST]
    assert not ids[Partition.VALIDATION] & ids[Partition.TEST]
    assert sum(len(v) for v in ids.values()) == 100


def test_sample_rows_is_reproducible_and_bounded(opened: Path) -> None:
    project = open_project(opened)
    with project.database() as conn:
        rows = build_evaluator(project, conn).rows(Partition.TRAIN)
    first = sample_rows(rows, 10, seed=1)
    assert first == sample_rows(rows, 10, seed=1)
    assert first != sample_rows(rows, 10, seed=2)
    assert len(sample_rows(rows, 1000, seed=1)) == len(rows)


def test_identical_split_is_reused(opened: Path) -> None:
    project = open_project(opened)
    config = effective_split_config(project, {})
    outcome = create_split(project, config, skip_invalid=False)
    assert outcome.created is False
