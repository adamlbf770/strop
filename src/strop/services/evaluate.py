"""The ``baseline`` and ``eval`` workflows."""

import random
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace

from strop.data.models import Partition, Row
from strop.evaluation.models import EvaluationRecord
from strop.evaluation.runner import Progress
from strop.llm.factory import build_client
from strop.prompts.versions import PromptVersion
from strop.services.evaluator import RunTag, build_evaluator
from strop.services.project import Project
from strop.services.prompts import ensure_initial_version, resolve_version
from strop.storage.prompts import PromptRepository


@dataclass(frozen=True)
class BaselineOutcome:
    version: PromptVersion
    records: list[EvaluationRecord]
    train: EvaluationRecord | None
    noise: float | None

    @property
    def reference(self) -> EvaluationRecord:
        """The first validation run; later repeats only measure run-to-run noise."""
        return self.records[0]


def sample_rows(rows: Sequence[Row], size: int, seed: int) -> list[Row]:
    """A reproducible random subset of ``rows`` of at most ``size`` rows."""
    if size >= len(rows):
        return list(rows)
    chosen = random.Random(seed).sample(sorted(rows, key=lambda r: r.id), size)
    return sorted(chosen, key=lambda r: r.id)


async def run_baseline(
    project: Project,
    conn: sqlite3.Connection,
    *,
    repeats: int,
    sample: int | None,
    with_train: bool,
    use_cache: bool,
    progress: Progress | None = None,
) -> BaselineOutcome:
    """Evaluate the initial prompt (v1) on validation, optionally repeated and on train too.

    Repeats after the first bypass the response cache, so they measure real run-to-run noise.
    """
    version = ensure_initial_version(project, PromptRepository(conn))
    evaluator = build_evaluator(project, conn, use_cache=use_cache)
    rows = evaluator.rows(Partition.VALIDATION)
    if sample is not None:
        rows = sample_rows(rows, sample, evaluator.partitions.seed)
    records = []
    for repeat in range(1, repeats + 1):
        runner = evaluator
        if repeat > 1:
            runner = replace(evaluator, client=build_client(project.config.execution, None))
        details = {"repeat": repeat, "sampled": sample is not None}
        tag = RunTag("baseline", details)
        records.append(
            await runner.evaluate(version, Partition.VALIDATION, tag, rows=rows, progress=progress)
        )
    train = None
    if with_train:
        train = await evaluator.evaluate(
            version, Partition.TRAIN, RunTag("baseline"), progress=progress
        )
    return BaselineOutcome(version, records, train, _noise(records))


async def run_eval(
    project: Project,
    conn: sqlite3.Connection,
    *,
    ref: str | None,
    partition: Partition,
    sample: int | None,
    use_cache: bool,
    progress: Progress | None = None,
) -> tuple[PromptVersion, EvaluationRecord]:
    """Score one prompt version on a partition without changing anything."""
    repo = PromptRepository(conn)
    ensure_initial_version(project, repo)
    version = resolve_version(repo, ref)
    evaluator = build_evaluator(project, conn, use_cache=use_cache)
    rows = evaluator.rows(partition)
    if sample is not None:
        rows = sample_rows(rows, sample, evaluator.partitions.seed)
    tag = RunTag("eval", {"sampled": sample is not None})
    record = await evaluator.evaluate(version, partition, tag, rows=rows, progress=progress)
    return version, record


def _noise(records: Sequence[EvaluationRecord]) -> float | None:
    values = [r.report.primary_value.value for r in records]
    known = [v for v in values if v is not None]
    if len(records) < 2 or len(known) != len(values):
        return None
    return max(known) - min(known)
