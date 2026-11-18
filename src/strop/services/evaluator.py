"""Evaluating prompt versions on dataset partitions and storing the results."""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from strop.config.enums import ModelRole
from strop.data.models import Partition, Row
from strop.errors import DataError
from strop.evaluation.budget import BudgetTracker
from strop.evaluation.examples import to_examples, with_scores
from strop.evaluation.models import EvaluationRecord, NewEvaluation
from strop.evaluation.parsing import load_output_schema
from strop.evaluation.runner import Progress, PromptRunner
from strop.llm.client import LLMClient
from strop.llm.factory import build_client
from strop.metrics.suite import MetricSuite
from strop.prompts.versions import PromptVersion
from strop.services.metrics import build_suite
from strop.services.partitions import PartitionedData, load_partitions
from strop.services.project import Project
from strop.storage.cache import ResponseCache
from strop.storage.datasets import DatasetRepository
from strop.storage.evaluations import EvaluationRepository


@dataclass(frozen=True)
class RunTag:
    """Labels an evaluation: what it is for, free-form details, and its optimization run."""

    kind: str
    details: dict[str, Any] = field(default_factory=dict)
    run_id: int | None = None


@dataclass(frozen=True)
class Evaluator:
    """Runs prompt versions against partitions and persists each evaluation.

    The test partition is refused unless the evaluator was created with ``allow_test``;
    only the final ``strop test`` step does that.
    """

    project: Project
    partitions: PartitionedData
    suite: MetricSuite
    client: LLMClient
    repository: EvaluationRepository
    budget: BudgetTracker
    schema: dict[str, Any] | None = None
    allow_test: bool = False

    def rows(self, partition: Partition) -> list[Row]:
        if partition is Partition.TEST and not self.allow_test:
            raise DataError("the test partition is locked; it is only used by 'strop test'")
        return self.partitions.get(partition)

    async def evaluate(
        self,
        version: PromptVersion,
        partition: Partition,
        tag: RunTag,
        *,
        rows: Sequence[Row] | None = None,
        progress: Progress | None = None,
    ) -> EvaluationRecord:
        """Evaluate ``version`` on ``partition`` (or a subset of its rows) and store the result."""
        selected = list(rows) if rows is not None else self.rows(partition)
        if rows is not None:
            self.rows(partition)  # enforces the test lock for explicit subsets too
        config = self.project.config
        runner = PromptRunner(
            self.client,
            config.models.resolve(ModelRole.TARGET),
            config.output,
            self.schema,
            config.execution,
            self.budget,
        )
        outcomes = await runner.run(version.template, selected, progress)
        examples = to_examples(selected, outcomes)
        report = self.suite.evaluate(examples)
        return self.repository.save(
            NewEvaluation(
                prompt_number=version.number,
                split_id=self.partitions.split_id,
                partition=partition,
                kind=tag.kind,
                report=report,
                outcomes=with_scores(outcomes, examples, self.suite),
                details={"model": config.models.target.model, **tag.details},
                run_id=tag.run_id,
            )
        )


def build_evaluator(
    project: Project,
    conn: sqlite3.Connection,
    *,
    use_cache: bool = True,
    allow_test: bool = False,
    budget: BudgetTracker | None = None,
) -> Evaluator:
    """Assemble an evaluator from the project configuration and an open database."""
    config = project.config
    cache = ResponseCache(conn) if use_cache else None
    return Evaluator(
        project=project,
        partitions=load_partitions(project, DatasetRepository(conn)),
        suite=build_suite(project),
        client=build_client(config.execution, cache),
        repository=EvaluationRepository(conn),
        budget=budget or BudgetTracker(config.budgets),
        schema=load_output_schema(project.root, config.output),
        allow_test=allow_test,
    )
