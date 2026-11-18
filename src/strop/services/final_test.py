"""The final evaluation on the held-out test partition."""

import sqlite3
from dataclasses import dataclass

from strop.data.models import Partition
from strop.errors import ConfigError, DataError
from strop.evaluation.models import EvaluationRecord
from strop.evaluation.runner import Progress
from strop.prompts.versions import PromptVersion
from strop.services.evaluator import RunTag, build_evaluator
from strop.services.project import Project
from strop.services.prompts import ensure_initial_version, resolve_version
from strop.storage.prompts import PromptRepository

CONFIRM_MESSAGE = (
    "the test partition is the final check. Evaluating it and then changing the prompt makes "
    "the result an optimistic estimate. Run again with --confirm to continue"
)


@dataclass(frozen=True)
class FinalTest:
    version: PromptVersion
    record: EvaluationRecord
    validation: EvaluationRecord | None
    baseline: tuple[PromptVersion, EvaluationRecord] | None


async def run_final_test(  # noqa: PLR0913
    project: Project,
    conn: sqlite3.Connection,
    *,
    ref: str | None,
    confirm: bool,
    allow_rerun: bool,
    with_baseline: bool,
    use_cache: bool,
    progress: Progress | None = None,
) -> FinalTest:
    """Evaluate a prompt on the test partition, guarded by explicit confirmation."""
    if not confirm:
        raise ConfigError(CONFIRM_MESSAGE)
    prompts = PromptRepository(conn)
    ensure_initial_version(project, prompts)
    version = resolve_version(prompts, ref)
    evaluator = build_evaluator(project, conn, use_cache=use_cache, allow_test=True)
    if not evaluator.rows(Partition.TEST):
        raise DataError("there is no test partition; create one with 'strop data split --test 0.2'")
    used = [
        e
        for e in evaluator.repository.find(partition=Partition.TEST, limit=50)
        if e.split_id == evaluator.partitions.split_id
    ]
    if used and not allow_rerun:
        raise ConfigError(
            f"the test partition was already used (evaluation {used[-1].id}, "
            f"v{used[-1].prompt_number}). Testing again after changing the prompt biases the "
            "result; pass --allow-rerun to continue anyway"
        )
    record = await evaluator.evaluate(version, Partition.TEST, RunTag("test"), progress=progress)
    baseline = None
    initial = prompts.first()
    if with_baseline and initial is not None and initial.number != version.number:
        baseline_record = await evaluator.evaluate(
            initial, Partition.TEST, RunTag("test", {"role": "baseline"}), progress=progress
        )
        baseline = (initial, baseline_record)
    return FinalTest(version, record, evaluator.reusable(version, Partition.VALIDATION), baseline)
