"""Running one prompt version on several target models."""

import sqlite3
from dataclasses import dataclass, replace

from strop.data.models import Partition
from strop.errors import ConfigError
from strop.evaluation.models import EvaluationRecord
from strop.evaluation.runner import Progress
from strop.prompts.versions import PromptVersion
from strop.services.evaluate import sample_rows
from strop.services.evaluator import RunTag, build_evaluator
from strop.services.project import Project
from strop.services.prompts import ensure_initial_version, resolve_version
from strop.storage.prompts import PromptRepository


@dataclass(frozen=True)
class Comparison:
    version: PromptVersion
    partition: Partition
    results: list[tuple[str, EvaluationRecord]]


def parse_models(text: str) -> list[str]:
    """Split a comma-separated list of model names, rejecting empty and repeated entries."""
    names = [item.strip() for item in text.split(",") if item.strip()]
    if len(names) < 2:
        raise ConfigError("--models needs at least two comma-separated model names")
    if len(set(names)) != len(names):
        raise ConfigError("--models lists the same model more than once")
    return names


async def run_compare(  # noqa: PLR0913
    project: Project,
    conn: sqlite3.Connection,
    *,
    models: list[str],
    ref: str | None,
    partition: Partition,
    sample: int | None,
    use_cache: bool,
    progress: Progress | None = None,
) -> Comparison:
    """Evaluate the same version on the same rows with each model as the target."""
    repo = PromptRepository(conn)
    ensure_initial_version(project, repo)
    version = resolve_version(repo, ref)
    base = build_evaluator(project, conn, use_cache=use_cache)
    rows = base.rows(partition)
    if sample is not None:
        rows = sample_rows(rows, sample, base.partitions.seed)
    results = []
    for name in models:
        evaluator = replace(base, target=base.target.model_copy(update={"model": name}))
        tag = RunTag("compare", {"sampled": sample is not None})
        record = await evaluator.evaluate(version, partition, tag, rows=rows, progress=progress)
        results.append((name, record))
    return Comparison(version, partition, results)
