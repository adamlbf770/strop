"""The ``baseline`` and ``eval`` commands."""

import asyncio
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from strop.cli.options import DirOption, JsonOption, emit_json
from strop.cli.progress import stderr_progress
from strop.data.models import Partition
from strop.evaluation.models import EvaluationRecord
from strop.formatting import fmt
from strop.reporting import ColorMode, make_console
from strop.reporting.evaluation_view import record_payload, render_evaluation
from strop.services.evaluate import BaselineOutcome, run_baseline, run_eval
from strop.services.project import Project, open_project
from strop.services.prompts import has_unregistered_edits
from strop.storage.evaluations import EvaluationRepository
from strop.storage.prompts import PromptRepository

NoCacheOption = Annotated[bool, typer.Option("--no-cache", help="Do not read or write the cache.")]
SampleOption = Annotated[
    int | None, typer.Option(help="Use a seeded random sample of this many rows.", min=1)
]
ShowErrorsOption = Annotated[int, typer.Option(help="How many failing examples to list.", min=0)]


class EvalPartition(StrEnum):
    """Partitions that may be evaluated freely. The test partition is not among them."""

    TRAIN = "train"
    VALIDATION = "validation"


def baseline_command(
    repeats: Annotated[int, typer.Option(help="Run the baseline this many times.", min=1)] = 1,
    sample: SampleOption = None,
    with_train: Annotated[bool, typer.Option(help="Also evaluate on the train partition.")] = False,
    no_cache: NoCacheOption = False,
    show_errors: ShowErrorsOption = 5,
    as_json: JsonOption = False,
    directory: DirOption = Path(),
) -> None:
    """Measure the initial prompt (v1) on the validation partition."""
    project = open_project(directory)
    with project.database() as conn:
        progress = None if as_json else stderr_progress("Baseline")
        outcome = asyncio.run(
            run_baseline(
                project,
                conn,
                repeats=repeats,
                sample=sample,
                with_train=with_train,
                use_cache=not no_cache,
                progress=progress,
            )
        )
        repo = EvaluationRepository(conn)
        _warn_about_edits(project, PromptRepository(conn))
        if as_json:
            emit_json(_baseline_payload(outcome))
            return
        console = make_console(ColorMode.AUTO)
        for record in [outcome.reference, *([outcome.train] if outcome.train else [])]:
            render_evaluation(
                console,
                record,
                repo.outcomes(record.id),
                version_ref=outcome.version.ref,
                show_errors=show_errors,
            )
    if outcome.noise is not None:
        typer.echo(
            f"Run-to-run noise over {repeats} runs: the primary metric varied by "
            f"{fmt(outcome.noise)}. Improvements smaller than this are not meaningful."
        )


def eval_command(
    version: Annotated[
        str | None, typer.Argument(help="Version such as v2; default: best.")
    ] = None,
    partition: Annotated[EvalPartition, typer.Option(help="Partition to evaluate.")] = (
        EvalPartition.VALIDATION
    ),
    sample: SampleOption = None,
    no_cache: NoCacheOption = False,
    show_errors: ShowErrorsOption = 5,
    as_json: JsonOption = False,
    directory: DirOption = Path(),
) -> None:
    """Score a prompt version on a partition. Nothing is changed or promoted."""
    project = open_project(directory)
    with project.database() as conn:
        selected, record = asyncio.run(
            run_eval(
                project,
                conn,
                ref=version,
                partition=Partition(partition.value),
                sample=sample,
                use_cache=not no_cache,
                progress=None if as_json else stderr_progress("Evaluating"),
            )
        )
        outcomes = EvaluationRepository(conn).outcomes(record.id)
        if version is None:
            _warn_about_edits(project, PromptRepository(conn))
    if as_json:
        emit_json(record_payload(record, selected.ref))
        return
    render_evaluation(
        make_console(ColorMode.AUTO),
        record,
        outcomes,
        version_ref=selected.ref,
        show_errors=show_errors,
    )


def _warn_about_edits(project: Project, repo: PromptRepository) -> None:
    if has_unregistered_edits(project, repo):
        typer.echo(
            f"Warning: {project.config.prompt.file} differs from every stored prompt version. "
            f"Run 'strop prompt add --file {project.config.prompt.file} --use' to use your edits.",
            err=True,
        )


def _baseline_payload(outcome: BaselineOutcome) -> dict[str, object]:
    ref = outcome.version.ref

    def one(record: EvaluationRecord) -> dict[str, object]:
        return dict(record_payload(record, ref))

    return {
        "version": ref,
        "baseline": one(outcome.reference),
        "repeats": [one(r) for r in outcome.records],
        "noise": outcome.noise,
        "train": one(outcome.train) if outcome.train else None,
    }
