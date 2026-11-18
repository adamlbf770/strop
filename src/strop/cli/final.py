"""The ``test`` and ``compare`` commands."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from strop.cli.evaluate import EvalPartition, NoCacheOption, SampleOption, ShowErrorsOption
from strop.cli.options import DirOption, JsonOption, emit_json
from strop.cli.progress import stderr_progress
from strop.data.models import Partition
from strop.reporting import ColorMode, make_console
from strop.reporting.evaluation_view import render_evaluation
from strop.reporting.final_view import (
    comparison_payload,
    final_test_payload,
    render_comparison,
    render_final_test,
)
from strop.services.compare import parse_models, run_compare
from strop.services.final_test import run_final_test
from strop.services.project import open_project
from strop.storage.evaluations import EvaluationRepository


def final_test_command(
    version: Annotated[
        str | None, typer.Argument(help="Version such as v2; default: best.")
    ] = None,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Acknowledge that this uses up the test set.")
    ] = False,
    allow_rerun: Annotated[
        bool, typer.Option("--allow-rerun", help="Allow a second test evaluation.")
    ] = False,
    with_baseline: Annotated[
        bool, typer.Option("--with-baseline", help="Also test the initial prompt (v1).")
    ] = False,
    no_cache: NoCacheOption = False,
    show_errors: ShowErrorsOption = 5,
    as_json: JsonOption = False,
    directory: DirOption = Path(),
) -> None:
    """Run the final evaluation on the held-out test partition.

    The test set is only a fair estimate if it is looked at once. Pass --confirm to proceed.
    """
    project = open_project(directory)
    with project.database() as conn:
        result = asyncio.run(
            run_final_test(
                project,
                conn,
                ref=version,
                confirm=confirm,
                allow_rerun=allow_rerun,
                with_baseline=with_baseline,
                use_cache=not no_cache,
                progress=None if as_json else stderr_progress("Testing"),
            )
        )
        outcomes = EvaluationRepository(conn).outcomes(result.record.id)
    if as_json:
        emit_json(final_test_payload(result))
        return
    console = make_console(ColorMode.AUTO)
    render_evaluation(
        console, result.record, outcomes, version_ref=result.version.ref, show_errors=show_errors
    )
    render_final_test(console, result)


def compare_command(
    models: Annotated[str, typer.Option(help="Comma-separated target models to compare.")],
    version: Annotated[
        str | None, typer.Argument(help="Version such as v2; default: best.")
    ] = None,
    partition: Annotated[EvalPartition, typer.Option(help="Partition to evaluate.")] = (
        EvalPartition.VALIDATION
    ),
    sample: SampleOption = None,
    no_cache: NoCacheOption = False,
    as_json: JsonOption = False,
    directory: DirOption = Path(),
) -> None:
    """Run one prompt version on several models and compare the results."""
    names = parse_models(models)
    project = open_project(directory)
    with project.database() as conn:
        comparison = asyncio.run(
            run_compare(
                project,
                conn,
                models=names,
                ref=version,
                partition=Partition(partition.value),
                sample=sample,
                use_cache=not no_cache,
                progress=None if as_json else stderr_progress("Comparing"),
            )
        )
    if as_json:
        emit_json(comparison_payload(comparison))
        return
    render_comparison(make_console(ColorMode.AUTO), comparison)
