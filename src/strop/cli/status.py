"""The ``strop status`` command."""

from pathlib import Path

import typer

from strop.cli.options import DirOption, JsonOption, emit_json
from strop.formatting import fmt
from strop.services.project import open_project
from strop.storage.datasets import DatasetRepository
from strop.storage.evaluations import EvaluationRepository
from strop.storage.prompts import PromptRepository

RECENT = 5


def status_command(as_json: JsonOption = False, directory: DirOption = Path()) -> None:
    """Show the dataset split, the best prompt, and the latest evaluations."""
    project = open_project(directory)
    with project.database() as conn:
        datasets = DatasetRepository(conn)
        split = datasets.latest_split()
        counts = datasets.partition_counts(split.id) if split else {}
        prompts = PromptRepository(conn)
        versions = prompts.list()
        best = prompts.best_number()
        recent = EvaluationRepository(conn).find(limit=RECENT)
    payload = {
        "project": project.config.project.name,
        "split": (
            {"id": split.id, "seed": split.seed, "counts": {p.value: n for p, n in counts.items()}}
            if split
            else None
        ),
        "prompt_versions": len(versions),
        "best_version": best,
        "recent_evaluations": [
            {
                "id": e.id,
                "version": e.prompt_number,
                "partition": e.partition.value,
                "kind": e.kind,
                "primary_metric": e.report.primary,
                "value": e.report.primary_value.value,
                "samples": e.sample_count,
            }
            for e in recent
        ],
    }
    if as_json:
        emit_json(payload)
        return
    typer.echo(f"Project: {project.config.project.name}")
    if split:
        sizes = ", ".join(f"{p.value} {n}" for p, n in counts.items())
        typer.echo(f"Split {split.id} (seed {split.seed}): {sizes}")
    else:
        typer.echo("Split: none yet (run 'strop data split')")
    if versions:
        typer.echo(f"Prompt: v{best} is the best of {len(versions)} version(s)")
    else:
        typer.echo("Prompt: no versions yet (registered on the first run)")
    for evaluation in recent:
        typer.echo(
            f"  #{evaluation.id} v{evaluation.prompt_number} {evaluation.partition.value} "
            f"{evaluation.kind}: {evaluation.report.primary}="
            f"{fmt(evaluation.report.primary_value.value)} ({evaluation.sample_count} examples)"
        )
