"""Top-level commands about prompt history: ``diff``, ``history``, ``rollback``."""

from pathlib import Path
from typing import Annotated

import typer

from strop.cli.options import DirOption, JsonOption, emit_json
from strop.errors import ConfigError
from strop.prompts.diff import diff_templates
from strop.prompts.versions import PromptVersion
from strop.reporting import ColorMode, make_console, render_diff
from strop.services.project import open_project
from strop.services.prompts import resolve_version
from strop.storage.prompts import PromptRepository

SUMMARY_WIDTH = 60


def diff_command(
    old: Annotated[str | None, typer.Argument(help="Older version, such as v1.")] = None,
    new: Annotated[str | None, typer.Argument(help="Newer version; default: best.")] = None,
    color: Annotated[ColorMode, typer.Option(help="When to use color.")] = ColorMode.AUTO,
    as_json: JsonOption = False,
    directory: DirOption = Path(),
) -> None:
    """Show a colored diff of two prompt versions.

    With no arguments: the first prompt (v1) against the best prompt. With one argument:
    that version against its parent.
    """
    project = open_project(directory)
    with project.database() as conn:
        repo = PromptRepository(conn)
        before, after = _pick_pair(repo, old, new)
    diff = diff_templates(before.template, after.template)
    if as_json:
        emit_json({"from": before.ref, "to": after.ref, **diff.to_payload()})
        return
    console = make_console(color)
    console.print(f"{before.ref} -> {after.ref}", soft_wrap=True, highlight=False)
    render_diff(diff, console)


def history_command(
    as_json: JsonOption = False,
    directory: DirOption = Path(),
) -> None:
    """List every prompt version, oldest first. The best prompt is marked with *."""
    project = open_project(directory)
    with project.database() as conn:
        repo = PromptRepository(conn)
        versions = repo.list()
        best = repo.best_number()
    if as_json:
        emit_json({"best": best, "versions": [_row(v) for v in versions]})
        return
    if not versions:
        typer.echo("No prompt versions yet. The prompt file is registered as v1 on the first run.")
        return
    for version in versions:
        marker = "*" if version.number == best else " "
        parent = f"v{version.parent_number}" if version.parent_number else "-"
        summary = version.change_summary[:SUMMARY_WIDTH]
        typer.echo(f"{marker} {version.ref:<5} {version.source:<10} parent {parent:<5} {summary}")


def rollback_command(
    version: Annotated[str, typer.Argument(help="Version to make the best prompt.")],
    directory: DirOption = Path(),
) -> None:
    """Make an earlier version the best prompt. No version is deleted or changed."""
    project = open_project(directory)
    with project.database() as conn:
        repo = PromptRepository(conn)
        target = resolve_version(repo, version)
        previous = repo.best_number()
        repo.set_best(target.number)
    was = f" (was v{previous})" if previous else ""
    typer.echo(f"Best prompt is now {target.ref}{was}. No versions were deleted.")


def _pick_pair(
    repo: PromptRepository, old: str | None, new: str | None
) -> tuple[PromptVersion, PromptVersion]:
    if old is None:
        first = repo.first()
        if first is None:
            raise ConfigError("no prompt versions yet; run 'strop baseline' first")
        return first, resolve_version(repo, None)
    if new is not None:
        return resolve_version(repo, old), resolve_version(repo, new)
    selected = resolve_version(repo, old)
    if selected.parent_number is None:
        raise ConfigError(f"{selected.ref} has no parent; name two versions to compare")
    return repo.get(selected.parent_number), selected


def _row(version: PromptVersion) -> dict[str, object]:
    return {
        "version": version.number,
        "parent": version.parent_number,
        "source": version.source,
        "change_summary": version.change_summary,
        "reason": version.reason,
        "created_at": version.created_at,
    }
