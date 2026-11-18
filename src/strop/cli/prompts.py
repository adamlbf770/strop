"""The ``strop prompt`` command group."""

from pathlib import Path
from typing import Annotated

import typer

from strop.cli.options import DirOption, JsonOption, emit_json
from strop.prompts.versions import PromptVersion
from strop.services.project import Project, open_project
from strop.services.prompts import add_version_from_file, load_project_prompt, resolve_version
from strop.storage.prompts import PromptRepository

prompt_app = typer.Typer(help="Add and inspect prompt versions.", no_args_is_help=True)


@prompt_app.command("add")
def add_command(
    file: Annotated[Path, typer.Option(help="Prompt file holding the edited prompt.")],
    parent: Annotated[
        str | None, typer.Option(help="Version it derives from; default: best.")
    ] = None,
    reason: Annotated[str, typer.Option(help="Why this edit was made.")] = "",
    use: Annotated[bool, typer.Option("--use", help="Make this version the best prompt.")] = False,
    directory: DirOption = Path(),
) -> None:
    """Save an edited prompt file as a new immutable version."""
    project = open_project(directory)
    with project.database() as conn:
        repo = PromptRepository(conn)
        version = add_version_from_file(project, repo, file, parent=parent, reason=reason, use=use)
    typer.echo(f"Saved {version.ref} (parent v{version.parent_number})")
    if use:
        typer.echo(f"{version.ref} is now the best prompt.")


@prompt_app.command("show")
def show_command(
    version: Annotated[
        str | None, typer.Argument(help="Version such as v2; default: best.")
    ] = None,
    as_json: JsonOption = False,
    directory: DirOption = Path(),
) -> None:
    """Print the messages of a prompt version."""
    project = open_project(directory)
    with project.database() as conn:
        repo = PromptRepository(conn)
        if version is None and repo.first() is None:
            _show_unregistered(project, as_json)
            return
        selected = resolve_version(repo, version)
        is_best = repo.best_number() == selected.number
    if as_json:
        emit_json(_payload(selected, is_best=is_best))
        return
    typer.echo(_header(selected, is_best=is_best))
    for message in selected.template.messages:
        editable = "editable" if selected.template.is_editable(message) else "frozen"
        typer.echo(f"--- {message.role} ({editable})")
        typer.echo(message.content.rstrip("\n"))


def _show_unregistered(project: Project, as_json: bool) -> None:
    """Before the first run there are no versions yet, so show the prompt file itself."""
    template = load_project_prompt(project)
    file = project.config.prompt.file
    if as_json:
        emit_json(
            {
                "version": None,
                "file": str(file),
                "messages": [
                    {"role": m.role, "content": m.content, "editable": template.is_editable(m)}
                    for m in template.messages
                ],
            }
        )
        return
    typer.echo(f"{file} (not registered yet; it becomes v1 on the first run)")
    for message in template.messages:
        editable = "editable" if template.is_editable(message) else "frozen"
        typer.echo(f"--- {message.role} ({editable})")
        typer.echo(message.content.rstrip("\n"))


def _header(version: PromptVersion, *, is_best: bool) -> str:
    parent = f"parent v{version.parent_number}" if version.parent_number else "no parent"
    best = ", best" if is_best else ""
    summary = f"\nSummary: {version.change_summary}" if version.change_summary else ""
    return f"{version.ref} ({parent}, {version.source}{best}){summary}"


def _payload(version: PromptVersion, *, is_best: bool) -> dict[str, object]:
    return {
        "version": version.number,
        "parent": version.parent_number,
        "source": version.source,
        "best": is_best,
        "change_summary": version.change_summary,
        "reason": version.reason,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "editable": version.template.is_editable(m),
            }
            for m in version.template.messages
        ],
    }
