"""Command-line options shared by several commands."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

DirOption = Annotated[Path, typer.Option("--dir", help="Project directory.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]
SkipInvalidOption = Annotated[
    bool,
    typer.Option("--skip-invalid", help="Exclude invalid dataset rows instead of failing."),
]


def emit_json(payload: Mapping[str, Any]) -> None:
    """Print ``payload`` as the command's single JSON document on stdout."""
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
