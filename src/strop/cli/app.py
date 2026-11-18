"""The Typer application and the process entry point."""

import sys
from typing import Annotated

import typer

from strop import __version__
from strop.cli.data import data_app
from strop.cli.init import init_command
from strop.errors import ExitCode, StropError

app = typer.Typer(
    name="strop",
    help="Improve LLM prompts by testing them against your own data.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.command("init")(init_command)
app.add_typer(data_app, name="data")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"strop {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Improve LLM prompts by testing them against your own data."""


def main() -> None:
    """Run the CLI and translate expected errors into exit codes."""
    try:
        app()
    except StropError as exc:
        typer.echo(f"error: {exc}", err=True)
        sys.exit(exc.exit_code)
    except KeyboardInterrupt:
        typer.echo("interrupted", err=True)
        sys.exit(ExitCode.INTERRUPTED)
