import os
from pathlib import Path

import pytest

from strop.env import load_env_file
from strop.services.project import open_project
from tests.helpers import CliRunner


def unset(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make sure the variables are absent now and are removed again after the test."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_variables_are_loaded_in_the_supported_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unset(monkeypatch, "STROP_A", "STROP_B", "STROP_C", "STROP_D", "STROP_E")
    path = write(
        tmp_path,
        "# a comment\n"
        "STROP_A=plain\n"
        "export STROP_B = spaced value\n"
        'STROP_C="double quoted # not a comment"\n'
        "STROP_D='single quoted'\n"
        "STROP_E=value # trailing comment\n",
    )
    assert load_env_file(path) == ["STROP_A", "STROP_B", "STROP_C", "STROP_D", "STROP_E"]
    assert os.environ["STROP_A"] == "plain"
    assert os.environ["STROP_B"] == "spaced value"
    assert os.environ["STROP_C"] == "double quoted # not a comment"
    assert os.environ["STROP_D"] == "single quoted"
    assert os.environ["STROP_E"] == "value"


def test_existing_variables_are_not_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STROP_KEEP", "from the shell")
    assert load_env_file(write(tmp_path, "STROP_KEEP=from the file\n")) == []
    assert os.environ["STROP_KEEP"] == "from the shell"


def test_invalid_lines_and_missing_files_are_ignored(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent.env") == []
    assert load_env_file(write(tmp_path, "no equals sign\n1BAD=x\n=novalue\n\n")) == []


def test_opening_a_project_loads_its_env_file(
    run_cli: CliRunner, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unset(monkeypatch, "STROP_FROM_PROJECT")
    (project / ".env").write_text("STROP_FROM_PROJECT=yes\n", encoding="utf-8")
    open_project(project)
    assert os.environ["STROP_FROM_PROJECT"] == "yes"
