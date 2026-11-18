import sys
from pathlib import Path

import pytest

from strop.cli import main
from tests.helpers import CliRunner, write_dataset


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> CliRunner:
    """Run the real entry point and return (exit code, stdout, stderr)."""

    def run(*args: str) -> tuple[int, str, str]:
        monkeypatch.setattr(sys, "argv", ["strop", *args])
        code = 0
        try:
            main()
        except SystemExit as exc:
            code = int(exc.code or 0)
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return run


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A directory containing a small labeled dataset."""
    (tmp_path / "data.csv").write_text(
        "message,category\nwhere is my refund,refund\ncharge twice,billing\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def init_args(project_dir: Path) -> list[str]:
    """A complete, valid argument list for ``strop init``."""
    return [
        "init",
        "--dir", str(project_dir),
        "--name", "demo",
        "--task", "Classify a customer message.",
        "--data", "data.csv",
        "--input", "message",
        "--expected", "category",
        "--output-type", "label",
        "--labels", "billing,refund",
        "--metric", "macro_f1",
        "--target-model", "openai/model-a",
        "--optimizer-model", "anthropic/model-b",
    ]  # fmt: skip


@pytest.fixture
def project(run_cli: CliRunner, init_args: list[str], project_dir: Path) -> Path:
    """An initialized project with a 100-row dataset."""
    write_dataset(project_dir)
    assert run_cli(*init_args)[0] == 0
    return project_dir
