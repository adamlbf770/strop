"""Documentation must stay true: commands exist, samples validate, and no banned characters."""

import re
from pathlib import Path

import pytest
from ruamel.yaml import YAML

import strop
from strop.cli import app
from strop.config import StropConfig
from strop.errors import ExitCode

ROOT = Path(strop.__file__).resolve().parents[2]
DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    *(ROOT / "docs").glob("*.md"),
]
pytestmark = pytest.mark.skipif(
    not (ROOT / "docs").is_dir(), reason="docs are not in this checkout"
)


def command_names() -> set[str]:
    names = {c.name or c.callback.__name__ for c in app.registered_commands}  # type: ignore[union-attr]
    return names | {g.name for g in app.registered_groups if g.name}


def fenced(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, re.DOTALL)


def test_documents_contain_no_em_or_en_dashes() -> None:
    banned = (chr(0x2014), chr(0x2013))
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        assert not any(char in text for char in banned), path.name


def test_every_strop_command_in_the_docs_exists() -> None:
    known = command_names()
    for path in DOCS:
        for block in fenced(path.read_text(encoding="utf-8"), "(?:bash|console)"):
            for line in block.splitlines():
                match = re.match(r"^(?:\$ )?strop ([a-z]+)", line.strip())
                if match:
                    assert match.group(1) in known, f"{path.name}: strop {match.group(1)}"


def test_the_complete_configuration_example_is_valid() -> None:
    text = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    sample = fenced(text, "yaml")[0]
    data = YAML(typ="safe").load(sample)
    config = StropConfig.model_validate(data)
    assert config.metrics.primary == "macro_f1"
    assert config.metrics.constraints["recall:refund"].min == 0.8


def test_the_readme_configuration_example_is_valid() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    sample = next(b for b in fenced(text, "yaml") if "project:" in b)
    StropConfig.model_validate(YAML(typ="safe").load(sample))


def test_the_readme_exit_codes_match_the_code() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    documented = {int(code) for code in re.findall(r"^\| (\d+) \|", text, re.MULTILINE)}
    assert documented == {int(code) for code in ExitCode}


def test_every_configuration_key_is_documented() -> None:
    text = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    for section, model in StropConfig.model_fields.items():
        assert f"## {section}" in text, section
        sub = model.annotation
        fields = getattr(sub, "model_fields", {})
        for key in fields:
            if key in {"model", "temperature", "max_tokens", "timeout_s", "params"}:
                continue
            assert f"`{key}`" in text, f"{section}.{key}"
