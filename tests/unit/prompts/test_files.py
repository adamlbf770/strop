from pathlib import Path

import pytest

from strop.errors import ConfigError
from strop.prompts import PromptMessage, PromptTemplate, dump_prompt, load_prompt_file, parse_prompt

YAML_PROMPT = """\
messages:
  - role: system
    content: |
      You classify tickets.
  - role: user
    content: "Text: {{text}}"
optimize: [system]
"""


def test_yaml_prompt_is_parsed() -> None:
    template = parse_prompt(YAML_PROMPT, ".yaml")
    assert [m.role for m in template.messages] == ["system", "user"]
    assert template.messages[0].content == "You classify tickets.\n"
    assert template.optimize == {"system"}
    assert template.variables == ("text",)


def test_optimize_defaults_to_system_when_present() -> None:
    text = "messages:\n  - {role: system, content: a}\n  - {role: user, content: b}\n"
    assert parse_prompt(text, ".yaml").optimize == {"system"}


def test_optimize_defaults_to_user_without_a_system_message() -> None:
    assert parse_prompt("messages:\n  - {role: user, content: b}\n", ".yml").optimize == {"user"}


def test_plain_text_prompt_is_one_editable_user_message() -> None:
    template = parse_prompt("Classify: {{text}}\n", ".md")
    assert len(template.messages) == 1
    assert template.optimize == {"user"}


@pytest.mark.parametrize(
    ("text", "suffix", "message"),
    [
        ("", ".md", "empty"),
        ("messages: []\n", ".yaml", "messages"),
        ("messages:\n  - {role: bot, content: a}\n", ".yaml", "role"),
        ("messages:\n  - {role: user, content: ''}\n", ".yaml", "content"),
        ("messages:\n  - {role: user, content: a}\noptimize: [system]\n", ".yaml", "optimize"),
        ("messages:\n  - {role: user, content: a}\nextra: 1\n", ".yaml", "extra"),
        ("messages: [unclosed", ".yaml", "not valid YAML"),
        ("a <strop:fixed>open", ".md", "marker"),
    ],
)
def test_invalid_prompts_are_rejected(text: str, suffix: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_prompt(text, suffix)


def test_dump_and_parse_round_trip() -> None:
    template = PromptTemplate(
        (
            PromptMessage("system", "Line one.\nLine two: <strop:fixed>keep</strop:fixed>"),
            PromptMessage("user", "Text: {{text}}"),
        ),
        frozenset({"system"}),
    )
    parsed = parse_prompt(dump_prompt(template), ".yaml")
    assert [m.content.strip() for m in parsed.messages] == [m.content for m in template.messages]
    assert parsed.optimize == template.optimize


def test_load_prompt_file_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_prompt_file(tmp_path / "absent.yaml")


def test_json_round_trip_and_stable_hash() -> None:
    template = parse_prompt(YAML_PROMPT, ".yaml")
    again = PromptTemplate.from_json(template.to_json())
    assert again == template
    assert again.content_hash == template.content_hash
    other = parse_prompt(YAML_PROMPT.replace("classify", "label"), ".yaml")
    assert other.content_hash != template.content_hash
