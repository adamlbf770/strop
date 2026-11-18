from pathlib import Path

import pytest

from strop.config.rules import load_rubrics, parse_rules
from strop.errors import ConfigError


def test_parses_each_kind_of_rule() -> None:
    rules = parse_rules(
        [
            "short=max_words:120",
            "polite=not_matches:(?i)stupid|idiot",
            "needs=contains_all:refund, days",
        ]
    )
    assert rules["short"].value == 120
    assert rules["polite"].value == "(?i)stupid|idiot"
    assert rules["needs"].value == ["refund", "days"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("short", "invalid rule"),
        ("short=max_words", "invalid rule"),
        ("=max_words:5", "invalid rule"),
        ("short=max_words:", "invalid rule"),
        ("short=shortest:5", "unknown rule type 'shortest'"),
        ("short=max_words:many", "whole number"),
        ("bad=matches:(unclosed", "invalid regular expression"),
    ],
)
def test_invalid_rules_are_explained(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_rules([text])


def test_duplicate_rule_names_are_rejected() -> None:
    with pytest.raises(ConfigError, match="defined more than once"):
        parse_rules(["a=max_words:1", "a=min_words:1"])


def test_rubrics_are_loaded_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "rubric.yaml"
    path.write_text(
        "helpful:\n  criteria: Is it helpful?\n  scale_max: 10\ntone:\n  criteria: Is it polite?\n",
        encoding="utf-8",
    )
    rubrics = load_rubrics(path)
    assert rubrics["helpful"].scale_max == 10
    assert rubrics["tone"].scale_max == 5


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- not a mapping\n", "must map each rubric name"),
        ("helpful: {scale_max: 5}\n", "criteria"),
        ("helpful: {criteria: x, scale_min: 5, scale_max: 1}\n", "scale_max"),
        ("a: [unclosed", "not valid YAML"),
    ],
)
def test_invalid_rubric_files_are_explained(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "rubric.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_rubrics(path)


def test_missing_rubric_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read the rubric file"):
        load_rubrics(tmp_path / "absent.yaml")
