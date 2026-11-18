from pathlib import Path
from typing import Any

import pytest

from strop.config.output import OutputConfig
from strop.errors import ConfigError
from strop.evaluation.parsing import (
    canonical_expected,
    load_output_schema,
    parse_output,
)
from strop.metrics.types import ErrorCode

LABELS = ("billing", "refund")


def label_output(match: str = "strict") -> OutputConfig:
    return OutputConfig.model_validate({"type": "label", "labels": LABELS, "match": match})


def json_output(**extra: Any) -> OutputConfig:
    return OutputConfig.model_validate({"type": "json", **extra})


@pytest.mark.parametrize(
    "raw",
    ["refund", "  Refund \n", "REFUND", '"refund"', "`refund`", "refund.", "'Refund'", "“refund”"],
)
def test_label_is_matched_after_light_cleanup(raw: str) -> None:
    assert parse_output(raw, "stop", label_output()).value == "refund"


@pytest.mark.parametrize(
    "raw", ["Category: refund", "refund or billing", "other", "I think refund"]
)
def test_strict_label_rejects_extra_text(raw: str) -> None:
    result = parse_output(raw, "stop", label_output())
    assert result.value is None
    assert result.error is ErrorCode.INVALID_FORMAT


def test_contains_mode_accepts_one_whole_word_label() -> None:
    output = label_output("contains")
    assert parse_output("Category: refund", "stop", output).value == "refund"
    assert parse_output("refunded", "stop", output).value is None
    assert parse_output("refund or billing", "stop", output).value is None


def test_empty_and_truncated_outputs_have_their_own_codes() -> None:
    assert parse_output("   ", "stop", label_output()).error is ErrorCode.EMPTY_OUTPUT
    assert parse_output("refu", "length", label_output()).error is ErrorCode.TRUNCATED


def test_json_is_parsed_and_canonicalized() -> None:
    result = parse_output('{"b": 1, "a": "x"}', "stop", json_output())
    assert result.value == '{"a": "x", "b": 1}'
    assert result.parsed == {"b": 1, "a": "x"}


def test_json_inside_a_code_fence_is_accepted() -> None:
    result = parse_output('```json\n{"a": 1}\n```', "stop", json_output())
    assert result.parsed == {"a": 1}


def test_invalid_or_truncated_json() -> None:
    assert parse_output("{not json", "stop", json_output()).error is ErrorCode.INVALID_FORMAT
    assert parse_output('{"a": ', "length", json_output()).error is ErrorCode.TRUNCATED


def test_schema_violations_keep_the_parsed_value() -> None:
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    bad = parse_output('{"a": "x"}', "stop", json_output(), schema)
    assert bad.value is None
    assert bad.error is ErrorCode.SCHEMA_VIOLATION
    assert bad.parsed == {"a": "x"}
    assert parse_output('{"a": 1}', "stop", json_output(), schema).value == '{"a": 1}'


def test_text_output_is_stripped_and_truncation_is_flagged() -> None:
    text = OutputConfig.model_validate({"type": "text"})
    assert parse_output("  hello \n", "stop", text).value == "hello"
    assert parse_output("hel", "length", text).error is ErrorCode.TRUNCATED


def test_canonical_expected() -> None:
    assert canonical_expected(" x \n", OutputConfig.model_validate({"type": "text"}).type) == "x"
    assert canonical_expected('{"b":1, "a":2}', json_output().type) == '{"a": 2, "b": 1}'


def test_load_output_schema(tmp_path: Path) -> None:
    output = json_output(schema_file="schema.json")
    with pytest.raises(ConfigError, match="not found"):
        load_output_schema(tmp_path, output)
    (tmp_path / "schema.json").write_text('{"type": "object"}', encoding="utf-8")
    assert load_output_schema(tmp_path, output) == {"type": "object"}
    (tmp_path / "schema.json").write_text('{"type": "nonsense"}', encoding="utf-8")
    with pytest.raises(ConfigError, match="not a valid JSON schema"):
        load_output_schema(tmp_path, output)
    assert load_output_schema(tmp_path, json_output()) is None
