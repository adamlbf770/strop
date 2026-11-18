import json
from collections.abc import Mapping

import pytest

from strop.metrics.structured import (
    FieldAccuracy,
    FieldCompleteness,
    FieldF1,
    JsonValid,
    SchemaValid,
)
from strop.metrics.types import ErrorCode, Example


def ex(expected: Mapping[str, object] | None, parsed: object, error: str | None = None) -> Example:
    return Example(
        row_id="r",
        inputs={},
        expected=json.dumps(expected, sort_keys=True) if expected is not None else None,
        raw_output="",
        value=None if parsed is None or error else json.dumps(parsed, sort_keys=True),
        parsed=parsed,
        error=error,
    )


EXPECTED = {"name": "Ada", "age": 36, "city": "London"}


def test_json_and_schema_validity() -> None:
    examples = [
        ex(EXPECTED, EXPECTED),
        ex(EXPECTED, {"name": 1}, ErrorCode.SCHEMA_VIOLATION),
        ex(EXPECTED, None, ErrorCode.INVALID_FORMAT),
    ]
    assert JsonValid().compute(examples).value == pytest.approx(2 / 3)
    assert SchemaValid().compute(examples).value == pytest.approx(1 / 3)


def test_field_completeness_counts_present_keys() -> None:
    partial = ex(EXPECTED, {"name": "Bob", "age": 1})
    assert FieldCompleteness().per_example(partial) == pytest.approx(2 / 3)
    assert FieldCompleteness().per_example(ex(EXPECTED, None, ErrorCode.INVALID_FORMAT)) == 0.0


def test_field_accuracy_compares_values_exactly() -> None:
    output = ex(EXPECTED, {"name": "Ada", "age": 35, "city": "London"})
    assert FieldAccuracy().per_example(output) == pytest.approx(2 / 3)


def test_field_f1_penalizes_extra_and_missing_fields() -> None:
    output = ex(EXPECTED, {"name": "Ada", "age": 36, "extra": True})
    # 2 hits, 3 predicted, 3 expected -> precision = recall = 2/3
    assert FieldF1().per_example(output) == pytest.approx(2 / 3)
    assert FieldF1().per_example(ex(EXPECTED, {"x": 1})) == 0.0
    assert FieldF1().per_example(ex(EXPECTED, EXPECTED)) == 1.0


def test_field_metrics_skip_rows_without_an_expected_object() -> None:
    examples = [ex(EXPECTED, EXPECTED), ex(None, {"a": 1})]
    result = FieldAccuracy().compute(examples)
    assert result.samples == 1
    assert result.value == 1.0
    assert FieldAccuracy().compute([ex(None, {"a": 1})]).value is None


def test_field_comparison_ignores_key_order_in_nested_values() -> None:
    expected = {"tags": {"b": 1, "a": 2}}
    output = ex(expected, {"tags": {"a": 2, "b": 1}})
    assert FieldAccuracy().per_example(output) == 1.0
