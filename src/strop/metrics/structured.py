"""Metrics for JSON outputs: validity and field-level comparison against the expected object."""

import json
from collections.abc import Sequence
from typing import Any

from strop.metrics.types import ErrorCode, Example, Metric, MetricValue


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _expected_fields(example: Example) -> dict[str, str] | None:
    expected = example.expected_json
    if not isinstance(expected, dict):
        return None
    return {key: _canonical(value) for key, value in expected.items()}


def _predicted_fields(example: Example) -> dict[str, str]:
    if not isinstance(example.parsed, dict):
        return {}
    return {key: _canonical(value) for key, value in example.parsed.items()}


def _mean(scores: list[float]) -> float | None:
    return sum(scores) / len(scores) if scores else None


class JsonValid(Metric):
    """Share of outputs that parse as JSON, before any schema check."""

    name = "json_valid"
    needs_expected = False

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        value = _mean([self.per_example(e) or 0.0 for e in examples])
        return self._result(value, len(examples))

    def per_example(self, example: Example) -> float | None:
        return 1.0 if example.parsed is not None else 0.0


class SchemaValid(Metric):
    """Share of outputs that parse as JSON and satisfy the configured schema."""

    name = "schema_valid"
    needs_expected = False

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        value = _mean([self.per_example(e) or 0.0 for e in examples])
        return self._result(value, len(examples))

    def per_example(self, example: Example) -> float | None:
        ok = example.parsed is not None and example.error != ErrorCode.SCHEMA_VIOLATION
        return 1.0 if ok else 0.0


class _FieldMetric(Metric):
    def compute(self, examples: Sequence[Example]) -> MetricValue:
        scores = [s for e in examples if (s := self.per_example(e)) is not None]
        return self._result(_mean(scores), len(scores))


class FieldCompleteness(_FieldMetric):
    """Average share of expected fields that are present in the output."""

    name = "field_completeness"

    def per_example(self, example: Example) -> float | None:
        expected = _expected_fields(example)
        if not expected:
            return None
        predicted = _predicted_fields(example)
        return sum(key in predicted for key in expected) / len(expected)


class FieldAccuracy(_FieldMetric):
    """Average share of expected fields whose value matches exactly."""

    name = "field_accuracy"

    def per_example(self, example: Example) -> float | None:
        expected = _expected_fields(example)
        if not expected:
            return None
        predicted = _predicted_fields(example)
        return sum(predicted.get(key) == value for key, value in expected.items()) / len(expected)


class FieldF1(_FieldMetric):
    """Average F1 between the output's and the expected object's (field, value) pairs."""

    name = "field_f1"

    def per_example(self, example: Example) -> float | None:
        expected = _expected_fields(example)
        if expected is None:
            return None
        predicted = _predicted_fields(example)
        expected_pairs = set(expected.items())
        predicted_pairs = set(predicted.items())
        hits = len(expected_pairs & predicted_pairs)
        if not hits:
            return 1.0 if not expected_pairs and not predicted_pairs else 0.0
        precision = hits / len(predicted_pairs)
        recall = hits / len(expected_pairs)
        return 2 * precision * recall / (precision + recall)
