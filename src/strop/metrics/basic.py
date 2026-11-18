"""Metrics that apply to every output type."""

from collections.abc import Sequence

from strop.metrics.types import Example, Metric, MetricValue


class ValidOutputRate(Metric):
    """Share of outputs that were produced and parsed under the output contract."""

    name = "valid_output_rate"

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        value = sum(e.valid for e in examples) / len(examples) if examples else None
        return self._result(value, len(examples))

    def per_example(self, example: Example) -> float | None:
        return 1.0 if example.valid else 0.0


class ExactMatch(Metric):
    """Share of outputs equal to the expected value after normalization."""

    name = "exact_match"

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        items = [e for e in examples if e.expected is not None]
        value = sum(e.correct for e in items) / len(items) if items else None
        return self._result(value, len(items))
