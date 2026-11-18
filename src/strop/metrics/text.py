"""Metrics comparing text or numeric outputs with the expected value."""

import difflib
from collections.abc import Sequence

from strop.metrics.types import Example, Metric, MetricValue

DEFAULT_TOLERANCE = 1e-6


def _to_float(text: str) -> float | None:
    try:
        return float(text.strip().replace(",", ""))
    except ValueError:
        return None


class _MeanScore(Metric):
    def compute(self, examples: Sequence[Example]) -> MetricValue:
        scores = [s for e in examples if (s := self.per_example(e)) is not None]
        return self._result(sum(scores) / len(scores) if scores else None, len(scores))


class Similarity(_MeanScore):
    """Average character-level similarity (0 to 1) between output and expected text."""

    name = "similarity"

    def per_example(self, example: Example) -> float | None:
        if example.expected is None:
            return None
        if example.value is None:
            return 0.0
        return difflib.SequenceMatcher(None, example.value, example.expected).ratio()


class NumericClose(_MeanScore):
    """Share of outputs whose number is within an absolute tolerance of the expected number."""

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE) -> None:
        self._tolerance = tolerance
        self.name = (
            "numeric_close" if tolerance == DEFAULT_TOLERANCE else f"numeric_close:{tolerance:g}"
        )

    def per_example(self, example: Example) -> float | None:
        if example.expected is None:
            return None
        expected = _to_float(example.expected)
        predicted = _to_float(example.value) if example.value is not None else None
        if expected is None:
            return None
        if predicted is None:
            return 0.0
        return 1.0 if abs(predicted - expected) <= self._tolerance else 0.0
