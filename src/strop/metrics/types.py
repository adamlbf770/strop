"""Core metric types."""

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Direction(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ErrorCode(StrEnum):
    """Deterministic failure categories attached to individual examples."""

    PROVIDER_ERROR = "provider_error"
    EMPTY_OUTPUT = "empty_output"
    TRUNCATED = "truncated"
    INVALID_FORMAT = "invalid_format"
    SCHEMA_VIOLATION = "schema_violation"
    WRONG_LABEL = "wrong_label"
    WRONG_VALUE = "wrong_value"
    MISSING_FIELD = "missing_field"
    RULE_VIOLATION = "rule_violation"
    LOW_SCORE = "low_score"
    METRIC_ERROR = "metric_error"


@dataclass(frozen=True, slots=True)
class Example:
    """One model output paired with its row, ready to be scored."""

    row_id: str
    inputs: Mapping[str, str]
    expected: str | None
    raw_output: str
    value: str | None = None
    parsed: Any = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        """True if the output was produced and parsed under the output contract."""
        return self.value is not None

    @property
    def correct(self) -> bool:
        return self.value is not None and self.value == self.expected

    @property
    def expected_json(self) -> Any:
        """The expected output parsed as JSON, or None when absent or not JSON."""
        if self.expected is None:
            return None
        try:
            return json.loads(self.expected)
        except ValueError:
            return None


@dataclass(frozen=True)
class MetricValue:
    """A computed metric. ``value`` is None when it cannot be computed (no scorable examples)."""

    name: str
    value: float | None
    samples: int
    subjective: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """A named score over a set of examples."""

    name: str
    direction: Direction = Direction.MAXIMIZE
    subjective: bool = False
    needs_expected: bool = True

    @abstractmethod
    def compute(self, examples: Sequence[Example]) -> MetricValue:
        """Aggregate score over ``examples``."""

    async def prepare(self, examples: Sequence[Example]) -> None:
        """Do any batched or asynchronous work (model calls, subprocesses) before scoring."""
        return None

    def per_example(self, example: Example) -> float | None:
        """A per-example score used for paired comparison; None if not applicable.

        The default is correctness (1 or 0), or None when there is no expected value.
        Metrics that score on another scale override it.
        """
        if example.expected is None:
            return None
        return 1.0 if example.correct else 0.0

    def feedback(self, example: Example) -> str | None:
        """A short explanation of this example's score, shown to the optimizer."""
        return None

    def _result(self, value: float | None, samples: int, **details: Any) -> MetricValue:
        return MetricValue(self.name, value, samples, self.subjective, dict(details))
