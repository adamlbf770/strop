"""Classification metrics: accuracy, precision, recall, F1, and the confusion matrix.

An invalid or missing prediction counts as a miss for the true class and as a wrong
prediction for none, so a prompt cannot raise its scores by producing unusable output.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from strop.metrics.types import Example, Metric, MetricValue

INVALID = "(invalid)"
Average = Literal["macro", "micro", "weighted"]
Kind = Literal["precision", "recall", "f1"]


@dataclass(frozen=True, slots=True)
class ClassStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def score(self, kind: Kind) -> float:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1}[kind]


def scorable(examples: Sequence[Example]) -> list[Example]:
    """Examples that have an expected value to compare against."""
    return [e for e in examples if e.expected is not None]


def class_labels(examples: Sequence[Example], labels: Sequence[str]) -> tuple[str, ...]:
    """The configured labels, or the distinct expected values when none are configured."""
    if labels:
        return tuple(labels)
    return tuple(sorted({e.expected for e in examples if e.expected is not None}))


def class_stats(examples: Sequence[Example], labels: Sequence[str]) -> dict[str, ClassStats]:
    counts = {label: [0, 0, 0] for label in labels}
    for example in scorable(examples):
        truth, predicted = example.expected, example.value
        if truth == predicted and truth in counts:
            counts[truth][0] += 1
            continue
        if predicted in counts:
            counts[predicted][1] += 1
        if truth in counts:
            counts[truth][2] += 1
    return {label: ClassStats(*values) for label, values in counts.items()}


def confusion_matrix(
    examples: Sequence[Example], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Rows are true labels; columns are predictions, with ``(invalid)`` for unusable output."""
    columns = [*labels, INVALID]
    matrix = {label: dict.fromkeys(columns, 0) for label in labels}
    for example in scorable(examples):
        if example.expected in matrix:
            value = example.value
            predicted = value if value is not None and value in labels else INVALID
            matrix[example.expected][predicted] += 1
    return matrix


def averaged(stats: dict[str, ClassStats], average: Average, kind: Kind) -> float | None:
    """Combine per-class scores. Classes with no examples and no predictions are skipped."""
    present = {k: s for k, s in stats.items() if s.support or s.tp + s.fp}
    if not present:
        return None
    if average == "macro":
        return sum(s.score(kind) for s in present.values()) / len(present)
    if average == "weighted":
        total = sum(s.support for s in present.values())
        return sum(s.support * s.score(kind) for s in present.values()) / total if total else None
    pooled = ClassStats(
        sum(s.tp for s in present.values()),
        sum(s.fp for s in present.values()),
        sum(s.fn for s in present.values()),
    )
    return pooled.score(kind)


class _ClassificationMetric(Metric):
    def __init__(self, labels: Sequence[str]) -> None:
        self._labels = tuple(labels)

    def _details(self, examples: Sequence[Example]) -> tuple[dict[str, ClassStats], dict[str, Any]]:
        labels = class_labels(examples, self._labels)
        stats = class_stats(examples, labels)
        details = {
            "per_class": {
                label: {
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                    "support": s.support,
                }
                for label, s in stats.items()
            },
            "confusion": confusion_matrix(examples, labels),
        }
        return stats, details


class Accuracy(_ClassificationMetric):
    name = "accuracy"

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        _, details = self._details(examples)
        items = scorable(examples)
        value = sum(e.correct for e in items) / len(items) if items else None
        return self._result(value, len(items), **details)


class Averaged(_ClassificationMetric):
    """``macro_f1``, ``micro_recall``, ``weighted_precision`` and the like."""

    def __init__(self, average: Average, kind: Kind, labels: Sequence[str]) -> None:
        super().__init__(labels)
        self._average = average
        self._kind = kind
        self.name = f"{average}_{kind}"

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        stats, details = self._details(examples)
        value = averaged(stats, self._average, self._kind)
        return self._result(value, len(scorable(examples)), **details)


class ClassScore(_ClassificationMetric):
    """``precision:<label>``, ``recall:<label>`` and ``f1:<label>``."""

    def __init__(self, kind: Kind, label: str, labels: Sequence[str]) -> None:
        super().__init__(labels)
        self._kind = kind
        self._label = label
        self.name = f"{kind}:{label}"

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        stats, details = self._details(examples)
        class_stat = stats.get(self._label, ClassStats())
        has_data = class_stat.support or class_stat.tp + class_stat.fp
        value = class_stat.score(self._kind) if has_data else None
        return self._result(value, len(scorable(examples)), **details)
