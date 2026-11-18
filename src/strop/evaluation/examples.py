"""Joining stored outcomes with their dataset rows to form scorable examples."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from strop.data.models import Row
from strop.evaluation.models import ExampleOutcome
from strop.metrics.suite import MetricSuite
from strop.metrics.types import Example


def to_examples(rows: Sequence[Row], outcomes: Sequence[ExampleOutcome]) -> list[Example]:
    """Pair each outcome with the inputs of its row."""
    by_id: Mapping[str, Row] = {row.id: row for row in rows}
    return [
        Example(
            row_id=o.row_id,
            inputs=by_id[o.row_id].inputs,
            expected=o.expected,
            raw_output=o.raw_output,
            value=o.value,
            parsed=o.parsed,
            error=o.error,
        )
        for o in outcomes
    ]


def with_scores(
    outcomes: Sequence[ExampleOutcome], examples: Sequence[Example], suite: MetricSuite
) -> list[ExampleOutcome]:
    """Attach each example's per-metric scores to its outcome."""
    return [
        replace(
            outcome,
            scores={name: suite.metric(name).per_example(example) for name in suite.names},
        )
        for outcome, example in zip(outcomes, examples, strict=True)
    ]
