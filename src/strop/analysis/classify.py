"""Deterministic error labels for individual outcomes."""

import json

from strop.config.enums import OutputType
from strop.evaluation.models import ExampleOutcome
from strop.metrics.types import ErrorCode


def classify_outcome(outcome: ExampleOutcome, output_type: OutputType) -> str | None:
    """The failure category of ``outcome``, or None if it is correct or cannot be judged."""
    if outcome.error:
        return outcome.error
    if outcome.value is None:
        return None
    if outcome.expected is None or outcome.value == outcome.expected:
        return ErrorCode.RULE_VIOLATION if failed_rules(outcome) else None
    if output_type is OutputType.LABEL:
        return ErrorCode.WRONG_LABEL
    if output_type is OutputType.JSON and _missing_fields(outcome):
        return ErrorCode.MISSING_FIELD
    return ErrorCode.WRONG_VALUE


def _missing_fields(outcome: ExampleOutcome) -> bool:
    try:
        expected = json.loads(outcome.expected or "")
    except ValueError:
        return False
    if not isinstance(expected, dict) or not isinstance(outcome.parsed, dict):
        return False
    return any(key not in outcome.parsed for key in expected)


def failed_rules(outcome: ExampleOutcome) -> tuple[str, ...]:
    """Names of the rule checks this outcome failed."""
    return tuple(
        name for name, score in outcome.scores.items() if name.startswith("rule:") and score == 0.0
    )
