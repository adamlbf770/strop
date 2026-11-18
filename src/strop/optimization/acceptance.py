"""Deciding whether a candidate may replace the best prompt."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from strop.config.metrics import Bound
from strop.config.optimization import OptimizationConfig
from strop.evaluation.models import EvaluationRecord, ExampleOutcome
from strop.metrics.types import Direction

VALIDITY = "valid_output_rate"


@dataclass(frozen=True)
class Decision:
    """The outcome of the acceptance rules for one candidate."""

    passes: bool
    reasons: tuple[str, ...]
    primary_delta: float | None = None
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons)


@dataclass(frozen=True)
class AcceptanceRules:
    """The settings acceptance needs, copied from the project configuration."""

    min_improvement: float
    constraints: Mapping[str, Bound]
    max_class_regression: float | None
    max_validity_regression: float
    min_validation_samples: int
    require_paired_wins: bool

    @classmethod
    def from_config(
        cls, optimization: OptimizationConfig, constraints: Mapping[str, Bound]
    ) -> "AcceptanceRules":
        return cls(
            optimization.min_improvement,
            constraints,
            optimization.max_class_regression,
            optimization.max_validity_regression,
            optimization.min_validation_samples,
            optimization.require_paired_wins,
        )


def compare(
    best: EvaluationRecord,
    best_outcomes: Sequence[ExampleOutcome],
    candidate: EvaluationRecord,
    candidate_outcomes: Sequence[ExampleOutcome],
    rules: AcceptanceRules,
    direction: Direction,
) -> Decision:
    """Apply every rule to ``candidate`` against the current ``best``; list each failure."""
    sign = 1.0 if direction is Direction.MAXIMIZE else -1.0
    before, after = best.report.primary_value.value, candidate.report.primary_value.value
    if before is None or after is None:
        return Decision(False, ("the primary metric could not be computed",))
    delta = sign * (after - before)
    wins, losses, ties = _paired(best_outcomes, candidate_outcomes, best.report.primary, sign)

    failures: list[str] = []
    if candidate.sample_count < rules.min_validation_samples:
        failures.append(
            f"only {candidate.sample_count} validation examples; at least "
            f"{rules.min_validation_samples} are needed"
        )
    if delta < rules.min_improvement:
        failures.append(
            f"primary metric changed by {delta:+.4f}, below the required improvement "
            f"of {rules.min_improvement}"
        )
    failures += _constraint_failures(best, candidate, rules.constraints)
    failures += _validity_failures(best, candidate, rules.max_validity_regression)
    failures += _class_failures(best, candidate, rules.max_class_regression)
    if rules.require_paired_wins and wins + losses > 0 and wins <= losses:
        failures.append(
            f"the paired comparison does not favor the candidate (wins {wins}, losses {losses})"
        )
    reasons = tuple(failures) or (f"all rules passed (primary {delta:+.4f})",)
    return Decision(not failures, reasons, delta, wins, losses, ties)


def _paired(
    best: Sequence[ExampleOutcome],
    candidate: Sequence[ExampleOutcome],
    primary: str,
    sign: float,
) -> tuple[int, int, int]:
    old = {o.row_id: o.scores.get(primary) for o in best}
    wins = losses = ties = 0
    for outcome in candidate:
        before, after = old.get(outcome.row_id), outcome.scores.get(primary)
        if before is None or after is None:
            continue
        difference = sign * (after - before)
        wins += difference > 0
        losses += difference < 0
        ties += difference == 0
    return wins, losses, ties


def _constraint_failures(
    best: EvaluationRecord, candidate: EvaluationRecord, constraints: Mapping[str, Bound]
) -> list[str]:
    """A constraint must hold; if it already failed on the best prompt, it must not get worse."""
    failures = []
    for name, bound in constraints.items():
        new = candidate.report.values.get(name)
        old = best.report.values.get(name)
        if new is None or new.value is None:
            failures.append(f"{name} could not be computed")
            continue
        if bound.allows(new.value):
            continue
        already_failing = old is not None and old.value is not None and not bound.allows(old.value)
        worse = old is not None and old.value is not None and _worse(bound, new.value, old.value)
        if already_failing and not worse:
            continue
        failures.append(_violation(name, bound, new.value))
    return failures


def _worse(bound: Bound, new: float, old: float) -> bool:
    if bound.min is not None and new < bound.min:
        return new < old
    return new > old


def _violation(name: str, bound: Bound, value: float) -> str:
    if bound.min is not None and value < bound.min:
        return f"{name} is {value:.4f}, below the minimum {bound.min}"
    return f"{name} is {value:.4f}, above the maximum {bound.max}"


def _validity_failures(
    best: EvaluationRecord, candidate: EvaluationRecord, tolerance: float
) -> list[str]:
    old, new = best.report.values.get(VALIDITY), candidate.report.values.get(VALIDITY)
    if old is None or new is None or old.value is None or new.value is None:
        return []
    if new.value < old.value - tolerance:
        return [f"valid output rate fell from {old.value:.4f} to {new.value:.4f}"]
    return []


def _class_failures(
    best: EvaluationRecord, candidate: EvaluationRecord, tolerance: float | None
) -> list[str]:
    if tolerance is None:
        return []
    old = best.report.primary_value.details.get("per_class", {})
    new = candidate.report.primary_value.details.get("per_class", {})
    failures = []
    for label, scores in old.items():
        if not scores.get("support") or label not in new:
            continue
        drop = scores["f1"] - new[label]["f1"]
        if drop > tolerance:
            failures.append(f"class '{label}' F1 fell by {drop:.2f} (limit {tolerance:.2f})")
    return failures
