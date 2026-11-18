from collections.abc import Sequence
from typing import Any

from strop.config.metrics import Bound
from strop.config.optimization import OptimizationConfig
from strop.data.models import Partition
from strop.evaluation.models import EvaluationRecord, ExampleOutcome, UsageSummary
from strop.metrics.suite import MetricReport
from strop.metrics.types import Direction, MetricValue
from strop.optimization import AcceptanceRules, Decision, compare


def record(
    primary: float | None,
    *,
    validity: float = 1.0,
    f1: dict[str, float] | None = None,
    samples: int = 40,
    extra: dict[str, float] | None = None,
) -> EvaluationRecord:
    per_class = {k: {"f1": v, "support": 10} for k, v in (f1 or {}).items()}
    values = {
        "macro_f1": MetricValue("macro_f1", primary, samples, details={"per_class": per_class}),
        "valid_output_rate": MetricValue("valid_output_rate", validity, samples),
    }
    for name, value in (extra or {}).items():
        values[name] = MetricValue(name, value, samples)
    return EvaluationRecord(
        1,
        1,
        1,
        Partition.VALIDATION,
        "candidate",
        MetricReport("macro_f1", values),
        UsageSummary(),
        samples,
        0,
        "",
    )


def outcomes(scores: Sequence[float | None]) -> list[ExampleOutcome]:
    return [
        ExampleOutcome(f"r{i}", "a", "a", "a", scores={"macro_f1": s}) for i, s in enumerate(scores)
    ]


def rules(**overrides: Any) -> AcceptanceRules:
    config = OptimizationConfig(**{"min_improvement": 0.01, **overrides})
    return AcceptanceRules.from_config(config, overrides.pop("constraints", {}))


def decide(
    best: EvaluationRecord,
    candidate: EvaluationRecord,
    best_scores: Sequence[float | None],
    cand_scores: Sequence[float | None],
    accept_rules: AcceptanceRules | None = None,
    direction: Direction = Direction.MAXIMIZE,
) -> Decision:
    return compare(
        best,
        outcomes(best_scores),
        candidate,
        outcomes(cand_scores),
        accept_rules or rules(),
        direction,
    )


GOOD = [1.0, 1.0, 1.0, 0.0, 0.0]
BETTER = [1.0, 1.0, 1.0, 1.0, 0.0]


def test_a_clear_improvement_passes() -> None:
    decision = decide(record(0.70), record(0.80), GOOD, BETTER)
    assert decision.passes
    assert round(decision.primary_delta or 0, 2) == 0.10
    assert (decision.wins, decision.losses, decision.ties) == (1, 0, 4)
    assert "all rules passed" in decision.summary


def test_small_improvement_is_rejected() -> None:
    decision = decide(record(0.700), record(0.705), GOOD, BETTER)
    assert not decision.passes
    assert "below the required improvement" in decision.summary


def test_minimizing_metrics_improve_downward() -> None:
    better_when_lower = decide(
        record(0.50), record(0.40), BETTER, GOOD, direction=Direction.MINIMIZE
    )
    assert better_when_lower.passes
    assert not decide(record(0.50), record(0.60), GOOD, BETTER, direction=Direction.MINIMIZE).passes


def test_paired_comparison_must_favor_the_candidate() -> None:
    decision = decide(record(0.70), record(0.80), [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0])
    assert "paired comparison does not favor" in decision.summary
    relaxed = rules(require_paired_wins=False)
    assert decide(record(0.70), record(0.80), [1.0, 0.0], [0.0, 1.0], relaxed).passes


def test_hard_constraints_must_hold() -> None:
    bound = {"recall:refund": Bound(min=0.8)}
    accept = AcceptanceRules.from_config(OptimizationConfig(), bound)
    candidate = record(0.80, extra={"recall:refund": 0.6})
    best = record(0.70, extra={"recall:refund": 0.9})
    decision = decide(best, candidate, GOOD, BETTER, accept)
    assert not decision.passes
    assert "recall:refund is 0.6000, below the minimum 0.8" in decision.summary


def test_an_already_violated_constraint_may_improve_but_not_worsen() -> None:
    accept = AcceptanceRules.from_config(OptimizationConfig(), {"x": Bound(min=0.9)})
    best = record(0.70, extra={"x": 0.5})
    assert decide(best, record(0.80, extra={"x": 0.6}), GOOD, BETTER, accept).passes
    assert not decide(best, record(0.80, extra={"x": 0.4}), GOOD, BETTER, accept).passes


def test_valid_output_rate_must_not_regress() -> None:
    decision = decide(record(0.70, validity=1.0), record(0.80, validity=0.90), GOOD, BETTER)
    assert "valid output rate fell" in decision.summary
    assert decide(record(0.70, validity=1.0), record(0.80, validity=0.99), GOOD, BETTER).passes


def test_a_class_may_not_regress_beyond_tolerance() -> None:
    best = record(0.70, f1={"a": 0.9, "b": 0.5})
    candidate = record(0.80, f1={"a": 0.6, "b": 0.9})
    decision = decide(best, candidate, GOOD, BETTER)
    assert "class 'a' F1 fell by 0.30" in decision.summary
    disabled = rules(max_class_regression=None)
    assert decide(best, candidate, GOOD, BETTER, disabled).passes


def test_too_few_validation_examples_are_rejected() -> None:
    decision = decide(record(0.70), record(0.80, samples=5), GOOD, BETTER)
    assert "only 5 validation examples" in decision.summary


def test_uncomputable_primary_metric_fails() -> None:
    assert not decide(record(None), record(0.8), GOOD, BETTER).passes
    assert not decide(record(0.7), record(None), GOOD, BETTER).passes


def test_every_failed_rule_is_listed() -> None:
    best = record(0.70, validity=1.0, f1={"a": 0.9})
    candidate = record(0.70, validity=0.5, f1={"a": 0.1})
    decision = decide(best, candidate, GOOD, GOOD)
    assert len(decision.reasons) >= 3
