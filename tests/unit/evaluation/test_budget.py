import pytest

from strop.config.limits import BudgetsConfig
from strop.errors import BudgetExceededError
from strop.evaluation.budget import BudgetTracker
from strop.llm import LLMResponse, Usage


def reply(cost: float | None = None, *, cached: bool = False) -> LLMResponse:
    return LLMResponse(text="x", model="m", usage=Usage(cost_usd=cost), cached=cached)


def test_call_budget_is_enforced() -> None:
    tracker = BudgetTracker(BudgetsConfig(max_total_model_calls=2, max_wall_time_minutes=None))
    tracker.record(reply())
    tracker.check()
    tracker.record(reply())
    with pytest.raises(BudgetExceededError, match="call budget"):
        tracker.check()


def test_cached_responses_are_free() -> None:
    tracker = BudgetTracker(BudgetsConfig(max_total_model_calls=1))
    for _ in range(5):
        tracker.record(reply(cached=True))
    tracker.check()
    assert tracker.calls == 0


def test_cost_budget_uses_known_costs() -> None:
    config = BudgetsConfig(max_total_model_calls=None, max_total_cost_usd=0.01)
    tracker = BudgetTracker(config)
    tracker.record(reply(0.004))
    tracker.check()
    tracker.record(reply(0.007))
    with pytest.raises(BudgetExceededError, match="cost budget"):
        tracker.check()


def test_unknown_costs_are_reported_as_a_caveat() -> None:
    tracker = BudgetTracker(BudgetsConfig(max_total_cost_usd=1.0))
    tracker.record(reply(None))
    tracker.record_failed_call()
    assert "2 call(s) reported no cost" in tracker.warnings()[0]
    assert BudgetTracker(BudgetsConfig()).warnings() == []


def test_time_budget_uses_the_clock() -> None:
    now = [0.0]
    config = BudgetsConfig(max_total_model_calls=None, max_wall_time_minutes=1)
    tracker = BudgetTracker(config, clock=lambda: now[0])
    tracker.check()
    now[0] = 61.0
    with pytest.raises(BudgetExceededError, match="time budget"):
        tracker.check()


def test_disabled_limits_never_raise() -> None:
    config = BudgetsConfig(
        max_total_model_calls=None, max_total_cost_usd=None, max_wall_time_minutes=None
    )
    tracker = BudgetTracker(config)
    for _ in range(100):
        tracker.record(reply(5.0))
    tracker.check()
