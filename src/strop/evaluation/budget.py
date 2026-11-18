"""Tracking calls, cost and time against the configured budgets."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from strop.config.limits import BudgetsConfig
from strop.errors import BudgetExceededError
from strop.llm.types import LLMResponse


@dataclass
class BudgetTracker:
    """Counts fresh model calls and known cost, and raises once a limit is reached.

    Cached responses are free and are not counted. A cost limit can only be enforced
    for calls whose provider reported a cost; ``cost_unknown_calls`` shows the rest.
    """

    config: BudgetsConfig
    clock: Callable[[], float] = time.monotonic
    calls: int = 0
    cost_usd: float = 0.0
    cost_unknown_calls: int = 0
    _start: float = field(init=False)

    def __post_init__(self) -> None:
        self._start = self.clock()

    @property
    def elapsed_minutes(self) -> float:
        return (self.clock() - self._start) / 60

    def record(self, response: LLMResponse) -> None:
        if response.cached:
            return
        self.calls += 1
        if response.usage.cost_usd is None:
            self.cost_unknown_calls += 1
        else:
            self.cost_usd += response.usage.cost_usd

    def record_failed_call(self) -> None:
        self.calls += 1
        self.cost_unknown_calls += 1

    def check(self) -> None:
        """Raise ``BudgetExceededError`` if any configured limit has been reached."""
        limits = self.config
        if limits.max_total_model_calls is not None and self.calls >= limits.max_total_model_calls:
            raise BudgetExceededError(f"model call budget reached ({self.calls} calls)")
        if limits.max_total_cost_usd is not None and self.cost_usd >= limits.max_total_cost_usd:
            raise BudgetExceededError(f"cost budget reached (${self.cost_usd:.4f})")
        if (
            limits.max_wall_time_minutes is not None
            and self.elapsed_minutes >= limits.max_wall_time_minutes
        ):
            raise BudgetExceededError(f"time budget reached ({self.elapsed_minutes:.1f} minutes)")

    def warnings(self) -> list[str]:
        """Caveats about limits that cannot be fully enforced."""
        if self.config.max_total_cost_usd is not None and self.cost_unknown_calls:
            return [
                f"{self.cost_unknown_calls} call(s) reported no cost, so the cost budget "
                "only covers the calls that did"
            ]
        return []
