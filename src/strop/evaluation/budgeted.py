"""A client wrapper that counts every call against the run budget."""

from strop.evaluation.budget import BudgetTracker
from strop.llm.client import LLMClient
from strop.llm.types import LLMRequest, LLMResponse


class BudgetedClient:
    """Checks the budget before each call and records the call afterwards.

    Used for the optimizer, analyzer and judge. Target-model calls are counted by
    ``PromptRunner`` instead, so they must not go through this wrapper as well.
    """

    def __init__(self, inner: LLMClient, budget: BudgetTracker) -> None:
        self._inner = inner
        self._budget = budget

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._budget.check()
        response = await self._inner.complete(request)
        self._budget.record(response)
        return response
