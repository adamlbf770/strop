"""Bounded retries with exponential backoff for transient provider failures."""

import asyncio
import random
from collections.abc import Awaitable, Callable

from strop.errors import LLMError, TransientLLMError
from strop.llm.client import LLMClient
from strop.llm.types import LLMRequest, LLMResponse


class RetryingClient:
    """Wraps a client; retries ``TransientLLMError`` up to ``attempts`` times in total."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        attempts: int = 3,
        base_delay_s: float = 1.0,
        max_delay_s: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._inner = inner
        self._attempts = max(1, attempts)
        self._base = base_delay_s
        self._max = max_delay_s
        self._sleep = sleep
        self._jitter = jitter

    async def complete(self, request: LLMRequest) -> LLMResponse:
        for attempt in range(1, self._attempts + 1):
            try:
                return await self._inner.complete(request)
            except TransientLLMError as exc:
                if attempt == self._attempts:
                    raise LLMError(f"failed after {attempt} attempt(s): {exc}") from exc
                await self._sleep(self._delay(attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int) -> float:
        delay = float(min(self._max, self._base * 2 ** (attempt - 1)))
        return delay * (0.5 + self._jitter())
