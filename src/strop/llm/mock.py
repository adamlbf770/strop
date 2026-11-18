"""A scriptable client for tests and offline development. It never touches the network."""

from collections.abc import Callable, Sequence

from strop.errors import LLMError
from strop.llm.types import LLMRequest, LLMResponse, Usage

Responder = Callable[[LLMRequest], str]


class MockLLMClient:
    """Answers requests from a fixed text, an ordered list, or a function.

    Every request is recorded in ``calls`` so tests can assert on what was sent.
    Token and cost data are left unavailable unless ``usage`` is given.
    """

    def __init__(self, responses: str | Sequence[str] | Responder, *, usage: Usage | None = None):
        self._responses = responses
        self._usage = usage or Usage()
        self._next = 0
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self._answer(request),
            model=request.model,
            finish_reason="stop",
            usage=self._usage,
            request_id=f"mock-{len(self.calls)}",
        )

    def _answer(self, request: LLMRequest) -> str:
        if isinstance(self._responses, str):
            return self._responses
        if callable(self._responses):
            return self._responses(request)
        if self._next >= len(self._responses):
            raise LLMError("MockLLMClient has no more scripted responses")
        answer = self._responses[self._next]
        self._next += 1
        return answer
