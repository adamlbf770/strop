"""The interface every model provider implementation satisfies."""

from typing import Protocol, runtime_checkable

from strop.llm.types import LLMRequest, LLMResponse


@runtime_checkable
class LLMClient(Protocol):
    """Sends one chat request and returns a normalized response.

    Implementations raise ``strop.errors.LLMError`` when a call fails for good.
    """

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
