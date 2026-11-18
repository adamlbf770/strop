"""Response caching around any client."""

from strop.llm.client import LLMClient
from strop.llm.types import LLMRequest, LLMResponse
from strop.storage.cache import ResponseCache, cache_key


class CachingClient:
    """Serves repeated deterministic requests from the cache.

    Only requests at temperature 0 are cached: sampling is meant to vary between calls,
    so a cached answer would silently remove that variety.
    """

    def __init__(self, inner: LLMClient, cache: ResponseCache) -> None:
        self._inner = inner
        self._cache = cache

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.temperature > 0:
            return await self._inner.complete(request)
        key = cache_key(request)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        response = await self._inner.complete(request)
        self._cache.put(key, response)
        return response
