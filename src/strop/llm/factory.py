"""Assembling the production client stack."""

from strop.config.limits import ExecutionConfig
from strop.llm.caching import CachingClient
from strop.llm.client import LLMClient
from strop.llm.litellm_client import LiteLLMClient
from strop.llm.retry import RetryingClient
from strop.storage.cache import ResponseCache


def build_client(execution: ExecutionConfig, cache: ResponseCache | None = None) -> LLMClient:
    """LiteLLM with retries, and a response cache when ``cache`` is given."""
    client: LLMClient = RetryingClient(LiteLLMClient(), attempts=execution.retry_attempts)
    return CachingClient(client, cache) if cache is not None else client
