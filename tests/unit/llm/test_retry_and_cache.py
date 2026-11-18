import pytest

from strop.errors import LLMError, TransientLLMError
from strop.llm import LLMRequest, LLMResponse, Message, MockLLMClient
from strop.llm.caching import CachingClient
from strop.llm.retry import RetryingClient
from strop.storage import connect
from strop.storage.cache import ResponseCache, cache_key
from strop.storage.migrator import migrate


def make_request(text: str = "hi", temperature: float = 0.0) -> LLMRequest:
    return LLMRequest(
        model="provider/model-a",
        messages=(Message(role="user", content=text),),
        temperature=temperature,
    )


class Flaky:
    """Fails with a transient error a set number of times, then succeeds."""

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or TransientLLMError("temporary")
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return LLMResponse(text="ok", model=request.model)


async def no_sleep(_: float) -> None:
    return None


async def test_retries_transient_errors_then_succeeds() -> None:
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    inner = Flaky(failures=2)
    client = RetryingClient(inner, attempts=3, base_delay_s=1, sleep=record, jitter=lambda: 0.5)
    assert (await client.complete(make_request())).text == "ok"
    assert inner.calls == 3
    assert delays == [1.0, 2.0]


async def test_gives_up_after_the_configured_attempts() -> None:
    inner = Flaky(failures=10)
    client = RetryingClient(inner, attempts=3, sleep=no_sleep)
    with pytest.raises(LLMError, match="failed after 3 attempt"):
        await client.complete(make_request())
    assert inner.calls == 3


async def test_permanent_errors_are_not_retried() -> None:
    inner = Flaky(failures=10, error=LLMError("bad request"))
    client = RetryingClient(inner, attempts=5, sleep=no_sleep)
    with pytest.raises(LLMError, match="bad request"):
        await client.complete(make_request())
    assert inner.calls == 1


async def test_backoff_is_capped() -> None:
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    client = RetryingClient(
        Flaky(failures=10),
        attempts=5,
        base_delay_s=10,
        max_delay_s=15,
        sleep=record,
        jitter=lambda: 0.5,
    )
    with pytest.raises(LLMError):
        await client.complete(make_request())
    assert delays == [10.0, 15.0, 15.0, 15.0]


def make_cache() -> ResponseCache:
    conn = connect(":memory:")
    migrate(conn)
    return ResponseCache(conn)


def test_cache_key_depends_on_everything_that_shapes_the_response() -> None:
    base = cache_key(make_request("a"))
    assert base == cache_key(make_request("a"))
    assert base != cache_key(make_request("b"))
    assert base != cache_key(make_request("a").model_copy(update={"model": "other"}))
    assert base != cache_key(make_request("a").model_copy(update={"max_tokens": 5}))
    assert base != cache_key(make_request("a").model_copy(update={"params": {"seed": 1}}))


async def test_second_identical_request_is_served_from_the_cache() -> None:
    inner = MockLLMClient("answer")
    client = CachingClient(inner, make_cache())
    first = await client.complete(make_request())
    second = await client.complete(make_request())
    assert (first.cached, second.cached) == (False, True)
    assert second.text == "answer"
    assert second.latency_s == 0.0
    assert len(inner.calls) == 1


async def test_sampling_requests_are_never_cached() -> None:
    inner = MockLLMClient(["one", "two"])
    client = CachingClient(inner, make_cache())
    assert (await client.complete(make_request(temperature=0.7))).text == "one"
    assert (await client.complete(make_request(temperature=0.7))).text == "two"


async def test_different_prompts_do_not_share_cache_entries() -> None:
    client = CachingClient(MockLLMClient(["a", "b"]), make_cache())
    assert (await client.complete(make_request("x"))).text == "a"
    assert (await client.complete(make_request("y"))).text == "b"


def test_clear_empties_the_cache() -> None:
    cache = make_cache()
    cache.put("k", LLMResponse(text="t", model="m"))
    assert cache.get("k") is not None
    assert cache.clear() == 1
    assert cache.get("k") is None
