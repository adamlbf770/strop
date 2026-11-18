from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from strop.errors import LLMError, TransientLLMError
from strop.llm import LLMRequest, Message
from strop.llm import litellm_client as module
from strop.llm.litellm_client import LiteLLMClient


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


def fake_litellm(response: Any = None, error: Exception | None = None, cost: Any = 0.002) -> Any:
    calls: list[dict[str, Any]] = []

    async def acompletion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if error is not None:
            raise error
        return response

    def completion_cost(completion_response: Any) -> float:
        if isinstance(cost, Exception):
            raise cost
        return float(cost)

    fake = ModuleType("litellm")
    fake.RateLimitError = RateLimitError  # type: ignore[attr-defined]
    fake.acompletion = acompletion  # type: ignore[attr-defined]
    fake.completion_cost = completion_cost  # type: ignore[attr-defined]
    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def reply(text: str = "refund", finish: str = "stop", usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=usage or SimpleNamespace(prompt_tokens=11, completion_tokens=2, total_tokens=13),
        model="provider/model-a-2026",
        id="req-1",
    )


def request(**kwargs: Any) -> LLMRequest:
    return LLMRequest(
        model="provider/model-a",
        messages=(Message(role="system", content="s"), Message(role="user", content="u")),
        max_tokens=16,
        params={"seed": 7},
        **kwargs,
    )


def install(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    monkeypatch.setattr(module, "_load_litellm", lambda: fake)


async def test_response_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_litellm(reply())
    install(monkeypatch, fake)
    response = await LiteLLMClient().complete(request())
    assert response.text == "refund"
    assert response.finish_reason == "stop"
    assert response.model == "provider/model-a-2026"
    assert response.request_id == "req-1"
    assert response.usage.total_tokens == 13
    assert response.usage.cost_usd == 0.002
    assert response.latency_s >= 0


async def test_request_settings_reach_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_litellm(reply())
    install(monkeypatch, fake)
    await LiteLLMClient().complete(request())
    sent = fake.calls[0]
    assert sent["model"] == "provider/model-a"
    assert sent["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert (sent["temperature"], sent["max_tokens"], sent["seed"]) == (0.0, 16, 7)
    assert sent["drop_params"] is True


async def test_unavailable_cost_stays_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, fake_litellm(reply(), cost=ValueError("unknown model")))
    response = await LiteLLMClient().complete(request())
    assert response.usage.cost_usd is None


async def test_missing_usage_is_not_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = reply()
    raw.usage = None
    install(monkeypatch, fake_litellm(raw, cost=ValueError("no")))
    response = await LiteLLMClient().complete(request())
    assert response.usage.prompt_tokens is None
    assert response.usage.total_tokens is None


async def test_empty_content_becomes_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, fake_litellm(reply(text=None)))  # type: ignore[arg-type]
    assert (await LiteLLMClient().complete(request())).text == ""


async def test_transient_and_permanent_errors_are_distinguished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, fake_litellm(error=RateLimitError("slow down")))
    with pytest.raises(TransientLLMError, match="RateLimitError"):
        await LiteLLMClient().complete(request())
    install(monkeypatch, fake_litellm(error=AuthenticationError("bad key sk-secret123456")))
    with pytest.raises(LLMError) as info:
        await LiteLLMClient().complete(request())
    assert not isinstance(info.value, TransientLLMError)
    assert "sk-secret123456" not in str(info.value)


async def test_malformed_response_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, fake_litellm(SimpleNamespace(choices=[])))
    with pytest.raises(LLMError, match="without a message"):
        await LiteLLMClient().complete(request())
