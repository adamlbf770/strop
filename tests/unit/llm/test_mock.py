import pytest

from strop.errors import LLMError
from strop.llm import LLMClient, LLMRequest, Message, MockLLMClient, Usage


def make_request(text: str = "hi") -> LLMRequest:
    return LLMRequest(model="provider/model-a", messages=(Message(role="user", content=text),))


def test_mock_satisfies_the_client_protocol() -> None:
    assert isinstance(MockLLMClient("x"), LLMClient)


async def test_fixed_text_is_returned_every_time() -> None:
    client = MockLLMClient("billing")
    assert (await client.complete(make_request())).text == "billing"
    assert (await client.complete(make_request())).text == "billing"


async def test_scripted_list_is_consumed_in_order_then_fails() -> None:
    client = MockLLMClient(["a", "b"])
    assert (await client.complete(make_request())).text == "a"
    assert (await client.complete(make_request())).text == "b"
    with pytest.raises(LLMError, match="no more scripted"):
        await client.complete(make_request())


async def test_function_responder_sees_the_request() -> None:
    client = MockLLMClient(lambda request: request.messages[-1].content.upper())
    assert (await client.complete(make_request("refund"))).text == "REFUND"


async def test_calls_are_recorded() -> None:
    client = MockLLMClient("x")
    request = make_request("hello")
    await client.complete(request)
    assert client.calls == [request]


async def test_response_metadata() -> None:
    client = MockLLMClient("x")
    response = await client.complete(make_request())
    assert response.model == "provider/model-a"
    assert response.finish_reason == "stop"
    assert response.request_id == "mock-1"
    assert response.usage == Usage()


async def test_usage_can_be_scripted() -> None:
    usage = Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    response = await MockLLMClient("x", usage=usage).complete(make_request())
    assert response.usage.total_tokens == 12
