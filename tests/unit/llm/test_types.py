import pytest
from pydantic import ValidationError

from strop.config.llm import ModelSpec
from strop.llm import LLMRequest, LLMResponse, Message, Usage


def test_request_requires_a_message() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(model="provider/model-a", messages=())


def test_request_requires_a_model_name() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(model="", messages=(Message(role="user", content="hi"),))


def test_for_model_copies_the_configured_settings() -> None:
    spec = ModelSpec(
        model="provider/model-a", temperature=0.3, max_tokens=64, timeout_s=10, params={"seed": 1}
    )
    request = LLMRequest.for_model(spec, [Message(role="user", content="hi")])
    assert request.model == "provider/model-a"
    assert request.temperature == 0.3
    assert request.max_tokens == 64
    assert request.timeout_s == 10
    assert request.params == {"seed": 1}


def test_usage_fields_default_to_unavailable() -> None:
    usage = Usage()
    assert usage.prompt_tokens is None
    assert usage.cost_usd is None


def test_response_defaults_do_not_invent_usage() -> None:
    response = LLMResponse(text="ok", model="provider/model-a")
    assert response.usage == Usage()
    assert response.request_id is None
