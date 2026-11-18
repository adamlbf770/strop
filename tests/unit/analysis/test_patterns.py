import json

import pytest
from pydantic import BaseModel

from strop.analysis import AnalysisFacts, FailedExample, Pattern, find_patterns
from strop.config.limits import BudgetsConfig
from strop.config.llm import ModelSpec
from strop.config.output import OutputConfig
from strop.errors import BudgetExceededError, LLMError
from strop.evaluation.budget import BudgetTracker
from strop.evaluation.budgeted import BudgetedClient
from strop.llm import LLMRequest, Message, MockLLMClient
from strop.llm.structured import complete_json, extract_json

SPEC = ModelSpec(model="provider/analyzer")
OUTPUT = OutputConfig(type="label", labels=("refund", "exchange"))


def evidence() -> list[FailedExample]:
    return [
        FailedExample("r1", {"text": "swap my shoes"}, "exchange", "refund", "wrong_label"),
        FailedExample("r2", {"text": "different size"}, "exchange", "refund", "wrong_label"),
    ]


def facts() -> AnalysisFacts:
    return AnalysisFacts(10, tuple(evidence()), {"wrong_label": 2}, (("exchange", "refund", 2),))


def request() -> LLMRequest:
    return LLMRequest(model="m", messages=(Message(role="user", content="q"),))


async def run(reply: str) -> tuple[list[Pattern], MockLLMClient]:
    client = MockLLMClient(reply)
    patterns = await find_patterns(
        client,
        SPEC,
        task="Route tickets",
        output=OUTPUT,
        facts=facts(),
        evidence=evidence(),
        max_chars=100,
    )
    return patterns, client


async def test_patterns_cite_real_examples() -> None:
    reply = json.dumps(
        {
            "patterns": [
                {
                    "name": "swap_as_refund",
                    "description": "Swaps are called refunds",
                    "example_ids": ["r1", "r2", "ghost"],
                },
                {"name": "made_up", "description": "No evidence", "example_ids": ["ghost"]},
            ]
        }
    )
    patterns, client = await run(reply)
    assert [p.name for p in patterns] == ["swap_as_refund"]
    assert patterns[0].example_ids == ("r1", "r2")
    assert len(client.calls) == 1


async def test_data_is_delimited_and_flagged_as_untrusted() -> None:
    _, client = await run('{"patterns": []}')
    system, user = (m.content for m in client.calls[0].messages)
    assert "untrusted data" in system
    assert user.startswith("<data>")
    assert user.rstrip().endswith("</data>")
    assert "swap my shoes" in user


async def test_no_evidence_means_no_model_call() -> None:
    client = MockLLMClient("{}")
    empty = AnalysisFacts(5, (), {}, ())
    result = await find_patterns(
        client, SPEC, task="t", output=OUTPUT, facts=empty, evidence=[], max_chars=50
    )
    assert result == []
    assert client.calls == []


class Answer(BaseModel):
    value: int


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Here you go: {"a": 1} Done.') == '{"a": 1}'
    assert extract_json('{"a": 1}') == '{"a": 1}'


async def test_invalid_json_is_retried_with_the_error_shown() -> None:
    client = MockLLMClient(['{"value": "x"}', '{"value": 3}'])
    result = await complete_json(client, request(), Answer)
    assert result.value == 3
    assert "value" in client.calls[1].messages[-1].content
    assert len(client.calls[1].messages) == 3


async def test_gives_up_after_the_allowed_attempts() -> None:
    client = MockLLMClient("not json at all")
    with pytest.raises(LLMError, match="after 2 attempt"):
        await complete_json(client, request(), Answer, attempts=2)


async def test_budgeted_client_counts_and_enforces_calls() -> None:
    tracker = BudgetTracker(BudgetsConfig(max_total_model_calls=1))
    client = BudgetedClient(MockLLMClient("x"), tracker)
    await client.complete(request())
    assert tracker.calls == 1
    with pytest.raises(BudgetExceededError):
        await client.complete(request())
