import asyncio

import pytest

from strop.config.limits import BudgetsConfig, ExecutionConfig
from strop.config.llm import ModelSpec
from strop.config.output import OutputConfig
from strop.data.models import Row
from strop.errors import BudgetExceededError, LLMError
from strop.evaluation.budget import BudgetTracker
from strop.evaluation.runner import PromptRunner
from strop.llm import LLMClient, LLMRequest, LLMResponse, MockLLMClient, Usage
from strop.metrics.types import ErrorCode
from strop.prompts import PromptMessage, PromptTemplate

TEMPLATE = PromptTemplate(
    (PromptMessage("system", "Classify."), PromptMessage("user", "Text: {{text}}")),
    frozenset({"system"}),
)
OUTPUT = OutputConfig(type="label", labels=("billing", "refund"))
SPEC = ModelSpec(model="provider/model-a")


def rows(count: int = 3) -> list[Row]:
    return [Row(f"r{i}", i + 1, {"text": f"t{i}"}, "refund", {}, "h") for i in range(count)]


def runner(
    client: LLMClient, *, batch: int = 10, concurrency: int = 3, budget: BudgetTracker | None = None
) -> PromptRunner:
    execution = ExecutionConfig(batch_size=batch, max_concurrency=concurrency)
    return PromptRunner(client, SPEC, OUTPUT, None, execution, budget)


async def test_each_row_becomes_an_outcome_in_order() -> None:
    client = MockLLMClient(
        lambda request: "Refund" if "t1" in request.messages[-1].content else "billing"
    )
    outcomes = await runner(client).run(TEMPLATE, rows())
    assert [o.row_id for o in outcomes] == ["r0", "r1", "r2"]
    assert [o.value for o in outcomes] == ["billing", "refund", "billing"]
    assert [o.expected for o in outcomes] == ["refund"] * 3
    assert all(not o.failed for o in outcomes)


async def test_prompts_are_rendered_per_row() -> None:
    client = MockLLMClient("refund")
    await runner(client).run(TEMPLATE, rows(2))
    sent = sorted(call.messages[-1].content for call in client.calls)
    assert sent == ["Text: t0", "Text: t1"]
    assert client.calls[0].messages[0].content == "Classify."


async def test_invalid_output_is_recorded_not_dropped() -> None:
    outcomes = await runner(MockLLMClient("maybe")).run(TEMPLATE, rows(1))
    assert outcomes[0].value is None
    assert outcomes[0].error == ErrorCode.INVALID_FORMAT
    assert outcomes[0].raw_output == "maybe"


async def test_provider_failures_do_not_fail_the_run() -> None:
    def respond(request: LLMRequest) -> str:
        if "t1" in request.messages[-1].content:
            raise LLMError("provider down, api_key=supersecretvalue")
        return "refund"

    outcomes = await runner(MockLLMClient(respond)).run(TEMPLATE, rows())
    failed = outcomes[1]
    assert failed.error == ErrorCode.PROVIDER_ERROR
    assert failed.raw_output == ""
    assert "supersecretvalue" not in (failed.detail or "")
    assert [o.error for o in (outcomes[0], outcomes[2])] == [None, None]


async def test_usage_and_cache_flags_are_carried_over() -> None:
    usage = Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11, cost_usd=0.001)
    outcomes = await runner(MockLLMClient("refund", usage=usage)).run(TEMPLATE, rows(1))
    assert outcomes[0].total_tokens == 11
    assert outcomes[0].cost_usd == 0.001
    assert outcomes[0].cached is False


async def test_progress_is_reported_per_batch() -> None:
    seen: list[tuple[int, int]] = []
    await runner(MockLLMClient("refund"), batch=2).run(
        TEMPLATE, rows(5), lambda done, total: seen.append((done, total))
    )
    assert seen == [(2, 5), (4, 5), (5, 5)]


async def test_concurrency_is_bounded() -> None:
    state = {"now": 0, "peak": 0}

    class Slow:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.01)
            state["now"] -= 1
            return LLMResponse(text="refund", model=request.model)

    await runner(Slow(), batch=10, concurrency=2).run(TEMPLATE, rows(6))
    assert state["peak"] == 2


async def test_budget_is_checked_between_batches() -> None:
    budget = BudgetTracker(BudgetsConfig(max_total_model_calls=2))
    client = MockLLMClient("refund")
    with pytest.raises(BudgetExceededError):
        await runner(client, batch=2, budget=budget).run(TEMPLATE, rows(6))
    assert len(client.calls) == 2
