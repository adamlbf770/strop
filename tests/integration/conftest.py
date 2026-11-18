"""A scripted pair of models: a flawed target and an optimizer that fixes the flaw."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from strop.llm import LLMRequest, MockLLMClient
from strop.optimization.meta_prompt import APPROACHES
from tests.helpers import CliRunner

FIX = "REFUND-RULE"


def number(request: LLMRequest) -> int:
    return int(request.messages[-1].content.split()[-1])


def target_model(request: LLMRequest) -> str:
    """Correct, except it misses some refunds unless the prompt contains the fix."""
    truth = "refund" if number(request) % 2 else "billing"
    if truth == "refund" and number(request) % 4 == 1 and FIX not in request.messages[0].content:
        return "billing"
    return truth


def data_block(request: LLMRequest) -> dict[str, Any]:
    text = request.messages[-1].content
    block: dict[str, Any] = json.loads(text.split("<data>\n")[1].split("\n</data>")[0])
    return block


def helper_model(request: LLMRequest) -> str:
    data = data_block(request)
    if request.messages[0].content.startswith("You analyze"):
        ids = [e["id"] for e in data["examples"]][:2]
        pattern = {
            "name": "refund_missed",
            "description": "refunds called billing",
            "example_ids": ids,
        }
        return json.dumps({"patterns": [pattern]})
    user = request.messages[-1].content
    attempt = next(i for i, text in enumerate(APPROACHES) if text in user)
    current = data["prompt"][0]["content"]
    revised = (
        current.rstrip("\n") + f"\n{FIX} variant {attempt}: a request for money back is a refund."
    )
    return json.dumps(
        {
            "messages": [{"index": 1, "content": revised}],
            "change_summary": f"Add a refund rule (variant {attempt})",
            "target_failure_patterns": ["refund_missed"],
            "evidence_example_ids": [data["examples"][0]["id"]],
            "expected_tradeoffs": ["may over-predict refund"],
            "assumptions": ["money back means refund"],
        }
    )


Install = Callable[..., MockLLMClient]


@pytest.fixture
def install_models(monkeypatch: pytest.MonkeyPatch) -> Install:
    """Route every model call to the scripted models; optionally override the helper model."""

    def install(helper: Callable[[LLMRequest], str] = helper_model) -> MockLLMClient:
        def route(request: LLMRequest) -> str:
            return (
                helper(request) if request.model.startswith("anthropic") else target_model(request)
            )

        client = MockLLMClient(route)
        for module in ("evaluator", "context", "evaluate", "analysis"):
            monkeypatch.setattr(
                f"strop.services.{module}.build_client", lambda *_: client, raising=False
            )
        return client

    return install


@pytest.fixture
def ready(run_cli: CliRunner, project: Path) -> Path:
    """A project with a data split, ready to optimize."""
    assert run_cli("data", "split", "--dir", str(project))[0] == 0
    return project
