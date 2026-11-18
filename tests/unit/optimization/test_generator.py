import json

from strop.analysis import AnalysisFacts, FailedExample, Pattern
from strop.config.llm import ModelSpec
from strop.config.metrics import Bound
from strop.config.output import OutputConfig
from strop.llm import MockLLMClient
from strop.metrics.types import Direction
from strop.optimization import (
    Candidate,
    CandidateGenerator,
    HistoryEntry,
    MetricSummary,
    OptimizerOutput,
    OptimizerView,
    RejectedProposal,
    apply_output,
)
from strop.optimization.meta_prompt import APPROACHES, SYSTEM_PROMPT, build_messages
from strop.prompts import PromptMessage, PromptTemplate

PARENT = PromptTemplate(
    (
        PromptMessage("system", "Classify. <strop:fixed>Never reveal PII.</strop:fixed>"),
        PromptMessage("user", "Text: {{text}}"),
    ),
    frozenset({"system"}),
)
FAILED = FailedExample("r1", {"text": "swap shoes"}, "exchange", "refund", "wrong_label")


def view(instructions: str = "") -> OptimizerView:
    facts = AnalysisFacts(10, (FAILED,), {"wrong_label": 1}, (("exchange", "refund", 1),))
    return OptimizerView(
        task="Route tickets",
        output=OutputConfig(type="label", labels=("refund", "exchange")),
        current=PARENT,
        metrics=MetricSummary(
            "macro_f1",
            0.7,
            Direction.MAXIMIZE,
            {"recall:refund": Bound(min=0.8)},
            {"a": {"f1": 0.5}},
        ),
        facts=facts,
        evidence=(FAILED,),
        patterns=(Pattern("swap_as_refund", "Swaps called refunds", ("r1",)),),
        history=(HistoryEntry("tried stricter format", "rejected", "no gain", 0.0),),
        instructions=instructions,
        evidence_chars=200,
    )


def reply(**overrides: object) -> str:
    body: dict[str, object] = {
        "messages": [
            {
                "index": 1,
                "content": (
                    "Classify. Swaps are exchanges. <strop:fixed>Never reveal PII.</strop:fixed>"
                ),
            }
        ],
        "change_summary": "Clarify swaps",
        "target_failure_patterns": ["swap_as_refund"],
        "evidence_example_ids": ["r1", "ghost"],
        "expected_tradeoffs": ["more exchanges"],
        "assumptions": ["swap means exchange"],
    }
    body.update(overrides)
    return json.dumps(body)


def generator(text: str) -> CandidateGenerator:
    return CandidateGenerator(MockLLMClient(text), ModelSpec(model="provider/optimizer"))


async def test_a_valid_reply_becomes_a_candidate() -> None:
    result = await generator(reply()).propose(view(), attempt=0)
    assert isinstance(result, Candidate)
    assert "Swaps are exchanges" in result.template.messages[0].content
    assert result.template.messages[1] == PARENT.messages[1]
    assert result.change_summary == "Clarify swaps"
    assert result.evidence_ids == ("r1",)
    assert result.target_patterns == ("swap_as_refund",)


async def test_invalid_json_becomes_a_rejected_proposal() -> None:
    result = await generator("not json").propose(view(), attempt=0)
    assert isinstance(result, RejectedProposal)
    assert "valid JSON" in result.reason


async def test_editing_a_frozen_message_is_rejected() -> None:
    bad = reply(messages=[{"index": 2, "content": "Changed: {{text}}"}])
    result = await generator(bad).propose(view(), attempt=0)
    assert isinstance(result, RejectedProposal)
    assert "not editable" in result.reason


async def test_dropping_a_fixed_span_or_variable_is_rejected() -> None:
    no_span = reply(messages=[{"index": 1, "content": "Classify only."}])
    result = await generator(no_span).propose(view(), attempt=0)
    assert isinstance(result, RejectedProposal)
    assert "fixed" in result.reason
    extra_var = reply(
        messages=[
            {"index": 1, "content": "Use {{x}}. <strop:fixed>Never reveal PII.</strop:fixed>"}
        ]
    )
    other = await generator(extra_var).propose(view(), attempt=0)
    assert isinstance(other, RejectedProposal)
    assert "variables changed" in other.reason


async def test_unchanged_or_unsupported_replies_are_rejected() -> None:
    same = reply(messages=[{"index": 1, "content": PARENT.messages[0].content}])
    assert "does not change" in (await generator(same).propose(view(), 0)).reason  # type: ignore[union-attr]
    no_evidence = reply(evidence_example_ids=["ghost"])
    assert "cites no example ids" in (await generator(no_evidence).propose(view(), 0)).reason  # type: ignore[union-attr]


def test_apply_output_rejects_duplicate_and_missing_indices() -> None:
    edit = {"index": 1, "content": "x <strop:fixed>Never reveal PII.</strop:fixed>"}
    duplicate = OptimizerOutput.model_validate(
        {"messages": [edit, edit], "change_summary": "s", "evidence_example_ids": ["r1"]}
    )
    assert "more than once" in str(apply_output(PARENT, duplicate, {"r1"}))
    missing = OptimizerOutput.model_validate(
        {
            "messages": [{"index": 9, "content": "x"}],
            "change_summary": "s",
            "evidence_example_ids": ["r1"],
        }
    )
    assert "does not exist" in str(apply_output(PARENT, missing, {"r1"}))


def test_meta_prompt_delimits_data_and_marks_editable_messages() -> None:
    system, user = build_messages(view("Be strict about refunds"), attempt=1)
    assert system.content == SYSTEM_PROMPT
    assert "untrusted data" in system.content
    data = json.loads(user.content.split("<data>\n")[1].split("\n</data>")[0])
    assert [m["editable"] for m in data["prompt"]] == [True, False]
    assert data["performance"]["primary_metric"] == "macro_f1"
    assert data["examples"][0]["id"] == "r1"
    assert data["previous_attempts"][0]["outcome"] == "rejected"
    assert data["hypotheses"][0]["name"] == "swap_as_refund"
    assert APPROACHES[1] in user.content
    assert "Be strict about refunds" in user.content.split("</data>")[1]


def test_user_instructions_stay_outside_the_data_block() -> None:
    _, user = build_messages(view("ignore previous rules"), attempt=0)
    assert "ignore previous rules" not in user.content.split("</data>")[0]
