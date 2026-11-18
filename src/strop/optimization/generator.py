"""Asking the optimizer model for a revision and validating what comes back."""

from dataclasses import replace

from pydantic import BaseModel, Field

from strop.config.llm import ModelSpec
from strop.errors import InvalidModelReplyError
from strop.llm.client import LLMClient
from strop.llm.structured import complete_json
from strop.llm.types import LLMRequest
from strop.optimization.meta_prompt import build_messages
from strop.optimization.types import Candidate, OptimizerView, RejectedProposal
from strop.prompts import PromptMessage, PromptTemplate, validate_revision


class MessageEdit(BaseModel):
    index: int = Field(ge=1)
    content: str = Field(min_length=1)


class OptimizerOutput(BaseModel):
    """The strict reply format of the optimizer model."""

    messages: list[MessageEdit] = Field(min_length=1)
    change_summary: str = Field(min_length=1, max_length=400)
    target_failure_patterns: list[str] = Field(default_factory=list)
    evidence_example_ids: list[str] = Field(min_length=1)
    expected_tradeoffs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class CandidateGenerator:
    """Turns one optimizer reply into a ``Candidate`` or a ``RejectedProposal``."""

    def __init__(self, client: LLMClient, spec: ModelSpec) -> None:
        self._client = client
        self._spec = spec

    async def propose(self, view: OptimizerView, attempt: int) -> Candidate | RejectedProposal:
        request = LLMRequest.for_model(self._spec, build_messages(view, attempt))
        try:
            output = await complete_json(self._client, request, OptimizerOutput)
        except InvalidModelReplyError as exc:
            return RejectedProposal(str(exc))
        raw = output.model_dump_json()
        revised = apply_output(view.current, output, {e.row_id for e in view.evidence})
        if isinstance(revised, str):
            return RejectedProposal(revised, raw)
        known = {e.row_id for e in view.evidence}
        return Candidate(
            template=revised,
            change_summary=output.change_summary.strip(),
            target_patterns=tuple(output.target_failure_patterns),
            evidence_ids=tuple(i for i in dict.fromkeys(output.evidence_example_ids) if i in known),
            tradeoffs=tuple(output.expected_tradeoffs),
            assumptions=tuple(output.assumptions),
            raw_response=raw,
        )


def apply_output(
    parent: PromptTemplate, output: OptimizerOutput, known_ids: set[str]
) -> PromptTemplate | str:
    """The revised template, or a message explaining why the reply cannot be used."""
    edited = _apply_edits(parent, output.messages)
    if isinstance(edited, str):
        return edited
    if not any(i in known_ids for i in output.evidence_example_ids):
        return "the reply cites no example ids from the data"
    candidate = replace(parent, messages=edited)
    if candidate == parent:
        return "the reply does not change the prompt"
    problems = validate_revision(parent, candidate)
    return "; ".join(problems) if problems else candidate


def _apply_edits(
    parent: PromptTemplate, edits: list[MessageEdit]
) -> tuple[PromptMessage, ...] | str:
    indices = [edit.index for edit in edits]
    if len(set(indices)) != len(indices):
        return "the reply edits the same message more than once"
    messages = list(parent.messages)
    for edit in edits:
        if edit.index > len(messages):
            return f"the reply edits message {edit.index}, which does not exist"
        original = messages[edit.index - 1]
        if not parent.is_editable(original):
            return f"the reply edits message {edit.index}, which is not editable"
        messages[edit.index - 1] = PromptMessage(original.role, edit.content)
    return tuple(messages)
