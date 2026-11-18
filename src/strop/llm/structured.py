"""Requesting JSON from a model and validating it against a Pydantic schema."""

import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from strop.errors import InvalidModelReplyError
from strop.llm.client import LLMClient
from strop.llm.types import LLMRequest, Message

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """The JSON object in ``text``, with a surrounding code fence or prose removed."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        return fenced.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start : end + 1] if 0 <= start < end else stripped


async def complete_json(
    client: LLMClient, request: LLMRequest, schema: type[T], *, attempts: int = 2
) -> T:
    """Ask for JSON and validate it; on a bad reply, show the model its error and retry."""
    messages = list(request.messages)
    problem = ""
    for _ in range(max(1, attempts)):
        response = await client.complete(request.model_copy(update={"messages": tuple(messages)}))
        try:
            return schema.model_validate_json(extract_json(response.text))
        except ValidationError as exc:
            problem = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'reply'}: {e['msg']}" for e in exc.errors()
            )
        messages += [
            Message(role="assistant", content=response.text),
            Message(
                role="user",
                content=(
                    f"That reply was not valid ({problem}). "
                    "Reply again with only the corrected JSON object."
                ),
            ),
        ]
    raise InvalidModelReplyError(
        f"the model did not return valid JSON after {attempts} attempt(s): {problem}"
    )
