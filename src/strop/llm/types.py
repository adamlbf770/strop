"""Provider-independent request and response types."""

from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from strop.config.llm import ModelSpec

Role = Literal["system", "user", "assistant"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Message(_Frozen):
    role: Role
    content: str


class LLMRequest(_Frozen):
    model: str = Field(min_length=1)
    messages: tuple[Message, ...] = Field(min_length=1)
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_s: float = 60.0
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def for_model(cls, spec: ModelSpec, messages: Sequence[Message]) -> Self:
        """Build a request from a configured model and a list of messages."""
        return cls(
            model=spec.model,
            messages=tuple(messages),
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            timeout_s=spec.timeout_s,
            params=dict(spec.params),
        )


class Usage(_Frozen):
    """Token and cost data. ``None`` means the provider did not report it."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


class LLMResponse(_Frozen):
    text: str
    model: str
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    request_id: str | None = None
    cached: bool = False
