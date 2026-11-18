"""The LiteLLM-backed client. LiteLLM is imported lazily to keep CLI start-up fast."""

import logging
import os
import time
from types import ModuleType
from typing import Any

from strop.errors import LLMError, TransientLLMError
from strop.llm.types import LLMRequest, LLMResponse, Usage
from strop.log import redact

_TRANSIENT_NAMES = (
    "RateLimitError",
    "Timeout",
    "APIConnectionError",
    "ServiceUnavailableError",
    "InternalServerError",
)


def _load_litellm() -> ModuleType:
    # Use the price table bundled with LiteLLM instead of downloading one when it is imported.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm  # noqa: PLC0415 - deferred: importing litellm takes over a second

    litellm.suppress_debug_info = True
    litellm.telemetry = False
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    return litellm


class LiteLLMClient:
    """Calls any provider and model LiteLLM supports and normalizes the response."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        litellm = _load_litellm()
        started = time.perf_counter()
        try:
            raw = await litellm.acompletion(
                model=request.model,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=request.timeout_s,
                drop_params=True,
                **request.params,
            )
        except Exception as exc:
            raise _translate(litellm, exc) from exc
        return _normalize(litellm, raw, request, time.perf_counter() - started)


def _translate(litellm: ModuleType, exc: Exception) -> LLMError:
    transient = tuple(
        cls for name in _TRANSIENT_NAMES if isinstance(cls := getattr(litellm, name, None), type)
    )
    message = redact(f"{type(exc).__name__}: {exc}")
    return TransientLLMError(message) if isinstance(exc, transient) else LLMError(message)


def _normalize(litellm: ModuleType, raw: Any, request: LLMRequest, latency_s: float) -> LLMResponse:
    try:
        choice = raw.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMError("the provider returned a response without a message") from exc
    return LLMResponse(
        text=text,
        model=getattr(raw, "model", None) or request.model,
        finish_reason=finish_reason,
        usage=_usage(litellm, raw),
        latency_s=latency_s,
        request_id=getattr(raw, "id", None),
    )


def _usage(litellm: ModuleType, raw: Any) -> Usage:
    reported = getattr(raw, "usage", None)
    try:
        cost = float(litellm.completion_cost(completion_response=raw))
    except Exception:
        cost = None
    return Usage(
        prompt_tokens=getattr(reported, "prompt_tokens", None),
        completion_tokens=getattr(reported, "completion_tokens", None),
        total_tokens=getattr(reported, "total_tokens", None),
        cost_usd=cost,
    )
