"""Running a prompt over rows with bounded concurrency."""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from strop.config.limits import ExecutionConfig
from strop.config.llm import ModelSpec
from strop.config.output import OutputConfig
from strop.data.models import Row
from strop.errors import LLMError
from strop.evaluation.budget import BudgetTracker
from strop.evaluation.models import ExampleOutcome
from strop.evaluation.parsing import canonical_expected, parse_output
from strop.llm.client import LLMClient
from strop.llm.types import LLMRequest, LLMResponse
from strop.log import redact
from strop.metrics.types import ErrorCode
from strop.prompts import PromptTemplate, render_messages

Progress = Callable[[int, int], None]


class PromptRunner:
    """Sends one request per row and turns each reply into an ``ExampleOutcome``."""

    def __init__(
        self,
        client: LLMClient,
        spec: ModelSpec,
        output: OutputConfig,
        schema: dict[str, Any] | None,
        execution: ExecutionConfig,
        budget: BudgetTracker | None = None,
    ) -> None:
        self._client = client
        self._spec = spec
        self._output = output
        self._schema = schema
        self._execution = execution
        self._budget = budget

    async def run(
        self,
        template: PromptTemplate,
        rows: Sequence[Row],
        progress: Progress | None = None,
    ) -> list[ExampleOutcome]:
        """Evaluate ``rows`` in batches, checking the budget between batches."""
        semaphore = asyncio.Semaphore(self._execution.max_concurrency)
        outcomes: list[ExampleOutcome] = []
        for start in range(0, len(rows), self._execution.batch_size):
            if self._budget is not None:
                self._budget.check()
            batch = rows[start : start + self._execution.batch_size]
            outcomes.extend(
                await asyncio.gather(*(self._one(template, row, semaphore) for row in batch))
            )
            if progress is not None:
                progress(len(outcomes), len(rows))
        return outcomes

    async def _one(
        self, template: PromptTemplate, row: Row, semaphore: asyncio.Semaphore
    ) -> ExampleOutcome:
        expected = (
            canonical_expected(row.expected, self._output.type)
            if row.expected is not None
            else None
        )
        request = LLMRequest.for_model(self._spec, render_messages(template, row.inputs))
        async with semaphore:
            try:
                response = await self._client.complete(request)
            except LLMError as exc:
                if self._budget is not None:
                    self._budget.record_failed_call()
                return ExampleOutcome(
                    row.id,
                    expected,
                    "",
                    None,
                    error=ErrorCode.PROVIDER_ERROR,
                    detail=redact(str(exc)),
                )
        if self._budget is not None:
            self._budget.record(response)
        return self._outcome(row, expected, response)

    def _outcome(self, row: Row, expected: str | None, response: LLMResponse) -> ExampleOutcome:
        parsed = parse_output(response.text, response.finish_reason, self._output, self._schema)
        usage = response.usage
        return ExampleOutcome(
            row_id=row.id,
            expected=expected,
            raw_output=response.text,
            value=parsed.value,
            parsed=parsed.parsed,
            error=parsed.error,
            finish_reason=response.finish_reason,
            latency_s=response.latency_s,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
            cached=response.cached,
        )
