"""Evaluation result types."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from strop.data.models import Partition
from strop.metrics.suite import MetricReport


@dataclass(frozen=True, slots=True)
class ExampleOutcome:
    """The stored result for one row: the model output, its parse, and its per-metric scores."""

    row_id: str
    expected: str | None
    raw_output: str
    value: str | None
    parsed: Any = None
    error: str | None = None
    finish_reason: str | None = None
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cached: bool = False
    detail: str | None = None
    scores: dict[str, float | None] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Calls, tokens, cost and latency. ``None`` means the provider reported nothing."""

    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cost_known_calls: int = 0
    mean_latency_s: float | None = None

    @property
    def cost_complete(self) -> bool:
        """True if every non-cached call reported a cost."""
        return self.cost_known_calls == self.calls - self.cached_calls


def _sum_known(values: Sequence[int | float | None]) -> Any:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def summarize_usage(outcomes: Sequence[ExampleOutcome]) -> UsageSummary:
    """Totals over ``outcomes``. Cached answers add no tokens or cost and no new call."""
    fresh = [o for o in outcomes if not o.cached]
    latencies = [o.latency_s for o in fresh if o.latency_s > 0]
    return UsageSummary(
        calls=len(outcomes),
        cached_calls=len(outcomes) - len(fresh),
        prompt_tokens=_sum_known([o.prompt_tokens for o in fresh]),
        completion_tokens=_sum_known([o.completion_tokens for o in fresh]),
        total_tokens=_sum_known([o.total_tokens for o in fresh]),
        cost_usd=_sum_known([o.cost_usd for o in fresh]),
        cost_known_calls=sum(o.cost_usd is not None for o in fresh),
        mean_latency_s=sum(latencies) / len(latencies) if latencies else None,
    )


@dataclass(frozen=True)
class EvaluationRecord:
    """A stored evaluation of one prompt version on one partition."""

    id: int
    prompt_number: int
    split_id: int
    partition: Partition
    kind: str
    report: MetricReport
    usage: UsageSummary
    sample_count: int
    error_count: int
    created_at: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NewEvaluation:
    """Everything needed to store an evaluation."""

    prompt_number: int
    split_id: int
    partition: Partition
    kind: str
    report: MetricReport
    outcomes: Sequence[ExampleOutcome]
    details: dict[str, Any] = field(default_factory=dict)
    run_id: int | None = None
