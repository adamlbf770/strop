"""Optimizer inputs and outputs.

``OptimizerView`` is everything the optimizer may see. It is built from train-partition
failures and validation scores only; it has no reference to the dataset partitions, so
test rows cannot reach the optimizer through it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from strop.analysis import AnalysisFacts, FailedExample, Pattern
from strop.config.metrics import Bound
from strop.config.output import OutputConfig
from strop.metrics.types import Direction
from strop.prompts import PromptTemplate


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """An earlier attempt, shown to the optimizer so it does not repeat itself."""

    summary: str
    outcome: str
    reason: str
    primary_delta: float | None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    primary: str
    value: float | None
    direction: Direction
    constraints: Mapping[str, Bound]
    per_class: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerView:
    task: str
    output: OutputConfig
    current: PromptTemplate
    metrics: MetricSummary
    facts: AnalysisFacts
    evidence: tuple[FailedExample, ...]
    patterns: tuple[Pattern, ...]
    history: tuple[HistoryEntry, ...]
    instructions: str
    evidence_chars: int


@dataclass(frozen=True, slots=True)
class Candidate:
    """A valid proposed revision with the reasoning the optimizer gave for it."""

    template: PromptTemplate
    change_summary: str
    target_patterns: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    assumptions: tuple[str, ...]
    raw_response: str


@dataclass(frozen=True, slots=True)
class RejectedProposal:
    """A proposal that could not be used, with the reason it was discarded."""

    reason: str
    raw_response: str = ""
