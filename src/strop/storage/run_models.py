"""Records for optimization runs, turns, and candidates."""

from dataclasses import dataclass, field
from typing import Any

RESUMABLE = ("running", "paused", "interrupted")


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: int
    status: str
    start_prompt: int
    settings: dict[str, Any]
    stale_turns: int
    stop_reason: str | None
    model_calls: int
    cost_usd: float | None
    started_at: str
    finished_at: str | None


@dataclass(frozen=True, slots=True)
class TurnRecord:
    run_id: int
    number: int
    status: str
    best_before: int
    best_after: int | None
    train_evaluation_id: int | None
    analysis: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NewCandidate:
    run_id: int
    turn: int
    parent_number: int
    status: str
    reason: str = ""
    version_number: int | None = None
    change_summary: str = ""
    patterns: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    raw_response: str = ""
    evaluation_id: int | None = None
    passes: bool | None = None
    primary_delta: float | None = None
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    id: int
    run_id: int
    turn: int
    parent_number: int
    version_number: int | None
    change_summary: str
    patterns: list[str]
    evidence: list[str]
    tradeoffs: list[str]
    assumptions: list[str]
    evaluation_id: int | None
    passes: bool | None
    primary_delta: float | None
    wins: int | None
    losses: int | None
    ties: int | None
    status: str
    reason: str
