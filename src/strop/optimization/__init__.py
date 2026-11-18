"""Prompt optimization: proposing revisions and deciding whether to keep them."""

from strop.optimization.acceptance import AcceptanceRules, Decision, compare
from strop.optimization.generator import CandidateGenerator, OptimizerOutput, apply_output
from strop.optimization.types import (
    Candidate,
    HistoryEntry,
    MetricSummary,
    OptimizerView,
    RejectedProposal,
)

__all__ = [
    "AcceptanceRules",
    "Candidate",
    "CandidateGenerator",
    "Decision",
    "HistoryEntry",
    "MetricSummary",
    "OptimizerOutput",
    "OptimizerView",
    "RejectedProposal",
    "apply_output",
    "compare",
]
