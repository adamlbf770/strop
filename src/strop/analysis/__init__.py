"""Failure analysis: deterministic facts first, LLM-proposed patterns as hypotheses."""

from strop.analysis.classify import classify_outcome
from strop.analysis.evidence import evidence_payload, select_evidence
from strop.analysis.facts import AnalysisFacts, FailedExample, build_facts
from strop.analysis.patterns import Pattern, find_patterns

__all__ = [
    "AnalysisFacts",
    "FailedExample",
    "Pattern",
    "build_facts",
    "classify_outcome",
    "evidence_payload",
    "find_patterns",
    "select_evidence",
]
