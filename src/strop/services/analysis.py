"""Analyzing the failures of one prompt version on the train partition."""

import sqlite3
from dataclasses import dataclass
from typing import Any

from strop.analysis import (
    AnalysisFacts,
    FailedExample,
    Pattern,
    build_facts,
    find_patterns,
    select_evidence,
)
from strop.config.enums import ModelRole
from strop.config.llm import ModelSpec
from strop.config.optimization import OptimizationConfig
from strop.config.overrides import apply_overrides
from strop.data.models import Partition
from strop.errors import InvalidModelReplyError
from strop.evaluation.budget import BudgetTracker
from strop.evaluation.budgeted import BudgetedClient
from strop.evaluation.models import EvaluationRecord
from strop.llm.client import LLMClient
from strop.llm.factory import build_client
from strop.prompts.versions import PromptVersion
from strop.services.evaluate import sample_rows
from strop.services.evaluator import Evaluator, RunTag, build_evaluator
from strop.services.project import Project
from strop.services.prompts import ensure_initial_version, resolve_version
from strop.storage.cache import ResponseCache
from strop.storage.prompts import PromptRepository


@dataclass(frozen=True)
class TurnAnalysis:
    train_record: EvaluationRecord
    facts: AnalysisFacts
    evidence: tuple[FailedExample, ...]
    patterns: tuple[Pattern, ...]

    def to_json(self) -> dict[str, Any]:
        """The analysis as stored with a turn. Patterns are labeled as hypotheses."""
        return {
            "facts": {
                "examples": self.facts.total,
                "failed": self.facts.failed,
                "codes": dict(self.facts.codes),
                "confusions": [
                    {"expected": e, "got": g, "count": n} for e, g, n in self.facts.confusions
                ],
            },
            "hypotheses": [
                {"name": p.name, "description": p.description, "examples": list(p.example_ids)}
                for p in self.patterns
            ],
        }


async def analyze_version(
    evaluator: Evaluator,
    version: PromptVersion,
    settings: OptimizationConfig,
    *,
    seed: int,
    analyzer: tuple[LLMClient, ModelSpec] | None,
    run_id: int | None = None,
) -> TurnAnalysis:
    """Evaluate ``version`` on train, then collect facts, evidence, and (optionally) patterns."""
    rows = evaluator.rows(Partition.TRAIN)
    sampled = settings.train_sample is not None and settings.train_sample < len(rows)
    if sampled and settings.train_sample is not None:
        rows = sample_rows(rows, settings.train_sample, seed)
    record = None if sampled else evaluator.reusable(version, Partition.TRAIN)
    if record is None:
        tag = RunTag("analysis", {"sampled": sampled}, run_id)
        record = await evaluator.evaluate(version, Partition.TRAIN, tag, rows=rows)
    outcomes = evaluator.repository.outcomes(record.id)
    config = evaluator.project.config
    facts = build_facts(outcomes, rows, config.output.type, evaluator.suite.primary)
    evidence = select_evidence(facts.failures, settings.evidence_examples, seed)
    patterns: list[Pattern] = []
    if analyzer is not None and evidence:
        client, spec = analyzer
        try:
            patterns = await find_patterns(
                client,
                spec,
                task=config.project.task_description,
                output=config.output,
                facts=facts,
                evidence=evidence,
                max_chars=settings.evidence_chars,
            )
        except InvalidModelReplyError:
            patterns = []
    return TurnAnalysis(record, facts, tuple(evidence), tuple(patterns))


async def run_analysis(
    project: Project,
    conn: sqlite3.Connection,
    ref: str | None,
    *,
    patterns: bool,
    examples: int | None,
    use_cache: bool,
) -> tuple[PromptVersion, TurnAnalysis]:
    """Analyze the train failures of a prompt version, optionally asking for patterns."""
    repo = PromptRepository(conn)
    ensure_initial_version(project, repo)
    version = resolve_version(repo, ref)
    settings = apply_overrides(project.config.optimization, {"evidence_examples": examples})
    budget = BudgetTracker(project.config.budgets)
    evaluator = build_evaluator(project, conn, use_cache=use_cache, budget=budget)
    analyzer = None
    if patterns:
        cache = ResponseCache(conn) if use_cache else None
        client = BudgetedClient(build_client(project.config.execution, cache), budget)
        analyzer = (client, project.config.models.resolve(ModelRole.ANALYZER))
    analysis = await analyze_version(
        evaluator, version, settings, seed=evaluator.partitions.seed, analyzer=analyzer
    )
    return version, analysis
