"""Everything the optimization loop needs, assembled once per command."""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from strop.config.enums import ModelRole
from strop.config.limits import BudgetsConfig
from strop.config.llm import ModelSpec
from strop.config.optimization import OptimizationConfig
from strop.evaluation.budget import BudgetTracker
from strop.evaluation.budgeted import BudgetedClient
from strop.llm.client import LLMClient
from strop.llm.factory import build_client
from strop.optimization import AcceptanceRules, CandidateGenerator
from strop.services.evaluator import Evaluator, build_evaluator
from strop.services.project import Project
from strop.storage.cache import ResponseCache
from strop.storage.prompts import PromptRepository
from strop.storage.runs import RunRepository

Say = Callable[[str], None]


@dataclass(frozen=True)
class LoopContext:
    project: Project
    settings: OptimizationConfig
    evaluator: Evaluator
    prompts: PromptRepository
    runs: RunRepository
    generator: CandidateGenerator
    analyzer_client: LLMClient
    analyzer_spec: ModelSpec
    rules: AcceptanceRules
    say: Say

    @property
    def budget(self) -> BudgetTracker:
        return self.evaluator.budget


def build_context(
    project: Project,
    conn: sqlite3.Connection,
    settings: OptimizationConfig,
    budgets: BudgetsConfig,
    *,
    use_cache: bool,
    say: Say,
) -> LoopContext:
    """Wire the evaluator, optimizer, analyzer and acceptance rules to one shared budget."""
    config = project.config
    budget = BudgetTracker(budgets)
    evaluator = build_evaluator(project, conn, use_cache=use_cache, budget=budget)
    cache = ResponseCache(conn) if use_cache else None
    helper = BudgetedClient(build_client(config.execution, cache), budget)
    optimizer = config.models.resolve(ModelRole.OPTIMIZER)
    return LoopContext(
        project=project,
        settings=settings,
        evaluator=evaluator,
        prompts=PromptRepository(conn),
        runs=RunRepository(conn),
        generator=CandidateGenerator(helper, optimizer),
        analyzer_client=helper,
        analyzer_spec=config.models.resolve(ModelRole.ANALYZER),
        rules=AcceptanceRules.from_config(settings, config.metrics.constraints),
        say=say,
    )
