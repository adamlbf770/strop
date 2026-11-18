"""The optimization loop: run turns until a stop condition is reached."""

import asyncio
import sqlite3
from dataclasses import dataclass, field

from strop.config.limits import BudgetsConfig
from strop.config.optimization import OptimizationConfig
from strop.data.models import Partition
from strop.errors import BudgetExceededError, ConfigError, DataError
from strop.evaluation.models import EvaluationRecord
from strop.metrics.types import Direction
from strop.prompts.versions import PromptVersion
from strop.services.context import LoopContext, Say, build_context
from strop.services.project import Project
from strop.services.prompts import ensure_initial_version
from strop.services.turn import TurnRunner
from strop.storage.runs import CandidateRecord, RunRecord, TurnRecord


@dataclass(frozen=True)
class RunSummary:
    """What an optimization run did and why it stopped."""

    run: RunRecord
    initial: PromptVersion
    start: PromptVersion
    final: PromptVersion
    baseline: EvaluationRecord | None
    result: EvaluationRecord | None
    turns: list[TurnRecord]
    candidates: list[CandidateRecord]
    stop_reason: str
    stop_detail: str
    pending: CandidateRecord | None
    calls: int
    cost_usd: float | None
    warnings: list[str] = field(default_factory=list)


async def optimize(
    project: Project,
    conn: sqlite3.Connection,
    settings: OptimizationConfig,
    budgets: BudgetsConfig,
    *,
    resume: bool,
    use_cache: bool,
    say: Say,
) -> RunSummary:
    """Run (or resume) the loop: analyze failures, propose revisions, test, and decide."""
    ctx = build_context(project, conn, settings, budgets, use_cache=use_cache, say=say)
    validation = len(ctx.evaluator.rows(Partition.VALIDATION))
    if validation < settings.min_validation_samples:
        raise DataError(
            f"the validation partition has {validation} rows, fewer than the "
            f"{settings.min_validation_samples} needed to compare prompts; add data"
        )
    run = _begin(ctx, resume)
    runner = TurnRunner(ctx, run)
    reason, detail = "", ""
    try:
        reason = await _turns(ctx, run, runner)
    except BudgetExceededError as exc:
        reason, detail = "budget", str(exc)
    except (KeyboardInterrupt, asyncio.CancelledError):
        _save(ctx, run.id, status="interrupted")
        raise
    except BaseException as exc:
        _save(ctx, run.id, status="failed", stop_reason=str(exc)[:500])
        raise
    status = {"awaiting_approval": "paused", "budget": "interrupted"}.get(reason, "completed")
    _save(ctx, run.id, status=status, stop_reason=reason, finished=status == "completed")
    return _summary(ctx, run.id, reason, detail)


def _begin(ctx: LoopContext, resume: bool) -> RunRecord:
    pending = ctx.runs.pending()
    if pending:
        ref = f"v{pending[0].version_number}"
        raise ConfigError(
            f"{ref} is waiting for a decision; run 'strop approve {ref}' or "
            f"'strop reject {ref}' first"
        )
    if resume:
        run = ctx.runs.resumable_run()
        if run is None:
            raise ConfigError(
                "there is no unfinished run to resume; start one with 'strop optimize'"
            )
        ctx.runs.update_run(run.id, status="running")
        return ctx.runs.get_run(run.id)
    unfinished = ctx.runs.resumable_run()
    if unfinished is not None:
        ctx.runs.update_run(unfinished.id, status="interrupted")
    best = ctx.prompts.best() or ensure_initial_version(ctx.project, ctx.prompts)
    return ctx.runs.create_run(best.number, ctx.settings.model_dump(mode="json"))


async def _turns(ctx: LoopContext, run: RunRecord, runner: TurnRunner) -> str:
    stale = run.stale_turns
    while True:
        turns = ctx.runs.turns(run.id)
        if sum(t.status == "done" for t in turns) >= ctx.settings.max_turns:
            return "max_turns"
        best = _best(ctx)
        record = await runner.validation_record(best)
        if _target_reached(ctx, record):
            return "target_reached"
        open_turn = next((t for t in turns if t.status != "done"), None)
        number = open_turn.number if open_turn else (turns[-1].number + 1 if turns else 1)
        result = await runner.run(number, best)
        if result.awaiting is not None:
            return "awaiting_approval"
        stale = 0 if result.improved else stale + 1
        _save(ctx, run.id, stale_turns=stale)
        if result.no_failures:
            return "no_failures"
        if stale >= ctx.settings.patience:
            return "patience"


def _best(ctx: LoopContext) -> PromptVersion:
    best = ctx.prompts.best()
    if best is None:
        raise ConfigError("no best prompt is set; run 'strop baseline' first")
    return best


def _target_reached(ctx: LoopContext, record: EvaluationRecord) -> bool:
    target = ctx.settings.target
    value = record.report.primary_value.value
    if target is None or value is None:
        return False
    if ctx.evaluator.suite.direction(record.report.primary) is Direction.MAXIMIZE:
        return value >= target
    return value <= target


def _cost(ctx: LoopContext) -> float | None:
    """The known cost so far, or None when no call reported one."""
    budget = ctx.budget
    return budget.cost_usd if budget.calls > budget.cost_unknown_calls else None


def _save(
    ctx: LoopContext,
    run_id: int,
    *,
    status: str | None = None,
    stale_turns: int | None = None,
    stop_reason: str | None = None,
    finished: bool = False,
) -> None:
    """Update a run and record the calls and cost used so far."""
    ctx.runs.update_run(
        run_id,
        status=status,
        stale_turns=stale_turns,
        stop_reason=stop_reason,
        model_calls=ctx.budget.calls,
        cost_usd=_cost(ctx),
        finished=finished,
    )


def _summary(ctx: LoopContext, run_id: int, reason: str, detail: str) -> RunSummary:
    run = ctx.runs.get_run(run_id)
    start = ctx.prompts.get(run.start_prompt)
    final = _best(ctx)
    pending = next(iter(ctx.runs.pending()), None)
    return RunSummary(
        run=run,
        initial=ctx.prompts.first() or start,
        start=start,
        final=final,
        baseline=ctx.evaluator.reusable(start, Partition.VALIDATION),
        result=ctx.evaluator.reusable(final, Partition.VALIDATION),
        turns=ctx.runs.turns(run_id),
        candidates=ctx.runs.candidates(run_id),
        stop_reason=reason,
        stop_detail=detail,
        pending=pending,
        calls=ctx.budget.calls,
        cost_usd=_cost(ctx),
        warnings=ctx.budget.warnings(),
    )
