"""One optimization turn: analyze, propose, test, and decide."""

from dataclasses import dataclass

from strop.config.enums import ApprovalMode
from strop.data.models import Partition
from strop.evaluation.models import EvaluationRecord
from strop.formatting import fmt
from strop.optimization import (
    Candidate,
    HistoryEntry,
    MetricSummary,
    OptimizerView,
    RejectedProposal,
    compare,
)
from strop.prompts.versions import PromptVersion
from strop.services.analysis import TurnAnalysis, analyze_version
from strop.services.context import LoopContext
from strop.services.evaluator import RunTag
from strop.storage.runs import CandidateRecord, NewCandidate, RunRecord

HISTORY_LIMIT = 8


@dataclass(frozen=True)
class TurnResult:
    best: PromptVersion
    improved: bool = False
    awaiting: CandidateRecord | None = None
    no_failures: bool = False


class TurnRunner:
    def __init__(self, ctx: LoopContext, run: RunRecord) -> None:
        self._ctx = ctx
        self._run = run

    async def validation_record(self, version: PromptVersion) -> EvaluationRecord:
        """The validation result for ``version``: a stored one if the settings match."""
        evaluator = self._ctx.evaluator
        found = evaluator.reusable(version, Partition.VALIDATION)
        if found is not None:
            return found
        tag = RunTag("best", run_id=self._run.id)
        return await evaluator.evaluate(version, Partition.VALIDATION, tag)

    async def run(self, number: int, best: PromptVersion) -> TurnResult:
        ctx, run = self._ctx, self._run
        if not any(t.number == number for t in ctx.runs.turns(run.id)):
            ctx.runs.create_turn(run.id, number, best.number)
        best_record = await self.validation_record(best)
        value = best_record.report.primary_value.value
        ctx.say(f"Turn {number}: {best.ref} scores {best_record.report.primary} {fmt(value)}.")

        analysis = await analyze_version(
            ctx.evaluator,
            best,
            ctx.settings,
            seed=ctx.evaluator.partitions.seed + number,
            analyzer=(ctx.analyzer_client, ctx.analyzer_spec),
            run_id=run.id,
        )
        ctx.runs.update_turn(
            run.id,
            number,
            train_evaluation_id=analysis.train_record.id,
            analysis=analysis.to_json(),
        )
        if analysis.facts.failed == 0:
            ctx.say("  No failures on the training examples; nothing to improve.")
            ctx.runs.update_turn(run.id, number, status="done", best_after=best.number)
            return TurnResult(best, no_failures=True)
        ctx.say(
            f"  Analyzed {analysis.facts.failed} failed training example(s); "
            f"{len(analysis.patterns)} pattern hypothesis(es)."
        )
        view = self._view(best_record, best, analysis)
        existing = len(ctx.runs.candidates(run.id, number))
        for attempt in range(existing, ctx.settings.candidates_per_turn):
            await self._attempt(number, attempt, best, best_record, view)
        return self._conclude(number, best)

    def _view(
        self, record: EvaluationRecord, best: PromptVersion, analysis: TurnAnalysis
    ) -> OptimizerView:
        ctx = self._ctx
        config = ctx.project.config
        primary = record.report.primary_value
        history = tuple(
            HistoryEntry(c.change_summary or c.reason, c.status, c.reason, c.primary_delta)
            for c in ctx.runs.recent(HISTORY_LIMIT)
        )
        return OptimizerView(
            task=config.project.task_description,
            output=config.output,
            current=best.template,
            metrics=MetricSummary(
                record.report.primary,
                primary.value,
                ctx.evaluator.suite.direction(record.report.primary),
                config.metrics.constraints,
                primary.details.get("per_class", {}),
            ),
            facts=analysis.facts,
            evidence=analysis.evidence,
            patterns=analysis.patterns,
            history=history,
            instructions=ctx.settings.instructions,
            evidence_chars=ctx.settings.evidence_chars,
        )

    async def _attempt(
        self,
        number: int,
        attempt: int,
        best: PromptVersion,
        best_record: EvaluationRecord,
        view: OptimizerView,
    ) -> None:
        ctx, run = self._ctx, self._run
        proposal = await ctx.generator.propose(view, (number - 1) + attempt)
        if isinstance(proposal, RejectedProposal):
            ctx.say(f"  Candidate {attempt + 1}: discarded ({proposal.reason})")
            ctx.runs.add_candidate(
                NewCandidate(
                    run.id,
                    number,
                    best.number,
                    "invalid",
                    reason=proposal.reason,
                    raw_response=proposal.raw_response,
                )
            )
            return
        duplicate = ctx.prompts.find_by_hash(proposal.template.content_hash)
        if duplicate is not None:
            reason = f"identical to the existing version {duplicate.ref}"
            ctx.say(f"  Candidate {attempt + 1}: discarded ({reason})")
            ctx.runs.add_candidate(
                _candidate_row(run.id, number, best, proposal, "invalid", reason)
            )
            return
        version = ctx.prompts.add(
            proposal.template,
            parent_number=best.number,
            change_summary=proposal.change_summary,
            reason=f"run {run.id}, turn {number}",
            source="optimizer",
        )
        record = await ctx.evaluator.evaluate(
            version, Partition.VALIDATION, RunTag("candidate", {"turn": number}, run.id)
        )
        decision = compare(
            best_record,
            ctx.evaluator.repository.outcomes(best_record.id),
            record,
            ctx.evaluator.repository.outcomes(record.id),
            ctx.rules,
            ctx.evaluator.suite.direction(best_record.report.primary),
        )
        ctx.runs.add_candidate(
            NewCandidate(
                run.id,
                number,
                best.number,
                "evaluated",
                reason=decision.summary,
                version_number=version.number,
                change_summary=proposal.change_summary,
                patterns=list(proposal.target_patterns),
                evidence=list(proposal.evidence_ids),
                tradeoffs=list(proposal.tradeoffs),
                assumptions=list(proposal.assumptions),
                raw_response=proposal.raw_response,
                evaluation_id=record.id,
                passes=decision.passes,
                primary_delta=decision.primary_delta,
                wins=decision.wins,
                losses=decision.losses,
                ties=decision.ties,
            )
        )
        value = record.report.primary_value.value
        verdict = "passes all rules" if decision.passes else f"rejected: {decision.summary}"
        ctx.say(
            f"  Candidate {version.ref}: {proposal.change_summary} "
            f"[{record.report.primary} {fmt(value)}, wins {decision.wins} / losses "
            f"{decision.losses}] {verdict}"
        )

    def _conclude(self, number: int, best: PromptVersion) -> TurnResult:
        ctx, run = self._ctx, self._run
        evaluated = [c for c in ctx.runs.candidates(run.id, number) if c.status == "evaluated"]
        passers = [c for c in evaluated if c.passes]
        winner = max(passers, key=lambda c: c.primary_delta or 0.0, default=None)
        for candidate in evaluated:
            if candidate is winner:
                continue
            reason = candidate.reason
            if candidate.passes and winner is not None:
                reason = f"passed all rules, but v{winner.version_number} was chosen this turn"
            ctx.runs.set_candidate_status(candidate.id, "rejected", reason)
        if winner is None or winner.version_number is None:
            ctx.say("  No candidate was accepted; the best prompt is unchanged.")
            ctx.runs.update_turn(run.id, number, status="done", best_after=best.number)
            return TurnResult(best)
        if ctx.settings.approval is ApprovalMode.MANUAL:
            ctx.runs.set_candidate_status(
                winner.id, "pending", f"{winner.reason}; waiting for approval"
            )
            ctx.runs.update_turn(run.id, number, status="awaiting_approval")
            return TurnResult(best, awaiting=ctx.runs.get_candidate(winner.id))
        ctx.runs.set_candidate_status(winner.id, "accepted", winner.reason)
        ctx.prompts.set_best(winner.version_number)
        ctx.runs.update_turn(run.id, number, status="done", best_after=winner.version_number)
        ctx.say(f"  Accepted v{winner.version_number} as the new best prompt.")
        return TurnResult(ctx.prompts.get(winner.version_number), improved=True)


def _candidate_row(
    run_id: int, turn: int, best: PromptVersion, proposal: Candidate, status: str, reason: str
) -> NewCandidate:
    return NewCandidate(
        run_id,
        turn,
        best.number,
        status,
        reason=reason,
        change_summary=proposal.change_summary,
        patterns=list(proposal.target_patterns),
        evidence=list(proposal.evidence_ids),
        raw_response=proposal.raw_response,
    )
