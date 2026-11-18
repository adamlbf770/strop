"""Human decisions on candidates that are waiting for approval."""

import sqlite3

from strop.errors import ConfigError
from strop.prompts.versions import parse_version_ref
from strop.storage.prompts import PromptRepository
from strop.storage.runs import CandidateRecord, RunRepository


def _pending(runs: RunRepository, ref: str) -> CandidateRecord:
    number = parse_version_ref(ref)
    candidate = runs.for_version(number)
    if candidate is None or candidate.status != "pending":
        raise ConfigError(f"v{number} is not waiting for a decision")
    return candidate


def approve(conn: sqlite3.Connection, ref: str) -> CandidateRecord:
    """Make a pending candidate the best prompt and close its turn."""
    runs, prompts = RunRepository(conn), PromptRepository(conn)
    candidate = _pending(runs, ref)
    number = candidate.version_number
    if number is None:
        raise ConfigError(f"{ref} has no prompt version")
    prompts.set_best(number)
    runs.set_candidate_status(candidate.id, "accepted", f"{candidate.reason}; approved by the user")
    runs.update_turn(candidate.run_id, candidate.turn, status="done", best_after=number)
    runs.update_run(candidate.run_id, stale_turns=0)
    return runs.get_candidate(candidate.id)


def reject(conn: sqlite3.Connection, ref: str, reason: str = "") -> CandidateRecord:
    """Discard a pending candidate; the best prompt stays as it was."""
    runs = RunRepository(conn)
    candidate = _pending(runs, ref)
    why = f"rejected by the user: {reason}" if reason else "rejected by the user"
    runs.set_candidate_status(candidate.id, "rejected", why)
    turn = runs.get_turn(candidate.run_id, candidate.turn)
    runs.update_turn(candidate.run_id, candidate.turn, status="done", best_after=turn.best_before)
    run = runs.get_run(candidate.run_id)
    runs.update_run(candidate.run_id, stale_turns=run.stale_turns + 1)
    return runs.get_candidate(candidate.id)
