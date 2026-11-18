"""Persistence for optimization runs, turns, and candidates."""

import json
import sqlite3
from typing import Any

from strop.errors import StorageError
from strop.storage.run_models import (
    RESUMABLE,
    CandidateRecord,
    NewCandidate,
    RunRecord,
    TurnRecord,
)

__all__ = [
    "CandidateRecord",
    "NewCandidate",
    "RunRecord",
    "RunRepository",
    "TurnRecord",
]

_CANDIDATE_COLUMNS = (
    "id, run_id, turn, parent_number, version_number, change_summary, patterns_json, "
    "evidence_json, tradeoffs_json, assumptions_json, evaluation_id, passes, primary_delta, "
    "wins, losses, ties, status, reason"
)


class RunRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- runs ---------------------------------------------------------------------------
    def create_run(self, start_prompt: int, settings: dict[str, Any]) -> RunRecord:
        cursor = self._conn.execute(
            "INSERT INTO runs (status, start_prompt, settings_json) VALUES ('running', ?, ?)",
            (start_prompt, json.dumps(settings)),
        )
        return self.get_run(int(cursor.lastrowid or 0))

    def get_run(self, run_id: int) -> RunRecord:
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise StorageError(f"run {run_id} does not exist")
        return _run(row)

    def latest_run(self) -> RunRecord | None:
        row = self._conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return _run(row) if row else None

    def all_runs(self) -> list[RunRecord]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        return [_run(row) for row in rows]

    def resumable_run(self) -> RunRecord | None:
        marks = ", ".join("?" * len(RESUMABLE))
        row = self._conn.execute(
            f"SELECT * FROM runs WHERE status IN ({marks}) ORDER BY id DESC LIMIT 1", RESUMABLE
        ).fetchone()
        return _run(row) if row else None

    def update_run(
        self,
        run_id: int,
        *,
        status: str | None = None,
        stale_turns: int | None = None,
        stop_reason: str | None = None,
        model_calls: int | None = None,
        cost_usd: float | None = None,
        finished: bool = False,
    ) -> None:
        fields = {
            "status": status,
            "stale_turns": stale_turns,
            "stop_reason": stop_reason,
            "model_calls": model_calls,
            "cost_usd": cost_usd,
        }
        updates = {k: v for k, v in fields.items() if v is not None}
        assignments = [f"{key} = ?" for key in updates]
        if finished:
            assignments.append("finished_at = CURRENT_TIMESTAMP")
        if not assignments:
            return
        self._conn.execute(
            f"UPDATE runs SET {', '.join(assignments)} WHERE id = ?", (*updates.values(), run_id)
        )

    # -- turns --------------------------------------------------------------------------
    def create_turn(self, run_id: int, number: int, best_before: int) -> TurnRecord:
        self._conn.execute(
            "INSERT INTO turns (run_id, number, status, best_before) VALUES (?, ?, 'running', ?)",
            (run_id, number, best_before),
        )
        return self.get_turn(run_id, number)

    def get_turn(self, run_id: int, number: int) -> TurnRecord:
        row = self._conn.execute(
            "SELECT * FROM turns WHERE run_id = ? AND number = ?", (run_id, number)
        ).fetchone()
        if row is None:
            raise StorageError(f"turn {number} of run {run_id} does not exist")
        return _turn(row)

    def turns(self, run_id: int) -> list[TurnRecord]:
        rows = self._conn.execute(
            "SELECT * FROM turns WHERE run_id = ? ORDER BY number", (run_id,)
        ).fetchall()
        return [_turn(row) for row in rows]

    def update_turn(
        self,
        run_id: int,
        number: int,
        *,
        status: str | None = None,
        best_after: int | None = None,
        train_evaluation_id: int | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "status": status,
            "best_after": best_after,
            "train_evaluation_id": train_evaluation_id,
            "analysis_json": json.dumps(analysis) if analysis is not None else None,
        }
        updates = {k: v for k, v in fields.items() if v is not None}
        if updates:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            self._conn.execute(
                f"UPDATE turns SET {assignments} WHERE run_id = ? AND number = ?",
                (*updates.values(), run_id, number),
            )

    # -- candidates ---------------------------------------------------------------------
    def add_candidate(self, new: NewCandidate) -> CandidateRecord:
        cursor = self._conn.execute(
            "INSERT INTO candidates (run_id, turn, parent_number, version_number, change_summary, "
            "patterns_json, evidence_json, tradeoffs_json, assumptions_json, raw_response, "
            "evaluation_id, passes, primary_delta, wins, losses, ties, status, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new.run_id,
                new.turn,
                new.parent_number,
                new.version_number,
                new.change_summary,
                json.dumps(new.patterns),
                json.dumps(new.evidence),
                json.dumps(new.tradeoffs),
                json.dumps(new.assumptions),
                new.raw_response,
                new.evaluation_id,
                None if new.passes is None else int(new.passes),
                new.primary_delta,
                new.wins,
                new.losses,
                new.ties,
                new.status,
                new.reason,
            ),
        )
        return self.get_candidate(int(cursor.lastrowid or 0))

    def get_candidate(self, candidate_id: int) -> CandidateRecord:
        row = self._conn.execute(
            f"SELECT {_CANDIDATE_COLUMNS} FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise StorageError(f"candidate {candidate_id} does not exist")
        return _candidate(row)

    def set_candidate_status(self, candidate_id: int, status: str, reason: str) -> None:
        self._conn.execute(
            "UPDATE candidates SET status = ?, reason = ? WHERE id = ?",
            (status, reason, candidate_id),
        )

    def candidates(self, run_id: int, turn: int | None = None) -> list[CandidateRecord]:
        sql = f"SELECT {_CANDIDATE_COLUMNS} FROM candidates WHERE run_id = ?"
        args: list[Any] = [run_id]
        if turn is not None:
            sql += " AND turn = ?"
            args.append(turn)
        rows = self._conn.execute(sql + " ORDER BY id", args).fetchall()
        return [_candidate(row) for row in rows]

    def for_version(self, version_number: int) -> CandidateRecord | None:
        row = self._conn.execute(
            f"SELECT {_CANDIDATE_COLUMNS} FROM candidates WHERE version_number = ? "
            "ORDER BY id DESC LIMIT 1",
            (version_number,),
        ).fetchone()
        return _candidate(row) if row else None

    def pending(self) -> list[CandidateRecord]:
        rows = self._conn.execute(
            f"SELECT {_CANDIDATE_COLUMNS} FROM candidates WHERE status = 'pending' ORDER BY id"
        ).fetchall()
        return [_candidate(row) for row in rows]

    def recent(self, limit: int) -> list[CandidateRecord]:
        """The latest decided candidates across all runs, newest first."""
        rows = self._conn.execute(
            f"SELECT {_CANDIDATE_COLUMNS} FROM candidates WHERE status != 'evaluated' "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_candidate(row) for row in rows]


def _run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        row["id"],
        row["status"],
        row["start_prompt"],
        json.loads(row["settings_json"]),
        row["stale_turns"],
        row["stop_reason"],
        row["model_calls"],
        row["cost_usd"],
        row["started_at"],
        row["finished_at"],
    )


def _turn(row: sqlite3.Row) -> TurnRecord:
    return TurnRecord(
        row["run_id"],
        row["number"],
        row["status"],
        row["best_before"],
        row["best_after"],
        row["train_evaluation_id"],
        json.loads(row["analysis_json"]),
    )


def _candidate(row: sqlite3.Row) -> CandidateRecord:
    passes = row["passes"]
    return CandidateRecord(
        row["id"],
        row["run_id"],
        row["turn"],
        row["parent_number"],
        row["version_number"],
        row["change_summary"],
        json.loads(row["patterns_json"]),
        json.loads(row["evidence_json"]),
        json.loads(row["tradeoffs_json"]),
        json.loads(row["assumptions_json"]),
        row["evaluation_id"],
        None if passes is None else bool(passes),
        row["primary_delta"],
        row["wins"],
        row["losses"],
        row["ties"],
        row["status"],
        row["reason"],
    )
