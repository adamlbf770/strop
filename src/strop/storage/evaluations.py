"""Persistence for evaluations and their per-example results."""

import json
import sqlite3
from dataclasses import asdict
from typing import Any

from strop.data.models import Partition
from strop.errors import StorageError
from strop.evaluation.models import (
    EvaluationRecord,
    ExampleOutcome,
    NewEvaluation,
    UsageSummary,
    summarize_usage,
)
from strop.metrics.suite import MetricReport
from strop.storage.database import transaction

_OUTCOME_COLUMNS = (
    "row_id, expected, raw_output, value, parsed_json, error, detail, finish_reason, latency_s, "
    "prompt_tokens, completion_tokens, total_tokens, cost_usd, cached, scores_json, feedback"
)


_INSERT_OUTCOME = (
    f"INSERT INTO example_results (evaluation_id, {_OUTCOME_COLUMNS}) "
    f"VALUES ({', '.join('?' * 17)})"
)


class EvaluationRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, new: NewEvaluation) -> EvaluationRecord:
        """Store an evaluation and all of its example results atomically."""
        usage = summarize_usage(new.outcomes)
        with transaction(self._conn):
            cursor = self._conn.execute(
                "INSERT INTO evaluations (prompt_number, split_id, partition, kind, run_id, "
                "metrics_json, usage_json, details_json, sample_count, error_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new.prompt_number,
                    new.split_id,
                    new.partition.value,
                    new.kind,
                    new.run_id,
                    json.dumps(new.report.to_dict()),
                    json.dumps(asdict(usage)),
                    json.dumps(new.details),
                    len(new.outcomes),
                    sum(o.failed for o in new.outcomes),
                ),
            )
            evaluation_id = int(cursor.lastrowid or 0)
            self._conn.executemany(
                _INSERT_OUTCOME, [self._row(evaluation_id, o) for o in new.outcomes]
            )
        return self.get(evaluation_id)

    def get(self, evaluation_id: int) -> EvaluationRecord:
        row = self._conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
        if row is None:
            raise StorageError(f"evaluation {evaluation_id} does not exist")
        return _record(row)

    def find(
        self,
        *,
        prompt_number: int | None = None,
        partition: Partition | None = None,
        kind: str | None = None,
        limit: int = 20,
    ) -> list[EvaluationRecord]:
        """Matching evaluations, newest first."""
        clauses, args = [], []
        for column, value in (
            ("prompt_number", prompt_number),
            ("partition", partition.value if partition else None),
            ("kind", kind),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                args.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM evaluations {where} ORDER BY id DESC LIMIT ?", (*args, limit)
        ).fetchall()
        return [_record(row) for row in rows]

    def outcomes(self, evaluation_id: int) -> list[ExampleOutcome]:
        rows = self._conn.execute(
            f"SELECT {_OUTCOME_COLUMNS} FROM example_results WHERE evaluation_id = ? "
            "ORDER BY rowid",
            (evaluation_id,),
        ).fetchall()
        return [_outcome(row) for row in rows]

    @staticmethod
    def _row(evaluation_id: int, o: ExampleOutcome) -> tuple[Any, ...]:
        return (
            evaluation_id,
            o.row_id,
            o.expected,
            o.raw_output,
            o.value,
            json.dumps(o.parsed) if o.parsed is not None else None,
            o.error,
            o.detail,
            o.finish_reason,
            o.latency_s,
            o.prompt_tokens,
            o.completion_tokens,
            o.total_tokens,
            o.cost_usd,
            int(o.cached),
            json.dumps(o.scores),
            o.feedback,
        )


def _record(row: sqlite3.Row) -> EvaluationRecord:
    return EvaluationRecord(
        id=row["id"],
        prompt_number=row["prompt_number"],
        split_id=row["split_id"],
        partition=Partition(row["partition"]),
        kind=row["kind"],
        report=MetricReport.from_dict(json.loads(row["metrics_json"])),
        usage=UsageSummary(**json.loads(row["usage_json"])),
        sample_count=row["sample_count"],
        error_count=row["error_count"],
        created_at=row["created_at"],
        details=json.loads(row["details_json"]),
    )


def _outcome(row: sqlite3.Row) -> ExampleOutcome:
    return ExampleOutcome(
        row_id=row["row_id"],
        expected=row["expected"],
        raw_output=row["raw_output"],
        value=row["value"],
        parsed=json.loads(row["parsed_json"]) if row["parsed_json"] else None,
        error=row["error"],
        detail=row["detail"],
        finish_reason=row["finish_reason"],
        latency_s=row["latency_s"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        total_tokens=row["total_tokens"],
        cost_usd=row["cost_usd"],
        cached=bool(row["cached"]),
        scores=json.loads(row["scores_json"]),
        feedback=row["feedback"],
    )
