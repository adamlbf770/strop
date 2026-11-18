"""Collecting everything a final report needs from the project database."""

import sqlite3
from datetime import UTC, datetime
from typing import Any

from strop import __version__
from strop.analysis import build_facts
from strop.data.models import Partition
from strop.evaluation.models import EvaluationRecord
from strop.prompts.diff import diff_templates
from strop.prompts.versions import PromptVersion
from strop.reporting.evaluation_view import record_payload
from strop.services.project import Project
from strop.storage.datasets import DatasetRepository
from strop.storage.evaluations import EvaluationRepository
from strop.storage.prompts import PromptRepository
from strop.storage.runs import RunRecord, RunRepository

SMALL_VALIDATION = 30


def build_report(
    project: Project, conn: sqlite3.Connection, selected: PromptVersion
) -> dict[str, Any]:
    """A machine-readable report for ``selected``; missing data is marked unavailable."""
    prompts, evals = PromptRepository(conn), EvaluationRepository(conn)
    datasets, runs = DatasetRepository(conn), RunRepository(conn)
    initial = prompts.first() or selected
    split = datasets.latest_split()
    counts = datasets.partition_counts(split.id) if split else {}
    results = {
        "initial_validation": _latest(evals, initial, Partition.VALIDATION),
        "selected_validation": _latest(evals, selected, Partition.VALIDATION),
        "initial_test": _latest(evals, initial, Partition.TEST),
        "selected_test": _latest(evals, selected, Partition.TEST),
    }
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "strop_version": __version__,
        "project": {
            "name": project.config.project.name,
            "task": project.config.project.task_description,
        },
        "dataset": _dataset(datasets, split.dataset_id) if split else None,
        "split": (
            {"id": split.id, "seed": split.seed, "counts": {p.value: n for p, n in counts.items()}}
            if split
            else None
        ),
        "models": _models(project),
        "metrics": _metrics(project),
        "prompt": {
            "selected": selected.number,
            "initial": initial.number,
            "best": prompts.best_number(),
        },
        "results": {
            key: record_payload(rec, f"v{rec.prompt_number}") if rec else None
            for key, rec in results.items()
        },
        "per_class": _per_class(results["selected_validation"]),
        "versions": [_version(v, runs.for_version(v.number)) for v in prompts.list()],
        "runs": [_run(r, runs) for r in runs.all_runs()],
        "diff": diff_templates(initial.template, selected.template).to_payload(),
        "error_analysis": _error_analysis(project, evals, selected),
        "usage": _usage(evals),
    }
    payload["limitations"] = limitations(payload)
    return payload


def limitations(report: dict[str, Any]) -> list[str]:
    """Plain statements about what this report cannot show."""
    notes = []
    validation = (report["results"].get("selected_validation") or {}).get("samples")
    if validation is not None and validation < SMALL_VALIDATION:
        notes.append(
            f"The validation partition has only {validation} examples, so small score "
            "differences may be noise."
        )
    if (report["split"] or {}).get("counts", {}).get("test", 0) == 0:
        notes.append("There is no test partition, so there is no held-out result.")
    elif report["results"].get("selected_test") is None:
        notes.append("The selected prompt has not been evaluated on the test partition yet.")
    if any(m["subjective"] for r in report["results"].values() if r for m in r["metrics"].values()):
        notes.append("LLM-judged metrics are subjective estimates, not ground truth.")
    if report["usage"]["cost_usd"] is None:
        notes.append("Cost is unavailable: no provider call reported a price.")
    notes.append("Results come from one dataset split and depend on the models that were used.")
    return notes


def _latest(
    evals: EvaluationRepository, version: PromptVersion, partition: Partition
) -> EvaluationRecord | None:
    found = evals.find(prompt_number=version.number, partition=partition, limit=1)
    return found[0] if found else None


def _dataset(datasets: DatasetRepository, dataset_id: int) -> dict[str, Any]:
    info = datasets.dataset_info(dataset_id)
    return {
        "path": info.source_path,
        "content_hash": info.content_hash,
        "rows": info.row_count,
        "skipped_rows": info.skipped_rows,
    }


def _models(project: Project) -> dict[str, str | None]:
    models = project.config.models
    return {
        role: (spec.model if spec else None)
        for role, spec in (
            ("target", models.target),
            ("optimizer", models.optimizer),
            ("analyzer", models.analyzer),
            ("judge", models.judge),
        )
    }


def _metrics(project: Project) -> dict[str, Any]:
    config = project.config.metrics
    return {
        "primary": config.primary,
        "secondary": list(config.secondary),
        "constraints": {k: b.model_dump(exclude_none=True) for k, b in config.constraints.items()},
        "rules": sorted(config.rules),
        "judges": sorted(config.judges),
        "custom": config.python is not None,
    }


def _per_class(record: EvaluationRecord | None) -> dict[str, Any]:
    if record is None:
        return {}
    per_class: dict[str, Any] = record.report.primary_value.details.get("per_class", {})
    return per_class


def _version(version: PromptVersion, decision: Any) -> dict[str, Any]:
    return {
        "version": version.number,
        "parent": version.parent_number,
        "source": version.source,
        "change_summary": version.change_summary,
        "status": decision.status if decision else None,
        "primary_delta": decision.primary_delta if decision else None,
        "decision_reason": decision.reason if decision else None,
    }


def _run(run: RunRecord, runs: RunRepository) -> dict[str, Any]:
    return {
        "id": run.id,
        "status": run.status,
        "stop_reason": run.stop_reason,
        "turns": len([t for t in runs.turns(run.id) if t.status == "done"]),
        "model_calls": run.model_calls,
        "cost_usd": run.cost_usd,
    }


def _error_analysis(
    project: Project, evals: EvaluationRepository, selected: PromptVersion
) -> dict[str, Any] | None:
    """Error counts from the latest validation outcomes; counts only, no example content."""
    record = _latest(evals, selected, Partition.VALIDATION)
    if record is None:
        return None
    outcomes = evals.outcomes(record.id)
    facts = build_facts(outcomes, [], project.config.output.type, record.report.primary)
    return {
        "evaluation": record.id,
        "examples": facts.total,
        "failed": facts.failed,
        "codes": dict(facts.codes),
        "confusions": [{"expected": e, "got": g, "count": n} for e, g, n in facts.confusions],
    }


def _usage(evals: EvaluationRepository) -> dict[str, Any]:
    records = evals.find(limit=100000)
    calls = sum(r.usage.calls - r.usage.cached_calls for r in records)
    costs = [r.usage.cost_usd for r in records if r.usage.cost_usd is not None]
    tokens = [r.usage.total_tokens for r in records if r.usage.total_tokens is not None]
    return {
        "evaluations": len(records),
        "model_calls": calls,
        "total_tokens": sum(tokens) if tokens else None,
        "cost_usd": sum(costs) if costs else None,
    }
