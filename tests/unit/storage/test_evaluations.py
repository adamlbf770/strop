import pytest

from strop.data.models import Partition
from strop.errors import StorageError
from strop.evaluation.models import ExampleOutcome, NewEvaluation, summarize_usage
from strop.metrics.suite import MetricReport
from strop.metrics.types import MetricValue
from strop.prompts import PromptMessage, PromptTemplate
from strop.storage import connect
from strop.storage.datasets import DatasetRepository
from strop.storage.evaluations import EvaluationRepository
from strop.storage.migrator import migrate
from strop.storage.prompts import PromptRepository


def make_repo() -> tuple[EvaluationRepository, int]:
    conn = connect(":memory:")
    migrate(conn)
    PromptRepository(conn).add(
        PromptTemplate((PromptMessage("user", "x"),), frozenset({"user"})), parent_number=None
    )
    datasets = DatasetRepository(conn)
    dataset_id = datasets.save_dataset(
        content_hash="h", source_path="d.csv", row_count=1, skipped_rows=0, skip_invalid=False
    )
    split_id = datasets.create_split(
        dataset_id=dataset_id,
        seed=1,
        strategy="random",
        config_json="{}",
        assignments={"r1": Partition.VALIDATION},
    )
    return EvaluationRepository(conn), split_id


def outcome(row_id: str, **kwargs: object) -> ExampleOutcome:
    return ExampleOutcome(row_id, "a", "a", "a", **kwargs)  # type: ignore[arg-type]


def report() -> MetricReport:
    value = MetricValue("accuracy", 0.5, 2, details={"per_class": {"a": {"f1": 0.5}}})
    return MetricReport("accuracy", {"accuracy": value})


def test_evaluation_round_trip() -> None:
    repo, split_id = make_repo()
    outcomes = [
        outcome(
            "r1",
            parsed={"k": 1},
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            cost_usd=0.001,
            latency_s=0.5,
            scores={"accuracy": 1.0, "other": None},
        ),
        ExampleOutcome("r2", "a", "", None, error="provider_error", detail="down"),
    ]
    saved = repo.save(
        NewEvaluation(1, split_id, Partition.VALIDATION, "baseline", report(), outcomes, {"n": 1})
    )
    loaded = repo.get(saved.id)
    assert loaded.partition is Partition.VALIDATION
    assert loaded.kind == "baseline"
    assert loaded.sample_count == 2
    assert loaded.error_count == 1
    assert loaded.details == {"n": 1}
    assert loaded.report.primary_value.value == 0.5
    assert loaded.report.primary_value.details["per_class"]["a"]["f1"] == 0.5
    assert loaded.usage.total_tokens == 12
    assert loaded.usage.cost_usd == 0.001
    stored = repo.outcomes(saved.id)
    assert [o.row_id for o in stored] == ["r1", "r2"]
    assert stored[0].parsed == {"k": 1}
    assert stored[0].scores == {"accuracy": 1.0, "other": None}
    assert stored[1].error == "provider_error"
    assert stored[1].detail == "down"
    assert stored[1].parsed is None


def test_find_filters_and_orders_newest_first() -> None:
    repo, split_id = make_repo()
    first = repo.save(NewEvaluation(1, split_id, Partition.VALIDATION, "baseline", report(), []))
    second = repo.save(NewEvaluation(1, split_id, Partition.VALIDATION, "eval", report(), []))
    repo.save(NewEvaluation(1, split_id, Partition.TRAIN, "eval", report(), []))
    found = repo.find(prompt_number=1, partition=Partition.VALIDATION)
    assert [e.id for e in found] == [second.id, first.id]
    assert [e.id for e in repo.find(kind="baseline")] == [first.id]
    assert len(repo.find()) == 3


def test_missing_evaluation_is_reported() -> None:
    repo, _ = make_repo()
    with pytest.raises(StorageError, match="does not exist"):
        repo.get(99)


def test_usage_summary_ignores_cached_calls_and_missing_data() -> None:
    outcomes = [
        outcome("a", total_tokens=10, cost_usd=0.01, latency_s=1.0),
        outcome("b", total_tokens=20, cost_usd=0.02, latency_s=3.0, cached=True),
        outcome("c", latency_s=2.0),
    ]
    usage = summarize_usage(outcomes)
    assert (usage.calls, usage.cached_calls) == (3, 1)
    assert usage.total_tokens == 10
    assert usage.cost_usd == 0.01
    assert usage.mean_latency_s == pytest.approx(1.5)
    assert usage.cost_complete is False
    empty = summarize_usage([])
    assert empty.total_tokens is None
    assert empty.cost_usd is None
