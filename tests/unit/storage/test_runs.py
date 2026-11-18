import pytest

from strop.errors import StorageError
from strop.prompts import PromptMessage, PromptTemplate
from strop.storage import connect
from strop.storage.migrator import migrate
from strop.storage.prompts import PromptRepository
from strop.storage.runs import NewCandidate, RunRepository


def make_repos() -> RunRepository:
    conn = connect(":memory:")
    migrate(conn)
    prompts = PromptRepository(conn)
    for text in ("a", "b", "c"):
        template = PromptTemplate((PromptMessage("user", text),), frozenset({"user"}))
        prompts.add(template, parent_number=None)
    return RunRepository(conn)


def test_run_lifecycle() -> None:
    repo = make_repos()
    run = repo.create_run(1, {"max_turns": 3})
    assert (run.status, run.start_prompt, run.settings) == ("running", 1, {"max_turns": 3})
    assert repo.resumable_run() == run
    repo.update_run(
        run.id, status="completed", stop_reason="patience", model_calls=12, finished=True
    )
    done = repo.get_run(run.id)
    assert (done.status, done.stop_reason, done.model_calls) == ("completed", "patience", 12)
    assert done.finished_at is not None
    assert repo.resumable_run() is None
    assert repo.latest_run() == done


def test_paused_and_interrupted_runs_are_resumable() -> None:
    repo = make_repos()
    run = repo.create_run(1, {})
    for status in ("paused", "interrupted"):
        repo.update_run(run.id, status=status)
        resumable = repo.resumable_run()
        assert resumable is not None
        assert resumable.status == status


def test_unknown_run_is_reported() -> None:
    with pytest.raises(StorageError, match="run 9"):
        make_repos().get_run(9)


def test_turn_updates_only_the_given_fields() -> None:
    repo = make_repos()
    run = repo.create_run(1, {})
    turn = repo.create_turn(run.id, 1, best_before=1)
    assert (turn.status, turn.best_after) == ("running", None)
    repo.update_turn(run.id, 1, analysis={"codes": {"wrong_label": 3}})
    repo.update_turn(run.id, 1, status="done", best_after=2)
    updated = repo.get_turn(run.id, 1)
    assert (updated.status, updated.best_after) == ("done", 2)
    assert updated.analysis == {"codes": {"wrong_label": 3}}
    assert [t.number for t in repo.turns(run.id)] == [1]


def test_candidate_round_trip_and_status_changes() -> None:
    repo = make_repos()
    run = repo.create_run(1, {})
    saved = repo.add_candidate(
        NewCandidate(
            run.id,
            1,
            1,
            "evaluated",
            version_number=2,
            change_summary="clarify",
            patterns=["p"],
            evidence=["r1"],
            tradeoffs=["t"],
            assumptions=["a"],
            passes=True,
            primary_delta=0.05,
            wins=3,
            losses=1,
            ties=6,
        )
    )
    assert saved.passes is True
    assert (saved.wins, saved.losses, saved.ties) == (3, 1, 6)
    assert saved.patterns == ["p"]
    repo.set_candidate_status(saved.id, "pending", "waiting")
    assert [c.id for c in repo.pending()] == [saved.id]
    found = repo.for_version(2)
    assert found is not None
    assert found.status == "pending"
    assert repo.for_version(3) is None


def test_invalid_candidates_have_no_version_or_evaluation() -> None:
    repo = make_repos()
    run = repo.create_run(1, {})
    invalid = repo.add_candidate(NewCandidate(run.id, 1, 1, "invalid", reason="bad JSON"))
    assert (invalid.version_number, invalid.passes, invalid.evaluation_id) == (None, None, None)
    assert invalid.reason == "bad JSON"


def test_candidate_queries_and_recent_history() -> None:
    repo = make_repos()
    run = repo.create_run(1, {})
    repo.add_candidate(NewCandidate(run.id, 1, 1, "evaluated", version_number=2))
    repo.add_candidate(NewCandidate(run.id, 1, 1, "rejected", version_number=3, reason="no gain"))
    repo.add_candidate(NewCandidate(run.id, 2, 1, "invalid"))
    assert len(repo.candidates(run.id)) == 3
    assert len(repo.candidates(run.id, turn=1)) == 2
    recent = repo.recent(5)
    assert [c.status for c in recent] == ["invalid", "rejected"]
    assert len(repo.recent(1)) == 1
