import json
import sqlite3
from pathlib import Path

from strop.data.models import Partition
from strop.llm import LLMRequest
from strop.services.partitions import load_partitions
from strop.services.project import open_project
from strop.storage.datasets import DatasetRepository
from tests.helpers import CliRunner
from tests.integration.conftest import Install, helper_model


def optimize(run_cli: CliRunner, project: Path, *args: str) -> tuple[int, str, str]:
    return run_cli("optimize", "--dir", str(project), "--candidates", "2", *args)


def history(run_cli: CliRunner, project: Path) -> list[str]:
    return run_cli("history", "--dir", str(project))[1].splitlines()


def test_automatic_run_improves_the_prompt_and_shows_the_diff(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    code, out, _ = optimize(run_cli, ready, "--max-turns", "3")
    assert code == 0
    assert "Run 1: no failures left on the training examples" in out
    assert "Accepted v2 as the new best prompt." in out
    assert "v1 macro_f1 0." in out
    assert "v2 macro_f1 1.0000" in out
    assert "Prompt changes, v1 -> v2:" in out
    assert "+REFUND-RULE variant 0" in out
    lines = history(run_cli, ready)
    assert lines[0].split()[:2] == ["v1", "user"]
    assert lines[1].startswith("* v2")
    assert "accepted" in lines[1]
    assert "rejected" in lines[2]


def test_every_candidate_is_kept_including_rejected_ones(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    optimize(run_cli, ready)
    _, out, _ = run_cli("history", "--dir", str(ready), "--json")
    versions = json.loads(out)["versions"]
    assert [v["version"] for v in versions] == [1, 2, 3]
    assert versions[2]["status"] == "rejected"
    assert "v2 was chosen this turn" in versions[2]["decision_reason"]


def test_manual_approval_pauses_then_resumes(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    code, out, err = optimize(run_cli, ready, "--approval", "manual")
    assert code == 10
    assert "paused for approval" in out
    assert "v2 is waiting for 'strop approve' or 'strop reject'" in out
    assert "strop approve v2" in err
    assert history(run_cli, ready)[0].startswith("* v1")
    assert "pending" in history(run_cli, ready)[1]

    blocked, _, blocked_err = optimize(run_cli, ready)
    assert blocked == 2
    assert "waiting for a decision" in blocked_err

    assert "v2 is now the best prompt." in run_cli("approve", "v2", "--dir", str(ready))[1]
    code, out, _ = run_cli("optimize", "--dir", str(ready), "--resume")
    assert code == 0
    assert "no failures left" in out
    assert history(run_cli, ready)[1].startswith("* v2")


def test_rejecting_a_candidate_keeps_the_best_prompt(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    optimize(run_cli, ready, "--approval", "manual")
    _, out, _ = run_cli("reject", "v2", "--reason", "too long", "--dir", str(ready))
    assert "v2 was rejected" in out
    assert history(run_cli, ready)[0].startswith("* v1")
    code, out, _ = run_cli("optimize", "--dir", str(ready), "--resume", "--candidates", "2")
    assert code == 0
    assert "identical to the existing version v3" in out
    assert "Accepted v4 as the new best prompt." in out
    _, raw, _ = run_cli("history", "--dir", str(ready), "--json")
    versions = {v["version"]: v for v in json.loads(raw)["versions"]}
    assert versions[2]["status"] == "rejected"
    assert "rejected by the user: too long" in versions[2]["decision_reason"]


def test_unusable_proposals_never_change_the_best_prompt(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    def frozen_edit(request: LLMRequest) -> str:
        if request.messages[0].content.startswith("You analyze"):
            return helper_model(request)
        return json.dumps(
            {
                "messages": [{"index": 2, "content": "Changed {{message}}"}],
                "change_summary": "edit the input template",
                "evidence_example_ids": ["row_0001"],
            }
        )

    install_models(frozen_edit)
    code, out, _ = optimize(run_cli, ready, "--patience", "1")
    assert code == 0
    assert "not editable" in out
    assert "No improvement was accepted; the best prompt is unchanged." in out
    assert history(run_cli, ready)[0].startswith("* v1")
    assert len(history(run_cli, ready)) == 1


def test_target_reached_stops_before_any_revision(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    client = install_models()
    code, out, _ = optimize(run_cli, ready, "--target", "0.5")
    assert code == 0
    assert "reached the target metric" in out
    assert not [c for c in client.calls if c.model.startswith("anthropic")]


def test_budget_stops_the_run_and_it_can_be_resumed(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    code, out, err = optimize(run_cli, ready, "--max-calls", "30")
    assert code == 11
    assert "stopped by a budget" in out
    assert "strop optimize --resume" in err
    code, out, _ = run_cli("optimize", "--dir", str(ready), "--resume", "--candidates", "2")
    assert code == 0
    assert "v2 macro_f1 1.0000" in out


def test_ctrl_c_saves_progress_and_resume_continues(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    def interrupt(request: LLMRequest) -> str:
        raise KeyboardInterrupt

    install_models(interrupt)
    code, _, err = optimize(run_cli, ready)
    assert code == 12
    assert "strop optimize --resume" in err
    install_models()
    code, out, _ = run_cli("optimize", "--dir", str(ready), "--resume", "--candidates", "2")
    assert code == 0
    assert "Accepted v2" in out


def test_resume_without_a_run_is_an_error(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    code, _, err = run_cli("optimize", "--dir", str(ready), "--resume")
    assert code == 2
    assert "no unfinished run" in err


def test_instructions_reach_the_optimizer_outside_the_data_block(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    client = install_models()
    feedback = ready / "feedback.txt"
    feedback.write_text("Prefer short rules.", encoding="utf-8")
    optimize(run_cli, ready, "--instructions", "Be strict", "--feedback-file", str(feedback))
    asked = [c for c in client.calls if c.messages[0].content.startswith("You improve")]
    assert asked
    trailer = asked[0].messages[-1].content.split("</data>")[1]
    assert "Be strict" in trailer
    assert "Prefer short rules." in trailer


def test_json_output_describes_the_run(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    code, out, _ = optimize(run_cli, ready, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["stop_reason"] == "no_failures"
    assert (payload["initial_version"], payload["best_version"]) == (1, 2)
    assert payload["best_value"] == 1.0
    assert payload["start_value"] < 1.0
    assert payload["diff"]["changed"] is True
    assert {c["status"] for c in payload["candidates"]} == {"accepted", "rejected"}


def test_test_rows_never_reach_the_models_or_the_database(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    client = install_models()
    optimize(run_cli, ready)
    project = open_project(ready)
    with project.database() as conn:
        held_out = load_partitions(project, DatasetRepository(conn)).rows
        stored = {r[0] for r in conn.execute("SELECT DISTINCT partition FROM evaluations")}
    test_texts = {row.inputs["message"] for row in held_out[Partition.TEST]}
    assert test_texts
    assert "test" not in stored
    for call in client.calls:
        sent = "\n".join(m.content for m in call.messages)
        for text in test_texts:
            assert f"message: {text}\n" not in sent + "\n"
            assert f'"message": "{text}"' not in sent


def test_analyze_lists_facts_and_labels_patterns_as_hypotheses(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    code, out, _ = run_cli("analyze", "--dir", str(ready), "--patterns")
    assert code == 0
    assert "Analysis of v1 on train" in out
    assert "wrong_label" in out
    assert "refund -> billing" in out
    assert "Hypotheses (proposed by an LLM, not verified):" in out
    assert "refund_missed" in out
    _, plain, _ = run_cli("analyze", "--dir", str(ready))
    assert "Hypotheses" not in plain


def test_database_is_left_consistent_after_a_run(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    optimize(run_cli, ready)
    conn = sqlite3.connect(ready / ".strop" / "strop.db")
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("SELECT status FROM runs").fetchall() == [("completed",)]
    assert conn.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0] == 3
