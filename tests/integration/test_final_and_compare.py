import json
from pathlib import Path

from tests.helpers import CliRunner
from tests.integration.conftest import Install


def improved(run_cli: CliRunner, project: Path) -> None:
    code, _, _ = run_cli("optimize", "--dir", str(project), "--candidates", "2")
    assert code == 0


def test_test_requires_explicit_confirmation(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    client = install_models()
    code, _, err = run_cli("test", "--dir", str(ready))
    assert code == 2
    assert "--confirm" in err
    assert "optimistic estimate" in err
    assert client.calls == []


def test_confirmed_test_reports_the_held_out_result(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    improved(run_cli, ready)
    code, out, _ = run_cli("test", "--dir", str(ready), "--confirm", "--with-baseline")
    assert code == 0
    assert "on test (20 examples, test)" in out
    assert "Held-out test result" in out
    assert "v1 (initial)" in out
    assert "validation" in out
    assert "95% Wilson interval" in out
    assert "Do not change the prompt based on this result" in out


def test_the_test_partition_can_only_be_used_once_without_a_flag(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    assert run_cli("test", "--dir", str(ready), "--confirm")[0] == 0
    code, _, err = run_cli("test", "--dir", str(ready), "--confirm")
    assert code == 2
    assert "already used" in err
    assert "--allow-rerun" in err
    assert run_cli("test", "--dir", str(ready), "--confirm", "--allow-rerun")[0] == 0


def test_test_json_output(run_cli: CliRunner, ready: Path, install_models: Install) -> None:
    install_models()
    _, out, _ = run_cli("test", "--dir", str(ready), "--confirm", "--json")
    payload = json.loads(out)
    assert payload["test"]["partition"] == "test"
    assert payload["test"]["kind"] == "test"


def test_a_project_without_a_test_partition_explains_how_to_make_one(
    run_cli: CliRunner, project: Path, install_models: Install
) -> None:
    install_models()
    run_cli("data", "split", "--dir", str(project), "--train", "0.8", "--test", "0")
    code, _, err = run_cli("test", "--dir", str(project), "--confirm")
    assert code == 2
    assert "no test partition" in err


def test_compare_runs_the_same_rows_through_each_model(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    client = install_models()
    code, out, _ = run_cli(
        "compare", "--dir", str(ready), "--models", "openai/model-a,openai/model-z"
    )
    assert code == 0
    assert "same rows for every model" in out
    assert "openai/model-a" in out
    assert "openai/model-z" in out
    assert {c.model for c in client.calls} == {"openai/model-a", "openai/model-z"}
    sent = {
        m: [c.messages[-1].content for c in client.calls if c.model == m]
        for m in ("openai/model-a", "openai/model-z")
    }
    assert sorted(sent["openai/model-a"]) == sorted(sent["openai/model-z"])


def test_compare_json_and_validation(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    _, out, _ = run_cli(
        "compare", "--dir", str(ready), "--models", "openai/model-a, openai/model-b", "--json"
    )
    payload = json.loads(out)
    assert [m["model"] for m in payload["models"]] == ["openai/model-a", "openai/model-b"]
    assert payload["partition"] == "validation"
    code, _, err = run_cli("compare", "--dir", str(ready), "--models", "only-one")
    assert code == 2
    assert "at least two" in err
    code, _, err = run_cli("compare", "--dir", str(ready), "--models", "a/x,a/x")
    assert code == 2
    assert "more than once" in err
