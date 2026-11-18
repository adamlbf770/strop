import json
from collections.abc import Callable
from pathlib import Path

import pytest

from strop.llm import LLMRequest, MockLLMClient
from tests.helpers import CliRunner, write_dataset

Responder = Callable[[LLMRequest], str]


def message_number(request: LLMRequest) -> int:
    return int(request.messages[-1].content.split()[-1])


def perfect(request: LLMRequest) -> str:
    return "refund" if message_number(request) % 2 else "billing"


def flawed(request: LLMRequest) -> str:
    """Wrong for every fifth message."""
    answer = perfect(request)
    if message_number(request) % 5 == 0:
        return "billing" if answer == "refund" else "refund"
    return answer


@pytest.fixture
def install_model(monkeypatch: pytest.MonkeyPatch) -> Callable[[Responder], MockLLMClient]:
    def install(responder: Responder) -> MockLLMClient:
        client = MockLLMClient(responder)
        monkeypatch.setattr("strop.services.evaluator.build_client", lambda *_: client)
        monkeypatch.setattr("strop.services.evaluate.build_client", lambda *_: client)
        return client

    return install


@pytest.fixture
def split_project(run_cli: CliRunner, project: Path) -> Path:
    assert run_cli("data", "split", "--dir", str(project))[0] == 0
    return project


def test_baseline_needs_a_split(
    run_cli: CliRunner, project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(perfect)
    code, _, err = run_cli("baseline", "--dir", str(project))
    assert code == 2
    assert "strop data split" in err


def test_baseline_reports_metrics_and_registers_v1(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    client = install_model(perfect)
    code, out, _ = run_cli("baseline", "--dir", str(split_project))
    assert code == 0
    assert "Evaluation 1: v1 on validation (20 examples, baseline)" in out
    assert "macro_f1 (primary)" in out
    assert "1.0000" in out
    assert "No output problems." in out
    assert "Per class" in out
    assert "Calls: 20" in out
    assert "Cost: unavailable" in out
    assert len(client.calls) == 20
    _, history, _ = run_cli("history", "--dir", str(split_project))
    assert history.startswith("* v1")


def test_baseline_lists_failing_examples(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(flawed)
    _, out, _ = run_cli("baseline", "--dir", str(split_project), "--show-errors", "2")
    assert "valid but wrong" in out
    assert "Examples that failed:" in out
    assert "expected 'refund', got 'billing'" in out or "expected 'billing', got 'refund'" in out


def test_baseline_flags_invalid_outputs(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(lambda request: "maybe")
    _, out, _ = run_cli("baseline", "--dir", str(split_project))
    assert "invalid_format 20" in out
    assert "valid_output_rate" in out


def test_baseline_json_output(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(perfect)
    _, out, _ = run_cli("baseline", "--dir", str(split_project), "--json")
    payload = json.loads(out)
    assert payload["version"] == "v1"
    assert payload["baseline"]["metrics"]["macro_f1"]["value"] == 1.0
    assert payload["noise"] is None
    assert payload["train"] is None


def test_repeats_measure_noise(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    client = install_model(perfect)
    _, out, _ = run_cli("baseline", "--dir", str(split_project), "--repeats", "2")
    assert "Run-to-run noise over 2 runs" in out
    assert len(client.calls) == 40


def test_with_train_evaluates_the_train_partition_too(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(perfect)
    _, out, _ = run_cli("baseline", "--dir", str(split_project), "--with-train")
    assert "v1 on train (60 examples" in out


def test_sampling_limits_the_rows(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    client = install_model(perfect)
    _, out, _ = run_cli("baseline", "--dir", str(split_project), "--sample", "5")
    assert "(5 examples, baseline)" in out
    assert len(client.calls) == 5


def test_eval_scores_a_version_without_changing_anything(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(perfect)
    run_cli("baseline", "--dir", str(split_project))
    code, out, _ = run_cli("eval", "--dir", str(split_project), "--partition", "train")
    assert code == 0
    assert "v1 on train (60 examples, eval)" in out
    _, history, _ = run_cli("history", "--dir", str(split_project))
    assert len(history.splitlines()) == 1


def test_eval_refuses_the_test_partition(run_cli: CliRunner, split_project: Path) -> None:
    code, _, err = run_cli("eval", "--dir", str(split_project), "--partition", "test")
    assert code == 2
    assert "test" in err


def test_eval_json_and_named_version(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(perfect)
    run_cli("baseline", "--dir", str(split_project))
    _, out, _ = run_cli("eval", "v1", "--dir", str(split_project), "--json")
    payload = json.loads(out)
    assert (payload["version"], payload["partition"], payload["kind"]) == (
        "v1",
        "validation",
        "eval",
    )


def test_edited_prompt_file_triggers_a_warning(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(perfect)
    run_cli("baseline", "--dir", str(split_project))
    prompt = split_project / "prompt.yaml"
    prompt.write_text(prompt.read_text(encoding="utf-8").replace("Classify", "Sort"), "utf-8")
    _, _, err = run_cli("eval", "--dir", str(split_project))
    assert "differs from every stored prompt version" in err


def test_changed_dataset_is_detected(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(perfect)
    write_dataset(split_project, rows=80)
    code, _, err = run_cli("baseline", "--dir", str(split_project))
    assert code == 2
    assert "changed since the split" in err


def test_status_summarizes_the_project(
    run_cli: CliRunner, split_project: Path, install_model: Callable[[Responder], MockLLMClient]
) -> None:
    install_model(perfect)
    run_cli("baseline", "--dir", str(split_project))
    _, out, _ = run_cli("status", "--dir", str(split_project))
    assert "Project: demo" in out
    assert "Split 1 (seed 42): train 60, validation 20, test 20" in out
    assert "Prompt: v1 is the best of 1 version(s)" in out
    assert "#1 v1 validation baseline: macro_f1=1.0000 (20 examples)" in out
    _, raw, _ = run_cli("status", "--dir", str(split_project), "--json")
    assert json.loads(raw)["best_version"] == 1
