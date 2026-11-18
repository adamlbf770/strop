"""Exercises the real LiteLLM client stack without a network, using LiteLLM's mock responses."""

from pathlib import Path

from tests.helpers import CliRunner


def configure_mock_response(project: Path, answer: str) -> None:
    path = project / "strop.yaml"
    text = path.read_text(encoding="utf-8")
    old = (
        "  target:\n    model: openai/model-a\n    temperature: 0.0\n"
        "    timeout_s: 60.0\n    params: {}\n"
    )
    new = old.replace("params: {}", f"params:\n      mock_response: {answer}")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_real_client_records_usage_and_serves_repeats_from_the_cache(
    run_cli: CliRunner, project: Path
) -> None:
    configure_mock_response(project, "refund")
    assert run_cli("data", "split", "--dir", str(project))[0] == 0
    code, out, _ = run_cli("baseline", "--dir", str(project))
    assert code == 0
    assert "Calls: 20 (0 cached)" in out
    assert "Tokens: unavailable" not in out
    assert "No output problems." not in out
    code, again, _ = run_cli("baseline", "--dir", str(project))
    assert code == 0
    assert "Calls: 20 (20 cached)" in again


def test_unknown_providers_fail_without_crashing_the_run(run_cli: CliRunner, project: Path) -> None:
    path = project / "strop.yaml"
    path.write_text(
        path.read_text("utf-8").replace("openai/model-a", "no-such-provider/x"), "utf-8"
    )
    assert run_cli("data", "split", "--dir", str(project))[0] == 0
    code, out, _ = run_cli("baseline", "--dir", str(project))
    assert code == 0
    assert "provider_error 20" in out
