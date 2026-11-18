"""An extraction task: JSONL data, a JSON output contract with a schema, and field-level metrics."""

import json
from pathlib import Path
from typing import Any

import pytest

from strop.llm import LLMRequest, MockLLMClient
from tests.helpers import CliRunner
from tests.integration.conftest import data_block

CITIES = ["Paris", "Berlin", "Madrid", "Rome", "Lisbon", "Vienna"]
SCHEMA = {
    "type": "object",
    "required": ["city", "date"],
    "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
}
RULE = "Write dates as ISO 8601."


def dataset(count: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(count):
        city, (day, month) = CITIES[i % len(CITIES)], (1 + i % 28, 1 + i % 12)
        rows.append(
            {
                "text": f"Meet in {city} on {day:02d}/{month:02d}/2026",
                "answer": {"city": city, "date": f"2026-{month:02d}-{day:02d}"},
            }
        )
    return rows


def target(request: LLMRequest) -> str:
    """Finds the city and date, but only writes ISO dates when the prompt says so."""
    text = request.messages[-1].content
    city = next(c for c in CITIES if c in text)
    day, month, year = text.rsplit(" ", 1)[1].split("/")
    iso = RULE in request.messages[0].content
    return json.dumps({"city": city, "date": f"{year}-{month}-{day}" if iso else f"{day}/{month}"})


def optimizer(request: LLMRequest) -> str:
    data = data_block(request)
    if request.messages[0].content.startswith("You analyze"):
        ids = [e["id"] for e in data["examples"]][:2]
        return json.dumps(
            {
                "patterns": [
                    {"name": "date_format", "description": "wrong date format", "example_ids": ids}
                ]
            }
        )
    current = data["prompt"][0]["content"]
    return json.dumps(
        {
            "messages": [{"index": 1, "content": f"{current}\n{RULE}"}],
            "change_summary": "Require ISO 8601 dates",
            "target_failure_patterns": ["date_format"],
            "evidence_example_ids": [data["examples"][0]["id"]],
        }
    )


@pytest.fixture
def extraction_project(run_cli: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "data.jsonl").write_text(
        "\n".join(json.dumps(row) for row in dataset()) + "\n", encoding="utf-8"
    )
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    code, _, err = run_cli(
        "init", "--dir", str(tmp_path), "--name", "extract", "--task", "Extract the city and date.",
        "--data", "data.jsonl", "--input", "text", "--expected", "answer", "--output-type", "json",
        "--output-schema", "schema.json", "--metric", "field_f1",
        "--secondary-metric", "field_accuracy", "--constraint", "schema_valid>=0.9",
        "--target-model", "openai/model-a", "--optimizer-model", "anthropic/model-b",
    )  # fmt: skip
    assert code == 0, err
    assert run_cli("data", "split", "--dir", str(tmp_path))[0] == 0

    client = MockLLMClient(lambda r: optimizer(r) if r.model.startswith("anthropic") else target(r))
    for module in ("evaluator", "context", "evaluate", "analysis"):
        monkeypatch.setattr(
            f"strop.services.{module}.build_client", lambda *_: client, raising=False
        )
    return tmp_path


def test_baseline_scores_each_field(run_cli: CliRunner, extraction_project: Path) -> None:
    code, out, _ = run_cli("baseline", "--dir", str(extraction_project))
    assert code == 0
    assert "field_f1 (primary)" in out
    assert "0.5000" in out
    assert "schema_valid" in out
    assert "wrong_value" in out or "valid but wrong" in out


def test_the_loop_fixes_the_date_format(run_cli: CliRunner, extraction_project: Path) -> None:
    code, out, _ = run_cli("optimize", "--dir", str(extraction_project), "--candidates", "1")
    assert code == 0
    assert "Accepted v2 as the new best prompt." in out
    assert "field_f1 1.0000" in out
    assert f"+{RULE}" in out


def test_schema_violations_are_reported_and_counted(
    run_cli: CliRunner, extraction_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = MockLLMClient('{"city": "Paris"}')
    for module in ("evaluator", "evaluate"):
        monkeypatch.setattr(
            f"strop.services.{module}.build_client", lambda *_: broken, raising=False
        )
    _, out, _ = run_cli("baseline", "--dir", str(extraction_project))
    assert "schema_violation 12" in out
    assert "valid_output_rate" in out


def test_init_rejects_a_bad_schema_file(run_cli: CliRunner, tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text(json.dumps(dataset(1)[0]) + "\n", encoding="utf-8")
    (tmp_path / "schema.json").write_text('{"type": "nonsense"}', encoding="utf-8")
    code, _, err = run_cli(
        "init", "--dir", str(tmp_path), "--name", "x", "--task", "t", "--data", "data.jsonl",
        "--input", "text", "--expected", "answer", "--output-type", "json",
        "--output-schema", "schema.json", "--metric", "field_f1", "--target-model", "a/b",
    )  # fmt: skip
    assert code == 2
    assert "not a valid JSON schema" in err
