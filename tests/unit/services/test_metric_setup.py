from pathlib import Path
from typing import Any

import pytest

from strop.config import StropConfig, dump_config
from strop.errors import ConfigError
from strop.llm import MockLLMClient
from strop.services.metrics import build_suite, judge_model
from strop.services.project import Project, open_project
from strop.storage import ProjectPaths


def project_with(tmp_path: Path, config_data: dict[str, Any]) -> Project:
    (tmp_path / "data.csv").write_text("message,category\nhi,refund\n", encoding="utf-8")
    dump_config(StropConfig.model_validate(config_data), ProjectPaths(tmp_path).config_file)
    return open_project(tmp_path)


def test_reference_metrics_need_an_expected_column(
    tmp_path: Path, config_data: dict[str, Any]
) -> None:
    del config_data["dataset"]["columns"]["category"]
    project = project_with(tmp_path, config_data)
    with pytest.raises(ConfigError, match="'macro_f1' compares outputs with expected"):
        build_suite(project)


def test_reference_free_metrics_work_without_an_expected_column(
    tmp_path: Path, config_data: dict[str, Any]
) -> None:
    del config_data["dataset"]["columns"]["category"]
    config_data["output"] = {"type": "text"}
    config_data["metrics"] = {
        "primary": "rule:short",
        "rules": {"short": {"type": "max_words", "value": 20}},
    }
    suite = build_suite(project_with(tmp_path, config_data))
    assert suite.primary == "rule:short"
    assert "valid_output_rate" in suite.names


def test_judge_metrics_need_a_judge_model(tmp_path: Path, config_data: dict[str, Any]) -> None:
    config_data["output"] = {"type": "text"}
    config_data["metrics"] = {
        "primary": "judge:helpful",
        "judges": {"helpful": {"criteria": "Is it helpful?"}},
    }
    project = project_with(tmp_path, config_data)
    with pytest.raises(ConfigError, match="--judge-model"):
        judge_model(project)
    with pytest.raises(ConfigError, match="needs a judge model"):
        build_suite(project)


def test_configured_judges_and_python_metric_are_added_to_the_suite(
    tmp_path: Path, config_data: dict[str, Any]
) -> None:
    config_data["models"]["judge"] = {"model": "provider/judge"}
    config_data["metrics"] = {
        "primary": "judge:helpful",
        "secondary": ["accuracy"],
        "judges": {"helpful": {"criteria": "Is it helpful?"}},
        "python": {"file": "metric.py", "direction": "minimize"},
    }
    project = project_with(tmp_path, config_data)
    spec = judge_model(project)
    assert spec is not None
    suite = build_suite(project, (MockLLMClient("{}"), spec))
    assert {"judge:helpful", "custom", "accuracy", "valid_output_rate"} <= set(suite.names)
    assert suite.direction("custom").value == "minimize"
    assert suite.metric("judge:helpful").subjective is True
