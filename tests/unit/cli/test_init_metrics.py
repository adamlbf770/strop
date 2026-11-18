from pathlib import Path

from strop.config import load_config
from tests.helpers import CliRunner


def test_init_accepts_rules_and_a_metric_file(
    run_cli: CliRunner, init_args: list[str], project_dir: Path
) -> None:
    (project_dir / "metric.py").write_text("def score(output, row):\n    return 1.0\n", "utf-8")
    code, _, _ = run_cli(
        *init_args,
        "--rule", "short=max_words:50",
        "--rule", "needs=contains_all:refund,days",
        "--metric-file", "metric.py",
        "--secondary-metric", "custom",
    )  # fmt: skip
    assert code == 0
    metrics = load_config(project_dir / "strop.yaml").metrics
    assert set(metrics.rules) == {"short", "needs"}
    assert metrics.python is not None
    assert metrics.python.file == Path("metric.py")


def test_init_loads_rubrics_and_judge_model(
    run_cli: CliRunner, init_args: list[str], project_dir: Path
) -> None:
    (project_dir / "rubric.yaml").write_text("helpful:\n  criteria: Is it helpful?\n", "utf-8")
    code, out, _ = run_cli(
        *init_args,
        "--rubric-file", "rubric.yaml",
        "--judge-model", "google/model-j",
        "--secondary-metric", "judge:helpful",
    )  # fmt: skip
    assert code == 0
    config = load_config(project_dir / "strop.yaml")
    assert "helpful" in config.metrics.judges
    assert config.models.judge is not None
    assert "anthropic, google, openai" in out


def test_init_rejects_unknown_metrics_before_writing_files(
    run_cli: CliRunner, init_args: list[str], project_dir: Path
) -> None:
    init_args[init_args.index("macro_f1")] = "macro_f2"
    code, _, err = run_cli(*init_args)
    assert code == 2
    assert "unknown metric 'macro_f2'" in err
    assert not (project_dir / "strop.yaml").exists()


def test_init_checks_that_referenced_custom_and_judge_metrics_exist(
    run_cli: CliRunner, init_args: list[str]
) -> None:
    assert run_cli(*init_args, "--secondary-metric", "custom")[0] == 2
    code, _, err = run_cli(*init_args, "--secondary-metric", "judge:helpful")
    assert code == 2
    assert "--rubric-file" in err


def test_init_reports_a_missing_metric_file(run_cli: CliRunner, init_args: list[str]) -> None:
    code, _, err = run_cli(*init_args, "--metric-file", "absent.py")
    assert code == 2
    assert "metric file not found" in err


def test_init_rejects_malformed_rules(run_cli: CliRunner, init_args: list[str]) -> None:
    code, _, err = run_cli(*init_args, "--rule", "short")
    assert code == 2
    assert "name=type:value" in err


def test_verbose_and_quiet_flags_are_accepted(run_cli: CliRunner, project: Path) -> None:
    assert run_cli("--verbose", "status", "--dir", str(project))[0] == 0
    assert run_cli("--quiet", "status", "--dir", str(project))[0] == 0
