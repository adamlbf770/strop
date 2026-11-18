from pathlib import Path

from strop.services.project import open_project
from tests.helpers import CliRunner, write_dataset
from tests.integration.conftest import Install


def test_optimizer_and_analyzer_models_can_be_chosen_per_run(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    client = install_models()
    code, _, _ = run_cli(
        "optimize", "--dir", str(ready), "--candidates", "2",
        "--optimizer-model", "anthropic/other-optimizer",
        "--analyzer-model", "anthropic/other-analyzer",
    )  # fmt: skip
    assert code == 0
    helper_models = {c.model for c in client.calls if c.model.startswith("anthropic")}
    assert helper_models == {"anthropic/other-optimizer", "anthropic/other-analyzer"}


def test_a_missing_optimizer_model_points_to_the_configuration(
    run_cli: CliRunner, init_args: list[str], tmp_path: Path, install_models: Install
) -> None:
    install_models()
    write_dataset(tmp_path)
    args = [a for a in init_args if a not in ("--optimizer-model", "anthropic/model-b")]
    args[args.index("--dir") + 1] = str(tmp_path)
    assert run_cli(*args)[0] == 0
    assert run_cli("data", "split", "--dir", str(tmp_path))[0] == 0
    code, _, err = run_cli("optimize", "--dir", str(tmp_path))
    assert code == 2
    assert "models.optimizer.model in strop.yaml" in err
    code, _, _ = run_cli(
        "optimize", "--dir", str(tmp_path), "--candidates", "2", "--optimizer-model", "anthropic/m"
    )
    assert code == 0


def test_with_models_leaves_other_roles_alone(ready: Path) -> None:
    project = open_project(ready)
    changed = project.with_models({"target": None, "optimizer": "anthropic/x", "judge": "g/j"})
    assert changed.config.models.optimizer is not None
    assert changed.config.models.optimizer.model == "anthropic/x"
    assert changed.config.models.judge is not None
    assert changed.config.models.judge.model == "g/j"
    assert changed.config.models.target == project.config.models.target
    assert project.with_models({"optimizer": None}) is project
