from pathlib import Path
from typing import Any

import pytest

from strop.config import StropConfig, dump_config, load_config
from strop.errors import ConfigError


def test_round_trip_keeps_every_value(config_data: dict[str, Any], tmp_path: Path) -> None:
    original = StropConfig.model_validate(config_data)
    target = tmp_path / "strop.yaml"
    dump_config(original, target)
    assert load_config(target) == original


def test_written_file_starts_with_a_header_comment(
    config_data: dict[str, Any], tmp_path: Path
) -> None:
    target = tmp_path / "strop.yaml"
    dump_config(StropConfig.model_validate(config_data), target)
    assert target.read_text(encoding="utf-8").startswith("# Strop project configuration")


def test_dump_refuses_to_overwrite_without_force(
    config_data: dict[str, Any], tmp_path: Path
) -> None:
    config = StropConfig.model_validate(config_data)
    target = tmp_path / "strop.yaml"
    dump_config(config, target)
    with pytest.raises(ConfigError, match="--force"):
        dump_config(config, target)
    dump_config(config, target, force=True)


def test_missing_file_suggests_init(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="strop init"):
        load_config(tmp_path / "strop.yaml")


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    target = tmp_path / "strop.yaml"
    target.write_text("project: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(target)


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "strop.yaml"
    target.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(target)


def test_validation_errors_name_the_field(tmp_path: Path) -> None:
    target = tmp_path / "strop.yaml"
    target.write_text("project:\n  name: demo\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(target)
    message = str(info.value)
    assert "project.task_description" in message
    assert "dataset" in message
