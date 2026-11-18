from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from strop.config import DatasetFormat, ModelRole, StropConfig
from strop.config.dataset import DatasetConfig
from strop.config.metrics import Bound, RuleSpec
from strop.config.output import OutputConfig
from strop.config.split import SplitConfig
from strop.errors import ConfigError


def test_valid_config_applies_defaults(config_data: dict[str, Any]) -> None:
    config = StropConfig.model_validate(config_data)
    assert config.optimization.max_turns == 5
    assert config.optimization.candidates_per_turn == 3
    assert config.split.seed == 42
    assert config.execution.max_concurrency == 3
    assert config.prompt.file == Path("prompt.yaml")


def test_unknown_key_is_rejected(config_data: dict[str, Any]) -> None:
    config_data["optimization"] = {"max_turn": 3}
    with pytest.raises(ValidationError):
        StropConfig.model_validate(config_data)


def test_dataset_needs_an_input_column() -> None:
    with pytest.raises(ValidationError, match="role 'input'"):
        DatasetConfig(path=Path("d.csv"), columns={"label": "expected"})


def test_dataset_allows_one_expected_column() -> None:
    with pytest.raises(ValidationError, match="at most one"):
        DatasetConfig(
            path=Path("d.csv"),
            columns={"a": "input", "b": "expected", "c": "expected"},
        )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("d.csv", DatasetFormat.CSV),
        ("d.JSONL", DatasetFormat.JSONL),
        ("d.ndjson", DatasetFormat.JSONL),
    ],
)
def test_dataset_format_is_inferred(filename: str, expected: DatasetFormat) -> None:
    dataset = DatasetConfig(path=Path(filename), columns={"a": "input"})
    assert dataset.resolved_format is expected


def test_unknown_dataset_extension_needs_explicit_format() -> None:
    dataset = DatasetConfig(path=Path("d.txt"), columns={"a": "input"})
    with pytest.raises(ConfigError, match=r"dataset.format"):
        _ = dataset.resolved_format


def test_dataset_exposes_roles() -> None:
    dataset = DatasetConfig(
        path=Path("d.csv"),
        columns={"text": "input", "tag": "expected", "when": "metadata"},
    )
    assert dataset.input_columns == ("text",)
    assert dataset.expected_column == "tag"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "label", "labels": ["only-one"]},
        {"type": "label", "labels": ["a", "a"]},
        {"type": "text", "labels": ["a", "b"]},
        {"type": "text", "schema_file": "s.json"},
    ],
)
def test_invalid_output_contracts(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        OutputConfig.model_validate(kwargs)


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match=r"sum to 1.0"):
        SplitConfig(train=0.5, validation=0.2, test=0.2)


def test_split_allows_no_test_partition() -> None:
    split = SplitConfig(train=0.8, validation=0.2, test=0)
    assert split.test == 0


def test_bound_requires_a_limit_and_a_sane_range() -> None:
    with pytest.raises(ValidationError):
        Bound()
    with pytest.raises(ValidationError):
        Bound(min=0.9, max=0.1)
    assert Bound(min=0.9).allows(0.95)
    assert not Bound(min=0.9).allows(0.5)
    assert Bound(max=1.0).allows(1.0)


def test_models_resolve_roles(config_data: dict[str, Any]) -> None:
    config_data["models"]["optimizer"] = {"model": "provider/model-b"}
    models = StropConfig.model_validate(config_data).models
    assert models.resolve(ModelRole.TARGET).model == "provider/model-a"
    assert models.resolve(ModelRole.ANALYZER).model == "provider/model-b"


def test_missing_role_names_the_flag(config_data: dict[str, Any]) -> None:
    models = StropConfig.model_validate(config_data).models
    with pytest.raises(ConfigError, match="--optimizer-model"):
        models.resolve(ModelRole.OPTIMIZER)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "max_words", "value": "ten"},
        {"type": "max_chars", "value": -1},
        {"type": "matches", "value": 5},
        {"type": "matches", "value": "(unclosed"},
        {"type": "contains_all", "value": []},
        {"type": "contains_none", "value": "text"},
    ],
)
def test_invalid_rule_specs(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RuleSpec.model_validate(kwargs)


def test_valid_rule_specs() -> None:
    assert RuleSpec(type="max_words", value=120).value == 120
    assert RuleSpec(type="not_matches", value=r"(?i)guarantee").value == r"(?i)guarantee"
    assert RuleSpec(type="contains_all", value=["a", "b"]).value == ["a", "b"]


def test_match_mode_is_only_for_label_outputs() -> None:
    assert OutputConfig(type="label", labels=("a", "b"), match="contains").match == "contains"
    with pytest.raises(ValidationError, match="only valid for type 'label'"):
        OutputConfig(type="text", match="contains")
