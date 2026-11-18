from typing import Any

from strop.config import StropConfig
from strop.data.loaders import RawRecord
from strop.data.rows import build_rows, row_hash, to_text


def make_config(config_data: dict[str, Any], **dataset: Any) -> StropConfig:
    config_data["dataset"].update(dataset)
    return StropConfig.model_validate(config_data)


def record(number: int, **values: Any) -> RawRecord:
    return RawRecord(number, values)


def test_to_text_normalizes_values() -> None:
    assert to_text("a") == "a"
    assert to_text(None) == ""
    assert to_text(3) == "3"
    assert to_text(True) == "true"
    assert to_text({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'


def test_valid_rows_get_positional_ids(config_data: dict[str, Any]) -> None:
    config = make_config(config_data)
    rows, issues = build_rows(
        [record(1, message="hello", category="refund")], config.dataset, config.output
    )
    assert issues == []
    assert rows[0].id == "row_0001"
    assert rows[0].inputs == {"message": "hello"}
    assert rows[0].expected == "refund"


def test_id_column_is_used_and_duplicates_are_reported(config_data: dict[str, Any]) -> None:
    config_data["dataset"]["columns"]["ticket"] = "metadata"
    config = make_config(config_data, id_column="ticket")
    rows, issues = build_rows(
        [
            record(1, ticket="T1", message="a", category="refund"),
            record(2, ticket="T1", message="b", category="billing"),
        ],
        config.dataset,
        config.output,
    )
    assert [r.id for r in rows] == ["T1"]
    assert [str(i) for i in issues] == ["row 2: duplicate id 'T1'"]


def test_empty_cells_and_unknown_labels_are_reported(config_data: dict[str, Any]) -> None:
    config = make_config(config_data)
    _, issues = build_rows(
        [
            record(1, message="  ", category="refund"),
            record(2, message="ok", category=""),
            record(3, message="ok", category="refnd"),
        ],
        config.dataset,
        config.output,
    )
    messages = [str(i) for i in issues]
    assert "row 1: input column 'message' is empty" in messages
    assert "row 2: expected column 'category' is empty" in messages
    assert "row 3: expected value 'refnd' is not one of the output labels" in messages


def test_parse_errors_are_carried_through(config_data: dict[str, Any]) -> None:
    config = make_config(config_data)
    failed = RawRecord(5, {}, "is not valid JSON (boom)")
    _, issues = build_rows([failed], config.dataset, config.output)
    assert str(issues[0]) == "row 5: is not valid JSON (boom)"


def test_row_hash_depends_on_content() -> None:
    base = row_hash({"a": "1"}, "x")
    assert base == row_hash({"a": "1"}, "x")
    assert base != row_hash({"a": "2"}, "x")
    assert base != row_hash({"a": "1"}, "y")
