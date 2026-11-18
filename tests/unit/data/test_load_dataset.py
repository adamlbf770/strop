from pathlib import Path
from typing import Any

import pytest

from strop.config import StropConfig
from strop.data import load_dataset, require_valid
from strop.errors import DataError


@pytest.fixture
def config(config_data: dict[str, Any]) -> StropConfig:
    return StropConfig.model_validate(config_data)


def test_loads_valid_rows_and_hashes_the_file(config: StropConfig, tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text(
        "message,category\nhi,refund\nyo,billing\n", encoding="utf-8"
    )
    dataset = load_dataset(config, tmp_path)
    assert len(dataset.rows) == 2
    assert dataset.issues == ()
    assert dataset.total_records == 2
    assert len(dataset.content_hash) == 64


def test_missing_mapped_column_lists_available_columns(config: StropConfig, tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("text,category\nhi,refund\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"no column named 'message'.*available columns: text, cat"):
        load_dataset(config, tmp_path)


def test_issues_block_unless_skipped(config: StropConfig, tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("message,category\nhi,refund\n,billing\n", encoding="utf-8")
    dataset = load_dataset(config, tmp_path)
    assert len(dataset.rows) == 1
    with pytest.raises(DataError, match=r"(?s)1 invalid row.*row 2.*--skip-invalid"):
        require_valid(dataset, skip_invalid=False)
    assert require_valid(dataset, skip_invalid=True) is dataset


def test_dataset_without_valid_rows_is_rejected(config: StropConfig, tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("message,category\n,billing\n", encoding="utf-8")
    with pytest.raises(DataError, match="no valid rows"):
        require_valid(load_dataset(config, tmp_path), skip_invalid=True)
