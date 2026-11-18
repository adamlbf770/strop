"""Turning raw records into validated rows."""

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from strop.config.dataset import DatasetConfig
from strop.config.enums import OutputType
from strop.config.output import OutputConfig
from strop.data.loaders import RawRecord
from strop.data.models import Row, RowIssue


def to_text(value: Any) -> str:
    """Normalize a cell to text: strings as-is, JSON-style for everything else."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bool | int | float | dict | list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def row_hash(inputs: dict[str, str], expected: str | None) -> str:
    """Stable hash of the content a model call or metric depends on."""
    payload = json.dumps(
        {"inputs": inputs, "expected": expected}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_rows(
    records: Iterable[RawRecord], dataset: DatasetConfig, output: OutputConfig
) -> tuple[list[Row], list[RowIssue]]:
    """Validate ``records`` and return the valid rows and the issues found."""
    rows: list[Row] = []
    issues: list[RowIssue] = []
    seen: set[str] = set()
    for record in records:
        if record.error:
            issues.append(RowIssue(record.number, record.error))
            continue
        problems = _check_record(record, dataset, output)
        row_id = _row_id(record, dataset)
        if row_id in seen:
            problems.append(f"duplicate id '{row_id}'")
        if problems:
            issues.extend(RowIssue(record.number, text) for text in problems)
            continue
        seen.add(row_id)
        rows.append(_make_row(record, row_id, dataset))
    return rows, issues


def _row_id(record: RawRecord, dataset: DatasetConfig) -> str:
    if dataset.id_column is None:
        return f"row_{record.number:04d}"
    return to_text(record.values.get(dataset.id_column)).strip()


def _check_record(record: RawRecord, dataset: DatasetConfig, output: OutputConfig) -> list[str]:
    problems = []
    if dataset.id_column and not _row_id(record, dataset):
        problems.append(f"id column '{dataset.id_column}' is empty")
    for column in dataset.input_columns:
        if not to_text(record.values.get(column)).strip():
            problems.append(f"input column '{column}' is empty")
    expected_column = dataset.expected_column
    if expected_column:
        expected = to_text(record.values.get(expected_column)).strip()
        if not expected:
            problems.append(f"expected column '{expected_column}' is empty")
        elif output.type is OutputType.LABEL and expected not in output.labels:
            problems.append(f"expected value '{expected}' is not one of the output labels")
        elif output.type is OutputType.JSON and not _is_json(expected):
            problems.append(f"expected column '{expected_column}' does not contain valid JSON")
    return problems


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _make_row(record: RawRecord, row_id: str, dataset: DatasetConfig) -> Row:
    inputs = {name: to_text(record.values.get(name)) for name in dataset.input_columns}
    expected_column = dataset.expected_column
    expected = to_text(record.values.get(expected_column)).strip() if expected_column else None
    metadata = {
        name: to_text(record.values.get(name))
        for name, role in dataset.columns.items()
        if role.value == "metadata"
    }
    return Row(row_id, record.number, inputs, expected, metadata, row_hash(inputs, expected))
