"""Dataset domain types."""

from dataclasses import dataclass
from enum import StrEnum


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class Row:
    """One valid dataset row, reduced to the columns Strop uses."""

    id: str
    number: int
    inputs: dict[str, str]
    expected: str | None
    metadata: dict[str, str]
    hash: str


@dataclass(frozen=True, slots=True)
class RowIssue:
    """A problem with one record; ``number`` is its CSV record or JSONL line number."""

    number: int
    message: str

    def __str__(self) -> str:
        return f"row {self.number}: {self.message}"


@dataclass(frozen=True)
class Dataset:
    """A loaded dataset: the valid rows plus every issue found, never silently dropped."""

    rows: tuple[Row, ...]
    issues: tuple[RowIssue, ...]
    total_records: int
    content_hash: str
    columns: tuple[str, ...]
