"""Dataset location and column roles."""

from pathlib import Path
from typing import Self

from pydantic import model_validator

from strop.config.base import StrictModel
from strop.config.enums import ColumnRole, DatasetFormat
from strop.errors import ConfigError

_SUFFIX_FORMATS = {
    ".csv": DatasetFormat.CSV,
    ".jsonl": DatasetFormat.JSONL,
    ".ndjson": DatasetFormat.JSONL,
}


class DatasetConfig(StrictModel):
    path: Path
    format: DatasetFormat | None = None
    id_column: str | None = None
    columns: dict[str, ColumnRole]

    @model_validator(mode="after")
    def _check_roles(self) -> Self:
        roles = list(self.columns.values())
        if ColumnRole.INPUT not in roles:
            raise ValueError("at least one column must have the role 'input'")
        if roles.count(ColumnRole.EXPECTED) > 1:
            raise ValueError("at most one column may have the role 'expected'")
        for name in self._columns_with(ColumnRole.INPUT):
            if not name.isidentifier():
                raise ValueError(
                    f"input column '{name}' cannot be used as a prompt variable; "
                    "rename it to letters, digits, and underscores, not starting with a digit"
                )
        return self

    @property
    def resolved_format(self) -> DatasetFormat:
        """The explicit format, or the one implied by the file extension."""
        if self.format is not None:
            return self.format
        try:
            return _SUFFIX_FORMATS[self.path.suffix.lower()]
        except KeyError:
            raise ConfigError(
                f"cannot infer the dataset format from '{self.path.name}'; "
                "set dataset.format to csv or jsonl"
            ) from None

    @property
    def input_columns(self) -> tuple[str, ...]:
        return self._columns_with(ColumnRole.INPUT)

    @property
    def expected_column(self) -> str | None:
        expected = self._columns_with(ColumnRole.EXPECTED)
        return expected[0] if expected else None

    def _columns_with(self, role: ColumnRole) -> tuple[str, ...]:
        return tuple(name for name, value in self.columns.items() if value == role)
