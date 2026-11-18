"""Output contract: what the prompted model must return."""

from pathlib import Path
from typing import Self

from pydantic import model_validator

from strop.config.base import StrictModel
from strop.config.enums import OutputType


class OutputConfig(StrictModel):
    type: OutputType
    labels: tuple[str, ...] = ()
    schema_file: Path | None = None

    @model_validator(mode="after")
    def _check_contract(self) -> Self:
        if self.type is OutputType.LABEL:
            if len(self.labels) < 2:
                raise ValueError("output.labels needs at least two labels for type 'label'")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError("output.labels must not contain duplicates")
        elif self.labels:
            raise ValueError("output.labels is only valid for type 'label'")
        if self.schema_file is not None and self.type is not OutputType.JSON:
            raise ValueError("output.schema_file is only valid for type 'json'")
        return self
