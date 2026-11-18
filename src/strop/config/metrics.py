"""Success definition: primary metric, rule checks, and constraints a candidate must keep."""

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from strop.config.base import StrictModel
from strop.config.enums import RuleType

_COUNT_RULES = {RuleType.MAX_CHARS, RuleType.MIN_CHARS, RuleType.MAX_WORDS, RuleType.MIN_WORDS}
_PATTERN_RULES = {RuleType.MATCHES, RuleType.NOT_MATCHES}


class Bound(StrictModel):
    """An inclusive lower and/or upper limit on a metric value."""

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("a constraint needs 'min' or 'max'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("'min' must not be greater than 'max'")
        return self

    def allows(self, value: float) -> bool:
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


class RuleSpec(StrictModel):
    """One deterministic output check. The rule's pass rate becomes the metric ``rule:<name>``."""

    type: RuleType
    value: int | str | list[str]

    @model_validator(mode="after")
    def _check_value(self) -> Self:
        if self.type in _COUNT_RULES:
            if not isinstance(self.value, int) or isinstance(self.value, bool) or self.value < 0:
                raise ValueError(f"rule type '{self.type}' needs a non-negative integer value")
        elif self.type in _PATTERN_RULES:
            if not isinstance(self.value, str):
                raise ValueError(f"rule type '{self.type}' needs a regular expression string")
            try:
                re.compile(self.value)
            except re.error as exc:
                raise ValueError(f"invalid regular expression: {exc}") from exc
        elif not isinstance(self.value, list) or not self.value:
            raise ValueError(f"rule type '{self.type}' needs a non-empty list of strings")
        return self


class PythonMetricConfig(StrictModel):
    """A user-written metric: a Python file defining ``score(output, row)``."""

    file: Path
    timeout_s: float = Field(default=120.0, gt=0)
    direction: Literal["maximize", "minimize"] = "maximize"


class JudgeSpec(StrictModel):
    """A rubric scored by a judge model. The score is normalized to the range 0 to 1."""

    criteria: str = Field(min_length=1)
    scale_min: int = 1
    scale_max: int = 5
    instructions: str = ""

    @model_validator(mode="after")
    def _check_scale(self) -> Self:
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        return self


class MetricsConfig(StrictModel):
    primary: str = Field(min_length=1)
    secondary: tuple[str, ...] = ()
    constraints: dict[str, Bound] = Field(default_factory=dict)
    rules: dict[str, RuleSpec] = Field(default_factory=dict)
    python: PythonMetricConfig | None = None
    judges: dict[str, JudgeSpec] = Field(default_factory=dict)
