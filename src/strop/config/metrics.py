"""Success definition: primary metric, rule checks, and constraints a candidate must keep."""

import re
from typing import Self

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


class MetricsConfig(StrictModel):
    primary: str = Field(min_length=1)
    secondary: tuple[str, ...] = ()
    constraints: dict[str, Bound] = Field(default_factory=dict)
    rules: dict[str, RuleSpec] = Field(default_factory=dict)
