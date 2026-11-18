"""Success definition: the primary metric and the constraints a candidate must keep."""

from typing import Self

from pydantic import Field, model_validator

from strop.config.base import StrictModel


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


class MetricsConfig(StrictModel):
    primary: str = Field(min_length=1)
    secondary: tuple[str, ...] = ()
    constraints: dict[str, Bound] = Field(default_factory=dict)
