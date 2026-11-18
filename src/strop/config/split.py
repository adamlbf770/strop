"""Train, validation, and test partition settings."""

import math
from typing import Self

from pydantic import Field, model_validator

from strop.config.base import StrictModel


class SplitConfig(StrictModel):
    train: float = Field(default=0.6, gt=0, lt=1)
    validation: float = Field(default=0.2, gt=0, lt=1)
    test: float = Field(default=0.2, ge=0, lt=1)
    seed: int = 42
    stratify: bool = True

    @model_validator(mode="after")
    def _check_ratios(self) -> Self:
        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"split ratios must sum to 1.0, got {total:.3f}")
        return self
