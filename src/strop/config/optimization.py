"""Optimization loop settings."""

from pydantic import Field

from strop.config.base import StrictModel
from strop.config.enums import ApprovalMode


class OptimizationConfig(StrictModel):
    max_turns: int = Field(default=5, ge=1)
    candidates_per_turn: int = Field(default=3, ge=1)
    target: float | None = None
    patience: int = Field(default=2, ge=1)
    approval: ApprovalMode = ApprovalMode.AUTO
    min_improvement: float = Field(default=0.01, ge=0)
    train_sample: int | None = Field(default=None, gt=0)
    instructions: str = ""
    evidence_examples: int = Field(default=12, ge=1)
    evidence_chars: int = Field(default=400, ge=40)
    max_class_regression: float | None = Field(default=0.10, ge=0)
    max_validity_regression: float = Field(default=0.02, ge=0)
    min_validation_samples: int = Field(default=10, ge=1)
    require_paired_wins: bool = True
