"""Budgets and execution settings."""

from pydantic import Field

from strop.config.base import StrictModel


class BudgetsConfig(StrictModel):
    """Upper limits for a run. ``None`` disables a limit."""

    max_total_model_calls: int | None = Field(default=500, gt=0)
    max_total_cost_usd: float | None = Field(default=None, gt=0)
    max_wall_time_minutes: float | None = Field(default=60.0, gt=0)


class ExecutionConfig(StrictModel):
    batch_size: int = Field(default=10, ge=1)
    max_concurrency: int = Field(default=3, ge=1)
    retry_attempts: int = Field(default=3, ge=1)
