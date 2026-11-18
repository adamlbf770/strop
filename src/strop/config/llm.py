"""Model selection. Strop never defaults to a specific provider or model."""

from typing import Any

from pydantic import Field

from strop.config.base import StrictModel
from strop.config.enums import ModelRole
from strop.errors import ConfigError


class ModelSpec(StrictModel):
    """One model, given as a LiteLLM model string such as ``<provider>/<model>``."""

    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(StrictModel):
    target: ModelSpec
    optimizer: ModelSpec | None = None
    analyzer: ModelSpec | None = None
    judge: ModelSpec | None = None

    def resolve(self, role: ModelRole) -> ModelSpec:
        """Return the model for ``role``; the analyzer falls back to the optimizer."""
        specs = {
            ModelRole.TARGET: self.target,
            ModelRole.OPTIMIZER: self.optimizer,
            ModelRole.ANALYZER: self.analyzer or self.optimizer,
            ModelRole.JUDGE: self.judge,
        }
        spec = specs[role]
        if spec is None:
            raise ConfigError(
                f"no model configured for the '{role.value}' role; set models.{role.value}.model "
                f"in strop.yaml (strop init --{role.value}-model writes it)"
            )
        return spec
