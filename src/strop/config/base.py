"""Shared base class for configuration models."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Immutable model that rejects unknown keys, so typos in strop.yaml fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)
