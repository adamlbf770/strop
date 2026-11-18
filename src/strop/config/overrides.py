"""Applying command-line overrides on top of a configuration section."""

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import ValidationError

from strop.config.base import StrictModel
from strop.config.io import format_errors
from strop.errors import ConfigError

T = TypeVar("T", bound=StrictModel)


def apply_overrides(section: T, overrides: Mapping[str, Any]) -> T:
    """A validated copy of ``section`` with every non-None override applied."""
    values = section.model_dump()
    values.update({key: value for key, value in overrides.items() if value is not None})
    try:
        return type(section).model_validate(values)
    except ValidationError as exc:
        raise ConfigError(f"invalid options:\n{format_errors(exc)}") from exc
