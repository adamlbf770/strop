"""Prompt version records."""

from dataclasses import dataclass
from typing import Literal

from strop.errors import ConfigError
from strop.prompts.models import PromptTemplate

Source = Literal["user", "optimizer"]


@dataclass(frozen=True, slots=True)
class PromptVersion:
    """One immutable revision of the prompt, identified as ``v<number>``."""

    number: int
    parent_number: int | None
    template: PromptTemplate
    change_summary: str
    reason: str
    source: Source
    created_at: str

    @property
    def ref(self) -> str:
        return f"v{self.number}"


def parse_version_ref(text: str) -> int:
    """Parse ``v3`` or ``3`` into a version number."""
    digits = text.strip().lower().removeprefix("v")
    if not digits.isdigit() or int(digits) < 1:
        raise ConfigError(f"invalid prompt version '{text}'; use a form like v3")
    return int(digits)
