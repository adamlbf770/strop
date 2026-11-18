"""Rendering a template for one dataset row."""

from collections.abc import Mapping

from strop.errors import ConfigError
from strop.llm.types import Message
from strop.prompts.models import PromptTemplate
from strop.prompts.template import substitute


def render_messages(template: PromptTemplate, values: Mapping[str, str]) -> list[Message]:
    """Fill every placeholder from ``values`` and return chat messages for the model."""
    missing = [name for name in template.variables if name not in values]
    if missing:
        raise ConfigError(f"no value for prompt variable(s): {', '.join(missing)}")
    return [Message(role=m.role, content=substitute(m.content, values)) for m in template.messages]
