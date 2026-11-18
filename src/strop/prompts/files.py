"""Reading and writing prompt files (``.yaml``/``.yml`` or plain ``.md``/``.txt``)."""

from io import StringIO
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.scalarstring import LiteralScalarString

from strop.config.io import format_errors
from strop.errors import ConfigError
from strop.llm.types import Role
from strop.prompts.models import PromptMessage, PromptTemplate
from strop.prompts.template import has_balanced_markers

_YAML_SUFFIXES = {".yaml", ".yml"}


class _MessageModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    content: str = Field(min_length=1)


class _PromptFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[_MessageModel] = Field(min_length=1)
    optimize: list[Role] | None = None


def load_prompt_file(path: Path) -> PromptTemplate:
    """Parse a prompt file; raise ``ConfigError`` with a readable message if it is invalid."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"prompt file not found: {path}") from None
    return parse_prompt(text, path.suffix.lower(), source=path.name)


def parse_prompt(text: str, suffix: str, *, source: str = "prompt") -> PromptTemplate:
    if suffix in _YAML_SUFFIXES:
        template = _parse_yaml(text, source)
    else:
        if not text.strip():
            raise ConfigError(f"{source} is empty")
        template = PromptTemplate((PromptMessage("user", text),), frozenset({"user"}))
    for message in template.messages:
        if not has_balanced_markers(message.content):
            raise ConfigError(f"{source} has an unclosed or stray <strop:fixed> marker")
    return template


def _parse_yaml(text: str, source: str) -> PromptTemplate:
    try:
        data = YAML(typ="safe").load(text)
    except YAMLError as exc:
        raise ConfigError(f"{source} is not valid YAML: {exc}") from exc
    try:
        parsed = _PromptFileModel.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid prompt file {source}:\n{format_errors(exc)}") from exc
    messages = tuple(PromptMessage(m.role, m.content) for m in parsed.messages)
    roles = {m.role for m in messages}
    optimize = set(parsed.optimize) if parsed.optimize is not None else {_default_role(roles)}
    unknown = optimize - roles
    if unknown or not optimize:
        raise ConfigError(
            f"{source}: 'optimize' must name at least one role present in the messages "
            f"(found roles: {', '.join(sorted(roles))})"
        )
    return PromptTemplate(messages, frozenset(optimize))


def _default_role(roles: set[Role]) -> Role:
    return "system" if "system" in roles else "user"


def dump_prompt(template: PromptTemplate) -> str:
    """Serialize ``template`` as prompt YAML with readable multi-line blocks."""
    document = {
        "messages": [
            {"role": m.role, "content": LiteralScalarString(m.content)} for m in template.messages
        ],
        "optimize": sorted(template.optimize),
    }
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    buffer = StringIO()
    yaml.dump(document, buffer)
    return buffer.getvalue()
