"""Reading and writing strop.yaml."""

from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import YAMLError

from strop.config.root import StropConfig
from strop.errors import ConfigError

HEADER = "Strop project configuration. Command-line flags override these values."


def load_config(path: Path) -> StropConfig:
    """Read and validate ``path``; raise ``ConfigError`` with readable messages."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path} (run 'strop init' first)") from None
    try:
        data = YAML(typ="safe").load(text)
    except YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    try:
        return StropConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {path}:\n{format_errors(exc)}") from exc


def dump_config(config: StropConfig, path: Path, *, force: bool = False) -> None:
    """Write ``config`` to ``path``; refuse to overwrite unless ``force`` is set."""
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists; use --force to overwrite it")
    document = CommentedMap(config.model_dump(mode="json", exclude_none=True))
    document.yaml_set_start_comment(HEADER)
    yaml = YAML()
    yaml.default_flow_style = False
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(document, handle)


def format_errors(exc: ValidationError) -> str:
    """One indented ``field.path: message`` line per validation error."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
