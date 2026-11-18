"""Parsing of ``--rule`` flags and rubric files used by ``strop init``."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from strop.config.enums import RuleType
from strop.config.io import format_errors
from strop.config.metrics import JudgeSpec, RuleSpec
from strop.errors import ConfigError

_COUNT = {RuleType.MAX_CHARS, RuleType.MIN_CHARS, RuleType.MAX_WORDS, RuleType.MIN_WORDS}
_LISTS = {RuleType.CONTAINS_ALL, RuleType.CONTAINS_NONE}
FORM = "use the form name=type:value, for example short=max_words:120"


def parse_rules(expressions: Iterable[str]) -> dict[str, RuleSpec]:
    """Turn ``name=type:value`` expressions into rule specs."""
    rules: dict[str, RuleSpec] = {}
    for text in expressions:
        name, equals, rest = text.partition("=")
        kind, colon, raw = rest.partition(":")
        if not (equals and colon and name.strip() and raw):
            raise ConfigError(f"invalid rule '{text}'; {FORM}")
        if name.strip() in rules:
            raise ConfigError(f"rule '{name.strip()}' is defined more than once")
        rules[name.strip()] = _spec(text, kind.strip(), raw)
    return rules


def _spec(text: str, kind: str, raw: str) -> RuleSpec:
    try:
        rule_type = RuleType(kind)
    except ValueError:
        known = ", ".join(t.value for t in RuleType)
        raise ConfigError(f"unknown rule type '{kind}' in '{text}'; types: {known}") from None
    value: Any = raw
    if rule_type in _COUNT:
        if not raw.strip().isdigit():
            raise ConfigError(f"rule '{text}' needs a whole number after the colon")
        value = int(raw)
    elif rule_type in _LISTS:
        value = [item.strip() for item in raw.split(",") if item.strip()]
    try:
        return RuleSpec(type=rule_type, value=value)
    except ValidationError as exc:
        raise ConfigError(f"invalid rule '{text}':\n{format_errors(exc)}") from exc


def load_rubrics(path: Path) -> dict[str, JudgeSpec]:
    """Read a YAML file mapping rubric names to ``criteria``, ``scale_min``, ``scale_max``."""
    try:
        data = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read the rubric file {path}: {exc.strerror}") from exc
    except YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ConfigError(f"{path.name} must map each rubric name to its settings")
    try:
        return {str(name): JudgeSpec.model_validate(spec) for name, spec in data.items()}
    except ValidationError as exc:
        raise ConfigError(f"invalid rubric in {path.name}:\n{format_errors(exc)}") from exc
