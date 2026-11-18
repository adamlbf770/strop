"""Turning raw model text into a value that metrics can score."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from strop.config.enums import OutputType
from strop.config.output import OutputConfig
from strop.errors import ConfigError
from strop.metrics.types import ErrorCode

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_QUOTES = "`'\"" + "".join(chr(code) for code in (0x201C, 0x201D, 0x2018, 0x2019))


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    value: str | None
    parsed: Any = None
    error: ErrorCode | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def canonical_expected(text: str, output_type: OutputType) -> str:
    """Normalize an expected value the same way outputs are normalized."""
    stripped = text.strip()
    if output_type is OutputType.JSON:
        return canonical_json(json.loads(stripped))
    return stripped


def load_output_schema(root: Path, output: OutputConfig) -> dict[str, Any] | None:
    """Read and check the JSON schema named in the output contract, if any."""
    if output.schema_file is None:
        return None
    path = root / output.schema_file
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except FileNotFoundError:
        raise ConfigError(f"output schema file not found: {path}") from None
    except (ValueError, SchemaError) as exc:
        raise ConfigError(f"output schema {path.name} is not a valid JSON schema: {exc}") from exc
    return schema  # type: ignore[no-any-return]


def parse_output(
    raw: str, finish_reason: str | None, output: OutputConfig, schema: dict[str, Any] | None = None
) -> ParsedOutput:
    """Normalize ``raw`` under the output contract; failures carry an error code."""
    text = raw.strip()
    if not text:
        return ParsedOutput(None, error=ErrorCode.EMPTY_OUTPUT)
    failure = ErrorCode.TRUNCATED if finish_reason == "length" else ErrorCode.INVALID_FORMAT
    if output.type is OutputType.LABEL:
        label = _match_label(text, output)
        return ParsedOutput(label) if label else ParsedOutput(None, error=failure)
    if output.type is OutputType.JSON:
        return _parse_json(text, schema, failure)
    if finish_reason == "length":
        return ParsedOutput(None, error=ErrorCode.TRUNCATED)
    return ParsedOutput(text)


def _match_label(text: str, output: OutputConfig) -> str | None:
    by_name = {label.casefold(): label for label in output.labels}
    candidate = text.strip(_QUOTES + " \t\r\n").rstrip(".!").strip(_QUOTES + " ")
    if candidate.casefold() in by_name:
        return by_name[candidate.casefold()]
    if output.match != "contains":
        return None
    found = [
        label
        for key, label in by_name.items()
        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text.casefold())
    ]
    return found[0] if len(found) == 1 else None


def _parse_json(text: str, schema: dict[str, Any] | None, failure: ErrorCode) -> ParsedOutput:
    fenced = _FENCE.match(text)
    try:
        parsed = json.loads(fenced.group(1) if fenced else text)
    except ValueError:
        return ParsedOutput(None, error=failure)
    if schema is not None and next(Draft202012Validator(schema).iter_errors(parsed), None):
        return ParsedOutput(None, parsed, ErrorCode.SCHEMA_VIOLATION)
    return ParsedOutput(canonical_json(parsed), parsed)
