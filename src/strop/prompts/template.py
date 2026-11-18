"""Variable discovery and substitution for ``{{name}}`` placeholders."""

import re
from collections.abc import Mapping

from strop.errors import ConfigError

_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")
_FIXED = re.compile(r"<strop:fixed>(.*?)</strop:fixed>", re.DOTALL)
FIXED_OPEN = "<strop:fixed>"
FIXED_CLOSE = "</strop:fixed>"


def find_variables(text: str) -> list[str]:
    """Distinct variable names in ``text``, in order of first use."""
    return list(dict.fromkeys(_VARIABLE.findall(text)))


def fixed_spans(text: str) -> list[str]:
    """Contents of every ``<strop:fixed>`` span, in order."""
    return _FIXED.findall(text)


def strip_markers(text: str) -> str:
    """Remove fixed-span markers, keeping their contents."""
    return _FIXED.sub(lambda match: match.group(1), text)


def has_balanced_markers(text: str) -> bool:
    return text.count(FIXED_OPEN) == text.count(FIXED_CLOSE) == len(_FIXED.findall(text))


def substitute(text: str, values: Mapping[str, str]) -> str:
    """Replace placeholders in one pass, so substituted values are never expanded again."""
    return _VARIABLE.sub(lambda match: values[match.group(1)], strip_markers(text))


def check_variables(used: tuple[str, ...], available: tuple[str, ...]) -> None:
    """Raise if the prompt uses a variable that no dataset input column provides."""
    unknown = [name for name in used if name not in available]
    if unknown:
        raise ConfigError(
            f"the prompt uses variables with no matching input column: {', '.join(unknown)}; "
            f"input columns: {', '.join(available)}"
        )
