"""Loading API keys and other settings from a project's .env file."""

import logging
import os
import re
from pathlib import Path

from strop.log import LOGGER_NAME

LOG = logging.getLogger(LOGGER_NAME)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(path: Path) -> list[str]:
    """Set variables from ``path`` that are not already in the environment.

    The format is ``NAME=value`` per line, with optional ``export`` prefix, ``#`` comments, and
    single or double quotes around the value. Variables already set are never overridden, and
    values are never logged. Returns the names that were set.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    loaded = []
    for raw in text.splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if not _NAME.match(name) or name in os.environ:
            continue
        os.environ[name] = _unquote(value.strip())
        loaded.append(name)
    if loaded:
        LOG.debug("loaded %d variable(s) from %s", len(loaded), path.name)
    return loaded


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value.split(" #", 1)[0].rstrip()
