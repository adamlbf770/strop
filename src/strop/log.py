"""Logging setup. Secrets are redacted before any record is written."""

import logging
import re
import sys
from typing import TextIO

LOGGER_NAME = "strop"

_REDACTED = "[REDACTED]"
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"),
    re.compile(r"(?i)(api[_-]?key[\"'=:\s]+)[\w.\-]{8,}"),
    re.compile(r"()\bsk-[A-Za-z0-9_\-]{8,}"),
)


def redact(text: str) -> str:
    """Replace API keys and bearer tokens in ``text`` with a placeholder."""
    for pattern in _PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}{_REDACTED}", text)
    return text


class RedactingFilter(logging.Filter):
    """Rewrite each record so its final message contains no secrets."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


def configure_logging(
    *, verbose: bool = False, quiet: bool = False, stream: TextIO | None = None
) -> logging.Logger:
    """Configure the ``strop`` logger and return it. Safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)

    logger.setLevel(logging.ERROR if quiet else logging.DEBUG if verbose else logging.WARNING)
    logger.propagate = False
    return logger
