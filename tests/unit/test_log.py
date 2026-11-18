import io
import logging

import pytest

from strop.log import LOGGER_NAME, configure_logging, redact


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer abc123def456",
        "api_key=abcdef123456",
        'api-key: "abcdef123456"',
        "using sk-abcdefghij1234 for the call",
    ],
)
def test_redact_removes_secrets(text: str) -> None:
    cleaned = redact(text)
    assert "[REDACTED]" in cleaned
    assert "abc123def456" not in cleaned
    assert "abcdef123456" not in cleaned
    assert "abcdefghij1234" not in cleaned


def test_redact_leaves_plain_text_alone() -> None:
    assert redact("loaded 120 rows from data.csv") == "loaded 120 rows from data.csv"


def test_logger_writes_redacted_message() -> None:
    stream = io.StringIO()
    logger = configure_logging(stream=stream)
    logger.warning("request failed, api_key=%s", "supersecretvalue")
    output = stream.getvalue()
    assert "supersecretvalue" not in output
    assert "[REDACTED]" in output


def test_levels_follow_flags() -> None:
    assert configure_logging().level == logging.WARNING
    assert configure_logging(verbose=True).level == logging.DEBUG
    assert configure_logging(quiet=True).level == logging.ERROR


def test_configure_is_idempotent() -> None:
    configure_logging()
    configure_logging()
    assert len(logging.getLogger(LOGGER_NAME).handlers) == 1
