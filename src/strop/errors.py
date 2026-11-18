"""Exception hierarchy and process exit codes.

The command-line layer is the only place that turns an exception into an exit
code. Library code raises a specific ``StropError`` subclass and stays unaware
of how the process ends.
"""

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes documented in the CLI help."""

    SUCCESS = 0
    FAILURE = 1
    USAGE = 2
    PAUSED = 10
    BUDGET = 11
    INTERRUPTED = 12


class StropError(Exception):
    """Base class for all expected, user-facing errors."""

    exit_code: ExitCode = ExitCode.FAILURE


class ConfigError(StropError):
    """The project configuration or a command-line value is missing or invalid."""

    exit_code = ExitCode.USAGE


class DataError(StropError):
    """A dataset file cannot be read or fails validation."""

    exit_code = ExitCode.USAGE


class StorageError(StropError):
    """The local project database or its files cannot be used."""


class LLMError(StropError):
    """A model provider call failed after retries."""


class TransientLLMError(LLMError):
    """A temporary provider failure (rate limit, timeout, outage) that may succeed on retry."""


class InvalidModelReplyError(LLMError):
    """The model kept returning replies that do not match the requested JSON schema."""


class BudgetExceededError(StropError):
    """A configured call, cost, or time budget was reached."""

    exit_code = ExitCode.BUDGET
