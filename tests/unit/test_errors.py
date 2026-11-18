import pytest

from strop.errors import (
    BudgetExceededError,
    ConfigError,
    DataError,
    ExitCode,
    LLMError,
    StorageError,
    StropError,
)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (StropError, ExitCode.FAILURE),
        (ConfigError, ExitCode.USAGE),
        (DataError, ExitCode.USAGE),
        (StorageError, ExitCode.FAILURE),
        (LLMError, ExitCode.FAILURE),
        (BudgetExceededError, ExitCode.BUDGET),
    ],
)
def test_exit_code_per_error(error: type[StropError], code: ExitCode) -> None:
    assert error.exit_code == code


def test_documented_exit_code_values() -> None:
    assert [c.value for c in ExitCode] == [0, 1, 2, 10, 11, 12]


def test_specific_errors_are_strop_errors() -> None:
    assert issubclass(ConfigError, StropError)
