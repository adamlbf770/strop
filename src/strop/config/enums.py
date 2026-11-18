"""Enumerations used across the configuration sections."""

from enum import StrEnum


class DatasetFormat(StrEnum):
    CSV = "csv"
    JSONL = "jsonl"


class ColumnRole(StrEnum):
    """What a dataset column is used for."""

    INPUT = "input"
    EXPECTED = "expected"
    METADATA = "metadata"
    IGNORE = "ignore"


class OutputType(StrEnum):
    """The kind of output the prompted model is expected to produce."""

    LABEL = "label"
    JSON = "json"
    TEXT = "text"


class ApprovalMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


class ModelRole(StrEnum):
    """Independent model roles; each can use a different provider and model."""

    TARGET = "target"
    OPTIMIZER = "optimizer"
    ANALYZER = "analyzer"
    JUDGE = "judge"


class RuleType(StrEnum):
    """Deterministic checks applied to each model output."""

    MAX_CHARS = "max_chars"
    MIN_CHARS = "min_chars"
    MAX_WORDS = "max_words"
    MIN_WORDS = "min_words"
    MATCHES = "matches"
    NOT_MATCHES = "not_matches"
    CONTAINS_ALL = "contains_all"
    CONTAINS_NONE = "contains_none"
