"""Dataset loading, validation, and splitting."""

from pathlib import Path

from strop.config import StropConfig
from strop.data.loaders import file_hash, read_records
from strop.data.models import Dataset, Partition, Row, RowIssue
from strop.data.rows import build_rows
from strop.data.splitting import SplitSummary, split_rows, summarize
from strop.errors import DataError

MAX_LISTED_ISSUES = 20


def load_dataset(config: StropConfig, root: Path) -> Dataset:
    """Read and validate the configured dataset. Invalid rows are reported, not dropped silently."""
    path = root / config.dataset.path
    columns, records = read_records(path, config.dataset.resolved_format)
    wanted = [
        *config.dataset.columns,
        *([config.dataset.id_column] if config.dataset.id_column else []),
    ]
    missing = [name for name in wanted if name not in columns]
    if missing:
        raise DataError(
            f"{path.name} has no column named {', '.join(repr(m) for m in missing)}; "
            f"available columns: {', '.join(columns)}"
        )
    rows, issues = build_rows(records, config.dataset, config.output)
    return Dataset(tuple(rows), tuple(issues), len(records), file_hash(path), columns)


def require_valid(dataset: Dataset, *, skip_invalid: bool) -> Dataset:
    """Return ``dataset`` if it has no issues or ``skip_invalid`` is set; otherwise raise."""
    if dataset.issues and not skip_invalid:
        shown = "\n".join(f"  {issue}" for issue in dataset.issues[:MAX_LISTED_ISSUES])
        extra = len(dataset.issues) - MAX_LISTED_ISSUES
        more = f"\n  ... and {extra} more" if extra > 0 else ""
        raise DataError(
            f"{len(dataset.issues)} invalid row(s) in the dataset:\n{shown}{more}\n"
            "fix the file, or pass --skip-invalid to exclude these rows"
        )
    if not dataset.rows:
        raise DataError("the dataset has no valid rows")
    return dataset


__all__ = [
    "Dataset",
    "Partition",
    "Row",
    "RowIssue",
    "SplitSummary",
    "load_dataset",
    "require_valid",
    "split_rows",
    "summarize",
]
