"""Dataset workflows: validation and creating persisted splits."""

from dataclasses import dataclass
from typing import Any

from strop.config.overrides import apply_overrides
from strop.config.split import SplitConfig
from strop.data import Dataset, SplitSummary, load_dataset, require_valid, split_rows, summarize
from strop.services.project import Project
from strop.storage.datasets import DatasetRepository


@dataclass(frozen=True)
class SplitOutcome:
    split_id: int
    created: bool
    dataset: Dataset
    config: SplitConfig
    summary: SplitSummary


def check_dataset(project: Project) -> Dataset:
    """Load and validate the dataset without raising on invalid rows."""
    return load_dataset(project.config, project.root)


def effective_split_config(project: Project, overrides: dict[str, Any]) -> SplitConfig:
    """The configured split settings with any non-None command-line overrides applied."""
    return apply_overrides(project.config.split, overrides)


def create_split(project: Project, split: SplitConfig, *, skip_invalid: bool) -> SplitOutcome:
    """Split the dataset and persist it. An identical existing split is reused, not recomputed."""
    dataset = require_valid(check_dataset(project), skip_invalid=skip_invalid)
    conn = project.connect()
    try:
        repo = DatasetRepository(conn)
        dataset_id = repo.save_dataset(
            content_hash=dataset.content_hash,
            source_path=str(project.config.dataset.path),
            row_count=len(dataset.rows),
            skipped_rows=dataset.total_records - len(dataset.rows),
            skip_invalid=skip_invalid,
        )
        config_json = split.model_dump_json()
        split_id = repo.find_split(dataset_id, config_json)
        created = split_id is None
        if split_id is None:
            assignments = split_rows(dataset.rows, split)
            stratified = split.stratify and project.config.dataset.expected_column is not None
            strategy = "stratified" if stratified else "random"
            split_id = repo.create_split(
                dataset_id=dataset_id,
                seed=split.seed,
                strategy=strategy,
                config_json=config_json,
                assignments=assignments,
            )
        assignments = repo.assignments(split_id)
    finally:
        conn.close()
    return SplitOutcome(
        split_id, created, dataset, split, summarize(dataset.rows, assignments, split)
    )
