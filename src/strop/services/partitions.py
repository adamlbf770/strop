"""Loading the dataset rows of the active split, grouped by partition."""

from dataclasses import dataclass, field

from strop.data import load_dataset, require_valid
from strop.data.models import Partition, Row
from strop.errors import ConfigError, DataError
from strop.services.project import Project
from strop.storage.datasets import DatasetRepository


@dataclass(frozen=True)
class PartitionedData:
    """Rows of the latest split. The test partition is only handed out on explicit request."""

    split_id: int
    seed: int
    rows: dict[Partition, list[Row]] = field(default_factory=dict)

    def get(self, partition: Partition) -> list[Row]:
        return self.rows.get(partition, [])


def load_partitions(project: Project, repo: DatasetRepository) -> PartitionedData:
    """Read the dataset and attach the stored split assignments.

    Fails if there is no split or if the dataset file changed since the split was made.
    """
    split = repo.latest_split()
    if split is None:
        raise ConfigError("no data split yet; run 'strop data split' first")
    dataset = require_valid(
        load_dataset(project.config, project.root), skip_invalid=split.skip_invalid
    )
    if dataset.content_hash != split.content_hash:
        raise DataError(
            "the dataset file changed since the split was created; run 'strop data split' again"
        )
    assignments = repo.assignments(split.id)
    rows: dict[Partition, list[Row]] = {p: [] for p in Partition}
    for row in dataset.rows:
        partition = assignments.get(row.id)
        if partition is None:
            raise DataError(f"row '{row.id}' has no partition; run 'strop data split' again")
        rows[partition].append(row)
    return PartitionedData(split.id, split.seed, rows)
