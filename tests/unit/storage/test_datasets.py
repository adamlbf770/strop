from strop.data.models import Partition
from strop.storage import connect
from strop.storage.datasets import DatasetRepository
from strop.storage.migrator import migrate

ASSIGNMENTS = {"a": Partition.TRAIN, "b": Partition.VALIDATION, "c": Partition.TEST}


def make_repo() -> DatasetRepository:
    conn = connect(":memory:")
    migrate(conn)
    return DatasetRepository(conn)


def save(repo: DatasetRepository, content_hash: str = "h1", *, skip: bool = False) -> int:
    return repo.save_dataset(
        content_hash=content_hash,
        source_path="data.csv",
        row_count=3,
        skipped_rows=0,
        skip_invalid=skip,
    )


def test_saving_the_same_dataset_twice_returns_the_same_id() -> None:
    repo = make_repo()
    assert save(repo) == save(repo)
    assert save(repo, skip=True) != save(repo)


def test_split_round_trip() -> None:
    repo = make_repo()
    dataset_id = save(repo)
    split_id = repo.create_split(
        dataset_id=dataset_id,
        seed=42,
        strategy="stratified",
        config_json="{}",
        assignments=ASSIGNMENTS,
    )
    assert repo.assignments(split_id) == ASSIGNMENTS
    assert repo.find_split(dataset_id, "{}") == split_id
    assert repo.find_split(dataset_id, "other") is None


def test_latest_split_describes_its_dataset() -> None:
    repo = make_repo()
    assert repo.latest_split() is None
    first = save(repo, "h1")
    second = save(repo, "h2", skip=True)
    for dataset_id in (first, second):
        repo.create_split(
            dataset_id=dataset_id,
            seed=1,
            strategy="random",
            config_json="{}",
            assignments=ASSIGNMENTS,
        )
    latest = repo.latest_split()
    assert latest is not None
    assert latest.content_hash == "h2"
    assert latest.skip_invalid is True
    assert latest.dataset_id == second
