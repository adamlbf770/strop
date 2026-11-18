"""The optimizer and analysis code must have no way to reach dataset partitions."""

from pathlib import Path

import pytest

import strop

SOURCE = Path(strop.__file__).parent
FORBIDDEN = (
    "Partition.TEST",
    "PartitionedData",
    "load_partitions",
    "strop.services",
    "strop.storage",
)


@pytest.mark.parametrize("package", ["optimization", "analysis"])
def test_package_never_references_partitions_or_storage(package: str) -> None:
    files = list((SOURCE / package).rglob("*.py"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in text, f"{path.name} must not use {token}"


def test_optimizer_view_has_no_partition_fields() -> None:
    from strop.optimization import OptimizerView  # noqa: PLC0415

    names = set(OptimizerView.__dataclass_fields__)
    assert not {name for name in names if "partition" in name or "test" in name}
