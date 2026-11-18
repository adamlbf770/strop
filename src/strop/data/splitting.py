"""Deterministic, optionally stratified train/validation/test splits."""

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from strop.config.split import SplitConfig
from strop.data.models import Partition, Row
from strop.errors import DataError

_ORDER = (Partition.TRAIN, Partition.VALIDATION, Partition.TEST)
MIN_RELIABLE_VALIDATION = 30


@dataclass(frozen=True)
class SplitSummary:
    counts: dict[Partition, int]
    warnings: list[str] = field(default_factory=list)


def split_rows(rows: Sequence[Row], config: SplitConfig) -> dict[str, Partition]:
    """Assign every row to a partition. The same rows and config always give the same result."""
    ratios = (config.train, config.validation, config.test)
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r.id):
        key = row.expected if config.stratify and row.expected is not None else ""
        groups[key].append(row)

    rng = random.Random(config.seed)
    assignments: dict[str, Partition] = {}
    for key in sorted(groups):
        members = groups[key]
        rng.shuffle(members)
        start = 0
        for partition, size in zip(_ORDER, _allocate(len(members), ratios), strict=True):
            for row in members[start : start + size]:
                assignments[row.id] = partition
            start += size

    present = set(assignments.values())
    missing = [p.value for p, r in zip(_ORDER, ratios, strict=True) if r > 0 and p not in present]
    if missing:
        raise DataError(
            f"{len(rows)} rows are too few for this split: the {', '.join(missing)} "
            "partition would be empty; add data or change the split ratios"
        )
    return assignments


def summarize(
    rows: Sequence[Row], assignments: dict[str, Partition], config: SplitConfig
) -> SplitSummary:
    """Partition sizes and warnings about splits that give unreliable comparisons."""
    counts = dict.fromkeys(_ORDER, 0)
    for partition in assignments.values():
        counts[partition] += 1
    warnings = []
    if counts[Partition.VALIDATION] < MIN_RELIABLE_VALIDATION:
        warnings.append(
            f"the validation partition has {counts[Partition.VALIDATION]} rows "
            f"(fewer than {MIN_RELIABLE_VALIDATION}); score differences may be noise"
        )
    if counts[Partition.TEST] == 0:
        warnings.append(
            "there is no test partition; the final result cannot be checked on unseen data"
        )
    labels = {r.expected for r in rows if r.expected is not None}
    for partition in (Partition.TRAIN, Partition.VALIDATION):
        seen = {r.expected for r in rows if assignments[r.id] is partition}
        absent = sorted(labels - seen)
        if config.stratify and absent:
            warnings.append(f"no '{partition.value}' rows for: {', '.join(absent)}")
    return SplitSummary(counts, warnings)


def _allocate(size: int, ratios: tuple[float, float, float]) -> list[int]:
    """Split ``size`` into three counts by largest remainder, keeping rare classes represented."""
    ideal = [size * ratio for ratio in ratios]
    counts = [int(value) for value in ideal]
    by_remainder = sorted(range(3), key=lambda i: (-(ideal[i] - counts[i]), i))
    for index in by_remainder[: size - sum(counts)]:
        counts[index] += 1
    for index in (1, 2):
        if ratios[index] > 0 and counts[index] == 0 and size > index:
            donor = max(range(3), key=lambda i: counts[i])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1
    return counts
