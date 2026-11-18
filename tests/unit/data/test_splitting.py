from collections import Counter

import pytest

from strop.config.split import SplitConfig
from strop.data.models import Partition, Row
from strop.data.splitting import split_rows, summarize
from strop.errors import DataError


def make_rows(per_label: dict[str, int]) -> list[Row]:
    rows: list[Row] = []
    for label, count in per_label.items():
        for i in range(count):
            rows.append(Row(f"{label}-{i:03d}", len(rows) + 1, {"t": "x"}, label, {}, "h"))
    return rows


def test_split_is_deterministic() -> None:
    rows = make_rows({"a": 50, "b": 50})
    config = SplitConfig()
    assert split_rows(rows, config) == split_rows(list(reversed(rows)), config)


def test_seed_changes_the_assignment() -> None:
    rows = make_rows({"a": 50, "b": 50})
    assert split_rows(rows, SplitConfig(seed=1)) != split_rows(rows, SplitConfig(seed=2))


def test_every_row_is_assigned_once_with_expected_sizes() -> None:
    rows = make_rows({"a": 50, "b": 50})
    counts = Counter(split_rows(rows, SplitConfig()).values())
    assert counts == {Partition.TRAIN: 60, Partition.VALIDATION: 20, Partition.TEST: 20}


def test_stratified_split_keeps_class_proportions() -> None:
    rows = make_rows({"a": 80, "b": 20})
    assignments = split_rows(rows, SplitConfig())
    validation = Counter(r.expected for r in rows if assignments[r.id] is Partition.VALIDATION)
    assert validation == {"a": 16, "b": 4}


def test_rare_class_is_still_represented_in_validation() -> None:
    rows = make_rows({"a": 40, "rare": 3})
    assignments = split_rows(rows, SplitConfig())
    seen = {r.expected for r in rows if assignments[r.id] is Partition.VALIDATION}
    assert "rare" in seen


def test_no_test_partition_when_ratio_is_zero() -> None:
    rows = make_rows({"a": 20, "b": 20})
    assignments = split_rows(rows, SplitConfig(train=0.8, validation=0.2, test=0))
    assert Partition.TEST not in set(assignments.values())


def test_too_few_rows_for_the_split_is_an_error() -> None:
    with pytest.raises(DataError, match="too few"):
        split_rows(make_rows({"a": 1}), SplitConfig())


def test_summary_warns_about_small_validation_and_missing_test() -> None:
    rows = make_rows({"a": 20, "b": 20})
    config = SplitConfig(train=0.8, validation=0.2, test=0)
    summary = summarize(rows, split_rows(rows, config), config)
    assert any("validation partition has" in w for w in summary.warnings)
    assert any("no test partition" in w for w in summary.warnings)


def test_summary_has_no_warnings_for_a_healthy_split() -> None:
    rows = make_rows({"a": 100, "b": 100})
    config = SplitConfig()
    summary = summarize(rows, split_rows(rows, config), config)
    assert summary.warnings == []
    assert summary.counts[Partition.VALIDATION] == 40
