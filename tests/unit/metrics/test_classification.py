import pytest

from strop.metrics.basic import ExactMatch, ValidOutputRate
from strop.metrics.classification import (
    INVALID,
    Accuracy,
    Averaged,
    ClassScore,
    class_stats,
    confusion_matrix,
)
from strop.metrics.types import Example

LABELS = ("a", "b", "c")


def ex(expected: str, value: str | None) -> Example:
    return Example(row_id="r", inputs={}, expected=expected, raw_output=value or "", value=value)


# a: 3 correct, 1 predicted b. b: 2 correct, 1 predicted a. c: 1 invalid output.
EXAMPLES = [
    ex("a", "a"), ex("a", "a"), ex("a", "a"), ex("a", "b"),
    ex("b", "b"), ex("b", "b"), ex("b", "a"),
    ex("c", None),
]  # fmt: skip


def test_per_class_counts() -> None:
    stats = class_stats(EXAMPLES, LABELS)
    assert (stats["a"].tp, stats["a"].fp, stats["a"].fn) == (3, 1, 1)
    assert (stats["b"].tp, stats["b"].fp, stats["b"].fn) == (2, 1, 1)
    assert (stats["c"].tp, stats["c"].fp, stats["c"].fn) == (0, 0, 1)


def test_accuracy_counts_invalid_outputs_as_wrong() -> None:
    assert Accuracy(LABELS).compute(EXAMPLES).value == pytest.approx(5 / 8)


def test_macro_f1_includes_every_class_with_support() -> None:
    value = Averaged("macro", "f1", LABELS).compute(EXAMPLES).value
    assert value == pytest.approx((0.75 + 2 / 3 + 0.0) / 3)


def test_micro_scores_pool_the_counts() -> None:
    result = Averaged("micro", "precision", LABELS).compute(EXAMPLES)
    assert result.value == pytest.approx(5 / 7)
    assert Averaged("micro", "recall", LABELS).compute(EXAMPLES).value == pytest.approx(5 / 8)


def test_weighted_f1_uses_support() -> None:
    value = Averaged("weighted", "f1", LABELS).compute(EXAMPLES).value
    assert value == pytest.approx((4 * 0.75 + 3 * (2 / 3) + 1 * 0.0) / 8)


def test_class_score_and_unknown_class() -> None:
    assert ClassScore("recall", "a", LABELS).compute(EXAMPLES).value == pytest.approx(0.75)
    assert ClassScore("precision", "b", LABELS).compute(EXAMPLES).value == pytest.approx(2 / 3)
    only_a = [ex("a", "a")]
    assert ClassScore("recall", "c", LABELS).compute(only_a).value is None


def test_macro_skips_classes_without_support_or_predictions() -> None:
    value = Averaged("macro", "f1", LABELS).compute([ex("a", "a")]).value
    assert value == 1.0


def test_details_carry_per_class_scores_and_the_confusion_matrix() -> None:
    details = Accuracy(LABELS).compute(EXAMPLES).details
    assert details["per_class"]["a"]["support"] == 4
    assert details["confusion"]["a"] == {"a": 3, "b": 1, "c": 0, INVALID: 0}
    assert details["confusion"]["c"][INVALID] == 1


def test_confusion_matrix_ignores_unknown_expected_labels() -> None:
    matrix = confusion_matrix([ex("z", "a")], LABELS)
    assert sum(sum(row.values()) for row in matrix.values()) == 0


def test_empty_input_gives_no_value() -> None:
    assert Accuracy(LABELS).compute([]).value is None
    assert Averaged("macro", "f1", LABELS).compute([]).value is None


def test_labels_default_to_the_expected_values() -> None:
    value = Averaged("macro", "f1", ()).compute([ex("x", "x"), ex("y", "y")]).value
    assert value == 1.0


def test_exact_match_and_valid_output_rate() -> None:
    assert ExactMatch().compute(EXAMPLES).value == pytest.approx(5 / 8)
    assert ValidOutputRate().compute(EXAMPLES).value == pytest.approx(7 / 8)
    assert ValidOutputRate().per_example(ex("a", None)) == 0.0
    assert ExactMatch().per_example(ex("a", "a")) == 1.0
