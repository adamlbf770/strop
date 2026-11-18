import pytest

from strop.data.models import Partition
from strop.evaluation.models import EvaluationRecord, UsageSummary
from strop.metrics.suite import MetricReport
from strop.metrics.types import MetricValue
from strop.reporting.final_view import uncertainty_note, wilson_interval


def record(value: float | None, samples: int) -> EvaluationRecord:
    report = MetricReport("m", {"m": MetricValue("m", value, samples)})
    return EvaluationRecord(1, 1, 1, Partition.TEST, "test", report, UsageSummary(), samples, 0, "")


def test_wilson_interval_for_a_perfect_score_is_not_zero_width() -> None:
    low, high = wilson_interval(1.0, 30)
    assert low == pytest.approx(0.886, abs=0.005)
    assert high == 1.0


def test_wilson_interval_matches_a_known_value() -> None:
    low, high = wilson_interval(0.5, 100)
    assert (round(low, 3), round(high, 3)) == (0.404, 0.596)


def test_interval_narrows_as_the_sample_grows() -> None:
    small = wilson_interval(0.8, 30)
    large = wilson_interval(0.8, 3000)
    assert (large[1] - large[0]) < (small[1] - small[0]) / 5


def test_note_describes_the_interval() -> None:
    note = uncertainty_note(record(1.0, 30))
    assert note is not None
    assert "30 test examples" in note
    assert "between about 0.89 and 1.00" in note


@pytest.mark.parametrize("value", [None, -0.1, 1.5])
def test_no_note_when_the_score_is_not_a_rate(value: float | None) -> None:
    assert uncertainty_note(record(value, 30)) is None


def test_no_note_without_samples() -> None:
    assert uncertainty_note(record(0.5, 0)) is None
