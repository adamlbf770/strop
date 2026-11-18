import pytest

from strop.config.constraints import parse_constraints
from strop.config.metrics import Bound
from strop.errors import ConfigError


def test_parses_lower_and_upper_limits() -> None:
    parsed = parse_constraints(["valid_output_rate>=0.98", "latency_s <= 2"])
    assert parsed == {"valid_output_rate": Bound(min=0.98), "latency_s": Bound(max=2.0)}


def test_merges_both_sides_for_one_metric() -> None:
    assert parse_constraints(["x>=0.1", "x<=0.9"]) == {"x": Bound(min=0.1, max=0.9)}


def test_accepts_per_class_metric_names() -> None:
    assert "recall:refund" in parse_constraints(["recall:refund>=0.8"])


@pytest.mark.parametrize("text", ["recall", "recall>0.5", ">=0.5", "recall>=high", "recall>=1.2.3"])
def test_rejects_malformed_expressions(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_constraints([text])


def test_rejects_a_repeated_side() -> None:
    with pytest.raises(ConfigError, match="more than once"):
        parse_constraints(["x>=0.1", "x>=0.2"])
