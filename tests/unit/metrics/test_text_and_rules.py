import pytest

from strop.config.metrics import RuleSpec
from strop.errors import ConfigError
from strop.metrics.registry import MetricContext, build_metric
from strop.metrics.rules import RuleMetric
from strop.metrics.text import NumericClose, Similarity
from strop.metrics.types import ErrorCode, Example


def ex(raw: str, expected: str | None = None, value: str | None = None, **kw: str) -> Example:
    return Example(
        "r", {}, expected, raw, value if value is not None else raw, error=kw.get("error")
    )


def rule(type_: str, value: object) -> RuleMetric:
    return RuleMetric("r", RuleSpec.model_validate({"type": type_, "value": value}))


def test_similarity_scores_partial_matches() -> None:
    assert Similarity().per_example(ex("hello", "hello")) == 1.0
    assert 0 < (Similarity().per_example(ex("hello", "help")) or 0) < 1
    assert Similarity().per_example(Example("r", {}, "x", "", None)) == 0.0
    assert Similarity().per_example(ex("hello")) is None


def test_numeric_close_uses_an_absolute_tolerance() -> None:
    metric = NumericClose(0.5)
    assert metric.name == "numeric_close:0.5"
    assert metric.per_example(ex("10.4", "10")) == 1.0
    assert metric.per_example(ex("11", "10")) == 0.0
    assert metric.per_example(ex("1,000", "1000")) == 1.0
    assert metric.per_example(ex("abc", "10")) == 0.0
    assert metric.per_example(ex("10", "abc")) is None
    assert NumericClose().name == "numeric_close"


@pytest.mark.parametrize(
    ("type_", "value", "text", "passes"),
    [
        ("max_words", 3, "one two three", True),
        ("max_words", 3, "one two three four", False),
        ("min_words", 2, "one", False),
        ("max_chars", 5, "hello", True),
        ("min_chars", 6, "hello", False),
        ("matches", r"^\d+$", "123", True),
        ("matches", r"^\d+$", "12a", False),
        ("not_matches", r"(?i)guarantee", "We guarantee it", False),
        ("not_matches", r"(?i)guarantee", "No promises", True),
        ("contains_all", ["Refund", "days"], "refund within 30 days", True),
        ("contains_all", ["refund", "days"], "refund soon", False),
        ("contains_none", ["free", "win"], "a fine offer", True),
        ("contains_none", ["free", "win"], "FREE stuff", False),
    ],
)
def test_rule_checks(type_: str, value: object, text: str, passes: bool) -> None:
    assert rule(type_, value).per_example(ex(text)) == (1.0 if passes else 0.0)


def test_rule_metric_name_and_aggregation() -> None:
    metric = RuleMetric("short", RuleSpec(type="max_words", value=2))
    assert metric.name == "rule:short"
    result = metric.compute([ex("a b"), ex("a b c"), ex("x")])
    assert result.value == pytest.approx(2 / 3)
    assert result.samples == 3


def test_rules_skip_provider_failures_and_fail_empty_output() -> None:
    metric = rule("max_words", 5)
    failed = Example("r", {}, None, "", None, error=ErrorCode.PROVIDER_ERROR)
    assert metric.per_example(failed) is None
    empty = Example("r", {}, None, "  ", None, error=ErrorCode.EMPTY_OUTPUT)
    assert metric.per_example(empty) == 0.0
    assert metric.compute([failed, ex("fine")]).samples == 1


def test_registry_builds_rule_and_numeric_metrics() -> None:
    spec = RuleSpec(type="max_words", value=5)
    context = MetricContext(rules={"short": spec})
    assert build_metric("rule:short", context).name == "rule:short"
    assert build_metric("numeric_close:0.1", context).name == "numeric_close:0.1"
    assert build_metric("json_valid", context).name == "json_valid"
    with pytest.raises(ConfigError, match=r"rule 'long' is not defined.*defined: short"):
        build_metric("rule:long", context)
    with pytest.raises(ConfigError, match="numeric tolerance"):
        build_metric("numeric_close:abc", context)
