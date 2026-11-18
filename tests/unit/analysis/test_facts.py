from strop.analysis import build_facts, classify_outcome, select_evidence
from strop.analysis.evidence import clip, evidence_payload
from strop.analysis.facts import FailedExample
from strop.config.enums import OutputType
from strop.data.models import Row
from strop.evaluation.models import ExampleOutcome
from strop.metrics.types import ErrorCode


def outcome(
    row_id: str, expected: str | None, value: str | None, **kwargs: object
) -> ExampleOutcome:
    return ExampleOutcome(row_id, expected, value or "", value, **kwargs)  # type: ignore[arg-type]


def row(row_id: str) -> Row:
    return Row(row_id, 1, {"text": f"text of {row_id}"}, None, {}, "h")


def test_correct_outputs_have_no_error() -> None:
    assert classify_outcome(outcome("a", "x", "x"), OutputType.LABEL) is None
    assert classify_outcome(outcome("a", None, "x"), OutputType.LABEL) is None


def test_parse_level_errors_pass_through() -> None:
    failed = outcome("a", "x", None, error=ErrorCode.INVALID_FORMAT)
    assert classify_outcome(failed, OutputType.LABEL) == ErrorCode.INVALID_FORMAT


def test_wrong_values_are_labelled_by_output_type() -> None:
    assert classify_outcome(outcome("a", "x", "y"), OutputType.LABEL) == ErrorCode.WRONG_LABEL
    assert classify_outcome(outcome("a", "x", "y"), OutputType.TEXT) == ErrorCode.WRONG_VALUE


def test_json_with_missing_fields() -> None:
    expected = '{"a": 1, "b": 2}'
    missing = outcome("r", expected, '{"a": 1}', parsed={"a": 1})
    wrong = outcome("r", expected, '{"a": 1, "b": 3}', parsed={"a": 1, "b": 3})
    assert classify_outcome(missing, OutputType.JSON) == ErrorCode.MISSING_FIELD
    assert classify_outcome(wrong, OutputType.JSON) == ErrorCode.WRONG_VALUE


def test_rule_violations_are_reported_on_otherwise_correct_outputs() -> None:
    rule_failed = outcome("a", "x", "x", scores={"rule:short": 0.0, "rule:ok": 1.0})
    assert classify_outcome(rule_failed, OutputType.LABEL) == ErrorCode.RULE_VIOLATION


def test_facts_count_codes_and_confusions() -> None:
    outcomes = [
        outcome("a", "refund", "exchange"),
        outcome("b", "refund", "exchange"),
        outcome("c", "exchange", "refund"),
        outcome("d", "refund", "refund"),
        outcome("e", "refund", None, error=ErrorCode.INVALID_FORMAT),
    ]
    facts = build_facts(outcomes, [row(o.row_id) for o in outcomes], OutputType.LABEL)
    assert (facts.total, facts.failed) == (5, 4)
    assert facts.codes == {"wrong_label": 3, "invalid_format": 1}
    assert facts.confusions[0] == ("refund", "exchange", 2)
    assert facts.failures[0].inputs == {"text": "text of a"}


def test_provider_errors_show_their_detail() -> None:
    failed = outcome("a", "x", None, error=ErrorCode.PROVIDER_ERROR, detail="timeout")
    facts = build_facts([failed], [row("a")], OutputType.LABEL)
    assert facts.failures[0].got == "timeout"


def failure(row_id: str, code: str, expected: str = "x") -> FailedExample:
    return FailedExample(row_id, {"t": "v" * 50}, expected, "out", code)


def test_evidence_is_bounded_and_diverse() -> None:
    failures = [failure(f"big{i}", "wrong_label", "a") for i in range(20)]
    failures += [failure("rare1", "invalid_format"), failure("rare2", "wrong_label", "b")]
    chosen = select_evidence(failures, 5, seed=1)
    assert len(chosen) == 5
    kinds = {(f.code, f.expected) for f in chosen}
    assert len(kinds) == 3


def test_evidence_selection_is_reproducible() -> None:
    failures = [failure(f"r{i}", "wrong_label") for i in range(30)]
    assert select_evidence(failures, 6, seed=3) == select_evidence(failures, 6, seed=3)
    assert select_evidence(failures, 6, seed=3) != select_evidence(failures, 6, seed=4)
    assert select_evidence([], 5, seed=1) == []


def test_payload_clips_long_text() -> None:
    payload = evidence_payload([failure("a", "wrong_label")], max_chars=20)
    assert payload[0]["input"]["t"] == "v" * 17 + "..."
    assert payload[0]["problem"] == "wrong_label"
    assert clip("short", 20) == "short"
