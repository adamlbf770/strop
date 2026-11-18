"""Observed facts about a set of outcomes: what failed, how often, and how."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from strop.analysis.classify import classify_outcome, failed_rules
from strop.config.enums import OutputType
from strop.data.models import Row
from strop.evaluation.models import ExampleOutcome

TOP_CONFUSIONS = 5


@dataclass(frozen=True, slots=True)
class FailedExample:
    row_id: str
    inputs: Mapping[str, str]
    expected: str | None
    got: str
    code: str
    checks: tuple[str, ...] = ()
    feedback: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class AnalysisFacts:
    """Counts and examples derived only from stored results; nothing here is inferred."""

    total: int
    failures: tuple[FailedExample, ...]
    codes: Mapping[str, int]
    confusions: tuple[tuple[str, str, int], ...]

    @property
    def failed(self) -> int:
        return len(self.failures)


def build_facts(
    outcomes: Sequence[ExampleOutcome],
    rows: Sequence[Row],
    output_type: OutputType,
    primary: str | None = None,
) -> AnalysisFacts:
    """Classify every outcome and collect the failures with their inputs."""
    inputs = {row.id: row.inputs for row in rows}
    failures: list[FailedExample] = []
    for outcome in outcomes:
        code = classify_outcome(outcome, output_type, primary)
        if code is None:
            continue
        got = outcome.detail if outcome.error == "provider_error" else outcome.raw_output
        failures.append(
            FailedExample(
                outcome.row_id,
                inputs.get(outcome.row_id, {}),
                outcome.expected,
                got or "",
                code,
                failed_rules(outcome),
                outcome.feedback,
                outcome.scores.get(primary) if primary else None,
            )
        )
    codes = Counter(f.code for f in failures)
    pairs = Counter((f.expected or "", _answer(f)) for f in failures if f.code == "wrong_label")
    confusions = tuple((e, g, n) for (e, g), n in pairs.most_common(TOP_CONFUSIONS))
    return AnalysisFacts(len(outcomes), tuple(failures), dict(codes), confusions)


def _answer(failure: FailedExample) -> str:
    return failure.got.strip()
