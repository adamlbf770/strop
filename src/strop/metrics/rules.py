"""Rule checks: deterministic pass/fail tests on each output, reported as pass rates."""

import re
from collections.abc import Callable, Sequence

from strop.config.enums import RuleType
from strop.config.metrics import RuleSpec
from strop.metrics.types import ErrorCode, Example, Metric, MetricValue

Check = Callable[[str, RuleSpec], bool]

_CHECKS: dict[RuleType, Check] = {
    RuleType.MAX_CHARS: lambda text, spec: len(text) <= int(str(spec.value)),
    RuleType.MIN_CHARS: lambda text, spec: len(text) >= int(str(spec.value)),
    RuleType.MAX_WORDS: lambda text, spec: len(text.split()) <= int(str(spec.value)),
    RuleType.MIN_WORDS: lambda text, spec: len(text.split()) >= int(str(spec.value)),
    RuleType.MATCHES: lambda text, spec: re.search(str(spec.value), text) is not None,
    RuleType.NOT_MATCHES: lambda text, spec: re.search(str(spec.value), text) is None,
    RuleType.CONTAINS_ALL: lambda text, spec: all(
        term.casefold() in text.casefold() for term in _terms(spec)
    ),
    RuleType.CONTAINS_NONE: lambda text, spec: (
        not any(term.casefold() in text.casefold() for term in _terms(spec))
    ),
}


def _terms(spec: RuleSpec) -> list[str]:
    return spec.value if isinstance(spec.value, list) else [str(spec.value)]


class RuleMetric(Metric):
    """Pass rate of one rule, named ``rule:<name>``.

    Outputs lost to provider failures are not scored; every other output, including an
    empty one, is checked as written.
    """

    needs_expected = False

    def __init__(self, rule_name: str, spec: RuleSpec) -> None:
        self.name = f"rule:{rule_name}"
        self._spec = spec

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        scores = [s for e in examples if (s := self.per_example(e)) is not None]
        return self._result(sum(scores) / len(scores) if scores else None, len(scores))

    def per_example(self, example: Example) -> float | None:
        if example.error == ErrorCode.PROVIDER_ERROR:
            return None
        text = example.raw_output.strip()
        if not text:
            return 0.0
        return 1.0 if _CHECKS[self._spec.type](text, self._spec) else 0.0
