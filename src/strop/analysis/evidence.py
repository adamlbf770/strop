"""Choosing a bounded, diverse sample of failures to show a model."""

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from strop.analysis.facts import FailedExample


def select_evidence(
    failures: Sequence[FailedExample], limit: int, seed: int
) -> list[FailedExample]:
    """Up to ``limit`` failures, taken round-robin across (error, expected) groups.

    Large groups do not crowd out small ones, so rare failure kinds stay visible.
    """
    groups: dict[tuple[str, str], list[FailedExample]] = defaultdict(list)
    for failure in sorted(failures, key=lambda f: f.row_id):
        groups[(failure.code, failure.expected or "")].append(failure)
    rng = random.Random(seed)
    queues = [rng.sample(members, len(members)) for _, members in sorted(groups.items())]
    chosen: list[FailedExample] = []
    while queues and len(chosen) < limit:
        for queue in list(queues):
            chosen.append(queue.pop())
            if not queue:
                queues.remove(queue)
            if len(chosen) >= limit:
                break
    return chosen


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def evidence_payload(evidence: Sequence[FailedExample], max_chars: int) -> list[dict[str, Any]]:
    """JSON-ready examples with every text field truncated to ``max_chars``."""
    return [
        {
            "id": item.row_id,
            "input": {k: clip(v, max_chars) for k, v in item.inputs.items()},
            "expected": clip(item.expected, max_chars) if item.expected is not None else None,
            "model_output": clip(item.got, max_chars),
            "problem": item.code,
            **({"failed_checks": list(item.checks)} if item.checks else {}),
            **({"feedback": clip(item.feedback, max_chars)} if item.feedback else {}),
            **({"score": round(item.score, 3)} if item.score is not None else {}),
        }
        for item in evidence
    ]
