"""Checks that an optimizer-proposed revision only changed what it was allowed to change."""

from collections import Counter

from strop.prompts.models import PromptTemplate
from strop.prompts.template import fixed_spans


def validate_revision(parent: PromptTemplate, candidate: PromptTemplate) -> list[str]:
    """Return a list of rule violations; an empty list means the revision is acceptable."""
    if [m.role for m in parent.messages] != [m.role for m in candidate.messages]:
        return ["the message structure (roles and order) must not change"]
    problems = []
    if parent.optimize != candidate.optimize:
        problems.append("the set of editable roles must not change")
    for index, (old, new) in enumerate(zip(parent.messages, candidate.messages, strict=True), 1):
        label = f"message {index} ({old.role})"
        if not parent.is_editable(old):
            if old.content != new.content:
                problems.append(f"{label} is frozen but was modified")
            continue
        if not new.content.strip():
            problems.append(f"{label} is empty")
        if Counter(fixed_spans(old.content)) != Counter(fixed_spans(new.content)):
            problems.append(f"{label} changed or removed a <strop:fixed> span")
    if set(parent.variables) != set(candidate.variables):
        problems.append(
            f"template variables changed: expected {sorted(parent.variables)}, "
            f"got {sorted(candidate.variables)}"
        )
    return problems
