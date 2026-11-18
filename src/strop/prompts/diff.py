"""Line-level diffs between two prompt templates."""

import difflib
from dataclasses import dataclass, field
from typing import Literal

from strop.prompts.models import PromptMessage, PromptTemplate

Kind = Literal["hunk", "context", "add", "remove"]
Status = Literal["unchanged", "frozen", "changed", "added", "removed"]
CONTEXT_LINES = 3


@dataclass(frozen=True, slots=True)
class DiffLine:
    kind: Kind
    text: str

    @property
    def prefix(self) -> str:
        return {"hunk": "", "context": " ", "add": "+", "remove": "-"}[self.kind]


@dataclass(frozen=True)
class MessageDiff:
    index: int
    role: str
    status: Status
    lines: tuple[DiffLine, ...] = ()


@dataclass(frozen=True)
class DiffStats:
    added_lines: int
    removed_lines: int
    chars_before: int
    chars_after: int

    @property
    def approx_token_change(self) -> int:
        """A rough estimate (4 characters per token), not a tokenizer count."""
        return round((self.chars_after - self.chars_before) / 4)


@dataclass(frozen=True)
class PromptDiff:
    messages: tuple[MessageDiff, ...]
    stats: DiffStats = field(default_factory=lambda: DiffStats(0, 0, 0, 0))

    @property
    def changed(self) -> bool:
        return any(m.status in {"changed", "added", "removed"} for m in self.messages)

    def unified(self) -> str:
        """The diff as plain text with ``-``/``+`` prefixes and ``@@`` hunk headers."""
        out = []
        for message in self.messages:
            out.append(f"== message {message.index} ({message.role}): {message.status}")
            out.extend(f"{line.prefix}{line.text}" for line in message.lines)
        return "\n".join(out)

    def to_payload(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "messages": [
                {
                    "index": m.index,
                    "role": m.role,
                    "status": m.status,
                    "lines": [{"kind": ln.kind, "text": ln.text} for ln in m.lines],
                }
                for m in self.messages
            ],
            "stats": {
                "added_lines": self.stats.added_lines,
                "removed_lines": self.stats.removed_lines,
                "chars_before": self.stats.chars_before,
                "chars_after": self.stats.chars_after,
                "approx_token_change": self.stats.approx_token_change,
            },
            "unified": self.unified(),
        }


def diff_templates(old: PromptTemplate, new: PromptTemplate) -> PromptDiff:
    """Compare messages position by position."""
    messages = []
    for index in range(max(len(old.messages), len(new.messages))):
        before = old.messages[index] if index < len(old.messages) else None
        after = new.messages[index] if index < len(new.messages) else None
        messages.append(_diff_message(index + 1, before, after, new))
    lines = [ln for m in messages for ln in m.lines]
    stats = DiffStats(
        added_lines=sum(1 for ln in lines if ln.kind == "add"),
        removed_lines=sum(1 for ln in lines if ln.kind == "remove"),
        chars_before=sum(len(m.content) for m in old.messages),
        chars_after=sum(len(m.content) for m in new.messages),
    )
    return PromptDiff(tuple(messages), stats)


def _diff_message(
    index: int, before: PromptMessage | None, after: PromptMessage | None, new: PromptTemplate
) -> MessageDiff:
    if before is None and after is not None:
        return MessageDiff(index, after.role, "added", _all_lines("add", after.content))
    if after is None and before is not None:
        return MessageDiff(index, before.role, "removed", _all_lines("remove", before.content))
    assert before is not None
    assert after is not None
    if before.content == after.content and before.role == after.role:
        return MessageDiff(index, after.role, "unchanged" if new.is_editable(after) else "frozen")
    old_lines = before.content.splitlines()
    new_lines = after.content.splitlines()
    return MessageDiff(index, after.role, "changed", _hunks(old_lines, new_lines))


def _all_lines(kind: Kind, text: str) -> tuple[DiffLine, ...]:
    return tuple(DiffLine(kind, line) for line in text.splitlines())


def _hunks(old: list[str], new: list[str]) -> tuple[DiffLine, ...]:
    lines: list[DiffLine] = []
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for group in matcher.get_grouped_opcodes(CONTEXT_LINES):
        first, last = group[0], group[-1]
        lines.append(
            DiffLine(
                "hunk",
                f"@@ -{first[1] + 1},{last[2] - first[1]} +{first[3] + 1},{last[4] - first[3]} @@",
            )
        )
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                lines.extend(DiffLine("context", text) for text in old[i1:i2])
                continue
            lines.extend(DiffLine("remove", text) for text in old[i1:i2])
            lines.extend(DiffLine("add", text) for text in new[j1:j2])
    return tuple(lines)
