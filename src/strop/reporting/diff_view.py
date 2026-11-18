"""Colored terminal rendering of a prompt diff.

Without color (pipes, files, ``NO_COLOR``) the output is a plain unified diff.
"""

import difflib
import re

from rich.console import Console
from rich.text import Text

from strop.prompts.diff import DiffLine, MessageDiff, PromptDiff

_TOKEN = re.compile(r"\s+|\w+|[^\w\s]")
_STYLES = {
    "remove": ("red", "bold red reverse"),
    "add": ("green", "bold green reverse"),
}


def render_diff(diff: PromptDiff, console: Console) -> None:
    """Print ``diff`` with colored lines, word-level highlights, and a summary line."""
    for message in diff.messages:
        console.print(_heading(message), soft_wrap=True)
        for text in _message_lines(message.lines):
            console.print(text, soft_wrap=True)
    console.print(Text(_summary(diff), style="dim"), soft_wrap=True)


def _heading(message: MessageDiff) -> Text:
    style = "bold" if message.status in {"changed", "added", "removed"} else "dim"
    return Text(f"== message {message.index} ({message.role}): {message.status}", style=style)


def _message_lines(lines: tuple[DiffLine, ...]) -> list[Text]:
    out: list[Text] = []
    index = 0
    while index < len(lines):
        removed, added, index = _take_change_block(lines, index)
        if removed or added:
            out.extend(_change_block(removed, added))
            continue
        line = lines[index]
        out.append(_plain_line(line))
        index += 1
    return out


def _take_change_block(lines: tuple[DiffLine, ...], start: int) -> tuple[list[str], list[str], int]:
    index = start
    removed: list[str] = []
    added: list[str] = []
    while index < len(lines) and lines[index].kind == "remove":
        removed.append(lines[index].text)
        index += 1
    while index < len(lines) and lines[index].kind == "add":
        added.append(lines[index].text)
        index += 1
    return removed, added, index


def _change_block(removed: list[str], added: list[str]) -> list[Text]:
    old_texts = [Text("-")] * len(removed)
    new_texts = [Text("+")] * len(added)
    for position in range(max(len(removed), len(added))):
        before = removed[position] if position < len(removed) else None
        after = added[position] if position < len(added) else None
        if before is not None and after is not None:
            old_texts[position], new_texts[position] = _word_diff(before, after)
        elif before is not None:
            old_texts[position] = Text(f"-{before}", style=_STYLES["remove"][0])
        elif after is not None:
            new_texts[position] = Text(f"+{after}", style=_STYLES["add"][0])
    return [*old_texts, *new_texts]


def _word_diff(before: str, after: str) -> tuple[Text, Text]:
    old_tokens = _TOKEN.findall(before)
    new_tokens = _TOKEN.findall(after)
    old_text = Text("-", style=_STYLES["remove"][0])
    new_text = Text("+", style=_STYLES["add"][0])
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        changed = tag != "equal"
        old_text.append("".join(old_tokens[i1:i2]), _STYLES["remove"][1] if changed else None)
        new_text.append("".join(new_tokens[j1:j2]), _STYLES["add"][1] if changed else None)
    return old_text, new_text


def _plain_line(line: DiffLine) -> Text:
    if line.kind == "hunk":
        return Text(line.text, style="cyan")
    return Text(f"{line.prefix}{line.text}", style="dim")


def _summary(diff: PromptDiff) -> str:
    stats = diff.stats
    change = stats.approx_token_change
    return (
        f"Summary: +{stats.added_lines} line(s), -{stats.removed_lines} line(s), "
        f"prompt {stats.chars_before} -> {stats.chars_after} chars (~{change:+d} tokens)"
    )
