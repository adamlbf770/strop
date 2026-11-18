from strop.prompts import PromptMessage, PromptTemplate
from strop.prompts.diff import diff_templates


def make(system: str, user: str = "Text: {{text}}") -> PromptTemplate:
    return PromptTemplate(
        (PromptMessage("system", system), PromptMessage("user", user)), frozenset({"system"})
    )


def test_identical_templates_have_no_changes() -> None:
    diff = diff_templates(make("a"), make("a"))
    assert not diff.changed
    assert [m.status for m in diff.messages] == ["unchanged", "frozen"]
    assert diff.stats.added_lines == diff.stats.removed_lines == 0


def test_changed_lines_are_marked_with_hunks() -> None:
    old = make("one\ntwo\nthree")
    new = make("one\nTWO\nthree\nfour")
    diff = diff_templates(old, new)
    system = diff.messages[0]
    assert system.status == "changed"
    assert [(ln.kind, ln.text) for ln in system.lines] == [
        ("hunk", "@@ -1,3 +1,4 @@"),
        ("context", "one"),
        ("remove", "two"),
        ("add", "TWO"),
        ("context", "three"),
        ("add", "four"),
    ]
    assert (diff.stats.added_lines, diff.stats.removed_lines) == (2, 1)


def test_lines_that_look_like_diff_headers_are_kept() -> None:
    diff = diff_templates(make("-- note\nkeep"), make("keep"))
    removed = [ln.text for ln in diff.messages[0].lines if ln.kind == "remove"]
    assert removed == ["-- note"]


def test_unified_text_uses_standard_prefixes() -> None:
    text = diff_templates(make("a\nb"), make("a\nc")).unified()
    assert "== message 1 (system): changed" in text
    assert "-b" in text
    assert "+c" in text
    assert "== message 2 (user): frozen" in text


def test_added_and_removed_messages() -> None:
    old = PromptTemplate((PromptMessage("user", "x"),), frozenset({"user"}))
    new = PromptTemplate(
        (PromptMessage("user", "x"), PromptMessage("assistant", "y")), frozenset({"user"})
    )
    assert diff_templates(old, new).messages[1].status == "added"
    assert diff_templates(new, old).messages[1].status == "removed"


def test_stats_and_payload() -> None:
    diff = diff_templates(make("a"), make("a" * 41))
    assert diff.stats.chars_after - diff.stats.chars_before == 40
    assert diff.stats.approx_token_change == 10
    payload = diff.to_payload()
    assert payload["changed"] is True
    assert "unified" in payload
