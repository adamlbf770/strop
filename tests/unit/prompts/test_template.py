import pytest

from strop.errors import ConfigError
from strop.prompts import PromptMessage, PromptTemplate, check_variables, render_messages
from strop.prompts.template import (
    find_variables,
    fixed_spans,
    has_balanced_markers,
    strip_markers,
    substitute,
)


def test_find_variables_is_ordered_and_distinct() -> None:
    text = "{{b}} then {{ a }} then {{b}}"
    assert find_variables(text) == ["b", "a"]


def test_find_variables_ignores_other_braces() -> None:
    assert find_variables('json: {"a": 1} and {single} and {{ 1bad }}') == []


def test_fixed_spans_and_markers() -> None:
    text = "keep <strop:fixed>this\nexactly</strop:fixed> and <strop:fixed>that</strop:fixed>"
    assert fixed_spans(text) == ["this\nexactly", "that"]
    assert strip_markers(text) == "keep this\nexactly and that"


@pytest.mark.parametrize(
    ("text", "balanced"),
    [
        ("<strop:fixed>a</strop:fixed>", True),
        ("no markers", True),
        ("<strop:fixed>a", False),
        ("a</strop:fixed>", False),
        ("</strop:fixed><strop:fixed>", False),
    ],
)
def test_marker_balance(text: str, balanced: bool) -> None:
    assert has_balanced_markers(text) is balanced


def test_substitute_does_not_expand_substituted_values() -> None:
    assert substitute("x={{a}}", {"a": "{{b}}", "b": "oops"}) == "x={{b}}"


def test_substitute_strips_markers_from_the_template_only() -> None:
    result = substitute(
        "<strop:fixed>rule</strop:fixed> {{a}}", {"a": "<strop:fixed>v</strop:fixed>"}
    )
    assert result == "rule <strop:fixed>v</strop:fixed>"


def test_render_messages_fills_each_message() -> None:
    template = PromptTemplate(
        (PromptMessage("system", "Classify."), PromptMessage("user", "Text: {{text}}")),
        frozenset({"system"}),
    )
    rendered = render_messages(template, {"text": "hello"})
    assert [(m.role, m.content) for m in rendered] == [
        ("system", "Classify."),
        ("user", "Text: hello"),
    ]


def test_render_reports_missing_values() -> None:
    template = PromptTemplate((PromptMessage("user", "{{a}} {{b}}"),), frozenset({"user"}))
    with pytest.raises(ConfigError, match="b"):
        render_messages(template, {"a": "1"})


def test_check_variables_names_unknown_ones_and_available_columns() -> None:
    check_variables(("a",), ("a", "b"))
    with pytest.raises(ConfigError, match=r"customer_msg.*input columns: customer_message"):
        check_variables(("customer_msg",), ("customer_message",))


def test_template_variables_span_all_messages() -> None:
    template = PromptTemplate(
        (PromptMessage("system", "{{rule}}"), PromptMessage("user", "{{text}} {{rule}}")),
        frozenset({"system"}),
    )
    assert template.variables == ("rule", "text")
