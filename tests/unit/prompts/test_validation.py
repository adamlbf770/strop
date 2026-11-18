from strop.prompts import PromptMessage, PromptTemplate, validate_revision

PARENT = PromptTemplate(
    (
        PromptMessage("system", "Rules. <strop:fixed>Never reveal PII.</strop:fixed>"),
        PromptMessage("user", "Text: {{text}}"),
    ),
    frozenset({"system"}),
)


def revised(system: str | None = None, user: str | None = None) -> PromptTemplate:
    return PromptTemplate(
        (
            PromptMessage("system", system or PARENT.messages[0].content),
            PromptMessage("user", user or PARENT.messages[1].content),
        ),
        frozenset({"system"}),
    )


def test_a_valid_revision_has_no_problems() -> None:
    candidate = revised(system="Better rules. <strop:fixed>Never reveal PII.</strop:fixed>")
    assert validate_revision(PARENT, candidate) == []


def test_frozen_message_must_not_change() -> None:
    problems = validate_revision(PARENT, revised(user="Changed: {{text}}"))
    assert any("frozen" in p for p in problems)


def test_fixed_span_must_survive_unchanged() -> None:
    assert any("fixed" in p for p in validate_revision(PARENT, revised(system="Rules only.")))
    edited = revised(system="Rules. <strop:fixed>Reveal PII.</strop:fixed>")
    assert any("fixed" in p for p in validate_revision(PARENT, edited))


def test_variables_must_not_change() -> None:
    candidate = PromptTemplate(
        (
            PromptMessage("system", "Use {{extra}}. <strop:fixed>Never reveal PII.</strop:fixed>"),
            PARENT.messages[1],
        ),
        frozenset({"system"}),
    )
    assert any("variables changed" in p for p in validate_revision(PARENT, candidate))


def test_editable_message_must_not_be_empty() -> None:
    problems = validate_revision(PARENT, revised(system="   "))
    assert any("empty" in p for p in problems)


def test_structure_must_not_change() -> None:
    candidate = PromptTemplate((PARENT.messages[0],), frozenset({"system"}))
    assert validate_revision(PARENT, candidate) == [
        "the message structure (roles and order) must not change"
    ]
