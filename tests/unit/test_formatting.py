from strop.formatting import fmt, shorten


def test_fmt_uses_fixed_decimals_and_marks_missing_values() -> None:
    assert fmt(0.5) == "0.5000"
    assert fmt(0.123456, 2) == "0.12"
    assert fmt(None) == "n/a"


def test_shorten_collapses_whitespace_and_marks_the_cut() -> None:
    assert shorten("a\n  b\tc") == "a b c"
    cut = shorten("word " * 30, 20)
    assert len(cut) == 20
    assert cut.endswith("...")
    assert shorten("short", 20) == "short"
