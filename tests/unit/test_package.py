import re

import strop


def test_version_is_a_release_string() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", strop.__version__)
