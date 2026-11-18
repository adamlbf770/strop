from pathlib import Path

import pytest

from strop.errors import ConfigError
from strop.metrics.python_metric import PythonMetric
from strop.metrics.types import Direction, ErrorCode, Example


def make_metric(tmp_path: Path, source: str, timeout: float = 30.0) -> PythonMetric:
    path = tmp_path / "metric.py"
    path.write_text(source, encoding="utf-8")
    return PythonMetric(path, timeout_s=timeout, direction=Direction.MAXIMIZE)


def example(row_id: str, raw: str, expected: str | None = "x", **kwargs: object) -> Example:
    return Example(row_id, {"text": f"in-{row_id}"}, expected, raw, raw or None, **kwargs)  # type: ignore[arg-type]


async def test_scores_and_feedback_come_from_the_users_function(tmp_path: Path) -> None:
    metric = make_metric(
        tmp_path,
        "def score(output, row):\n"
        "    if output == row['expected']:\n"
        "        return 1.0\n"
        "    return {'score': 0.25, 'feedback': 'expected ' + row['expected']}\n",
    )
    examples = [example("a", "x"), example("b", "y")]
    await metric.prepare(examples)
    assert metric.per_example(examples[0]) == 1.0
    assert metric.per_example(examples[1]) == 0.25
    assert metric.feedback(examples[1]) == "expected x"
    assert metric.feedback(examples[0]) is None
    result = metric.compute(examples)
    assert result.value == pytest.approx(0.625)
    assert result.samples == 2


async def test_row_contains_inputs_and_value(tmp_path: Path) -> None:
    metric = make_metric(
        tmp_path,
        "def score(output, row):\n"
        "    return 1.0 if row['inputs']['text'] == 'in-a' and row['value'] == output else 0.0\n",
    )
    item = example("a", "x")
    await metric.prepare([item])
    assert metric.per_example(item) == 1.0


async def test_print_calls_do_not_break_the_result(tmp_path: Path) -> None:
    metric = make_metric(tmp_path, "def score(output, row):\n    print('debug')\n    return True\n")
    item = example("a", "x")
    await metric.prepare([item])
    assert metric.per_example(item) == 1.0


async def test_errors_in_one_example_do_not_stop_the_others(tmp_path: Path) -> None:
    metric = make_metric(
        tmp_path,
        "def score(output, row):\n"
        "    if output == 'boom':\n"
        "        raise ValueError('bad output')\n"
        "    return 0.5\n",
    )
    good, bad = example("a", "fine"), example("b", "boom")
    await metric.prepare([good, bad])
    assert metric.per_example(good) == 0.5
    assert metric.per_example(bad) is None
    assert "ValueError: bad output" in (metric.feedback(bad) or "")
    assert metric.compute([good, bad]).details["metric_errors"] == 1


@pytest.mark.parametrize(
    "returned",
    ["'text'", "None", "1.5", "-0.1", "{'feedback': 'no score'}"],
)
async def test_invalid_return_values_are_metric_errors(tmp_path: Path, returned: str) -> None:
    metric = make_metric(tmp_path, f"def score(output, row):\n    return {returned}\n")
    item = example("a", "x")
    await metric.prepare([item])
    assert metric.per_example(item) is None
    assert (metric.feedback(item) or "").startswith("metric error")


async def test_provider_failures_are_not_scored(tmp_path: Path) -> None:
    metric = make_metric(tmp_path, "def score(output, row):\n    return 1.0\n")
    failed = Example("a", {}, "x", "", None, error=ErrorCode.PROVIDER_ERROR)
    await metric.prepare([failed])
    assert metric.per_example(failed) is None
    assert metric.compute([failed]).value is None


async def test_a_file_without_score_is_a_configuration_error(tmp_path: Path) -> None:
    metric = make_metric(tmp_path, "x = 1\n")
    with pytest.raises(ConfigError, match="define a function score"):
        await metric.prepare([example("a", "x")])


async def test_a_file_that_cannot_be_imported_is_reported(tmp_path: Path) -> None:
    metric = make_metric(tmp_path, "import not_a_real_module\n")
    with pytest.raises(ConfigError, match="not_a_real_module"):
        await metric.prepare([example("a", "x")])


async def test_a_slow_metric_times_out(tmp_path: Path) -> None:
    metric = make_metric(tmp_path, "import time\ndef score(output, row):\n    time.sleep(30)\n", 1)
    with pytest.raises(ConfigError, match="did not finish within 1 seconds"):
        await metric.prepare([example("a", "x")])


async def test_results_are_reused_for_the_same_output(tmp_path: Path) -> None:
    counter = tmp_path / "calls.txt"
    metric = make_metric(
        tmp_path,
        f"def score(output, row):\n    open({str(counter)!r}, 'a').write('x')\n    return 1.0\n",
    )
    item = example("a", "x")
    await metric.prepare([item])
    await metric.prepare([item])
    assert counter.read_text(encoding="utf-8") == "x"


async def test_the_files_own_directory_is_importable(tmp_path: Path) -> None:
    (tmp_path / "helper.py").write_text("def double(x):\n    return x * 2\n", encoding="utf-8")
    metric = make_metric(
        tmp_path, "from helper import double\ndef score(output, row):\n    return double(0.25)\n"
    )
    item = example("a", "x")
    await metric.prepare([item])
    assert metric.per_example(item) == 0.5
