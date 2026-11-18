"""A metric defined by the user's own Python file, run in a separate process."""

import asyncio
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from strop.errors import ConfigError
from strop.log import redact
from strop.metrics.types import Direction, ErrorCode, Example, Metric, MetricValue

RUNNER = Path(__file__).with_name("python_runner.py")
METRIC_NAME = "custom"
STDERR_TAIL = 800


class PythonMetric(Metric):
    """Scores outputs with ``score(output, row)`` from a local Python file.

    ``row`` is ``{"inputs": {...}, "expected": ..., "value": ...}``. The file is user-authored
    local code and runs in an isolated subprocess with a timeout. Dataset content is passed to
    it as data and is never executed.
    """

    name = METRIC_NAME
    needs_expected = False

    def __init__(self, path: Path, *, timeout_s: float, direction: Direction) -> None:
        self._path = path
        self._timeout = timeout_s
        self.direction = direction
        self._results: dict[tuple[str, str], dict[str, Any]] = {}

    async def prepare(self, examples: Sequence[Example]) -> None:
        pending = [e for e in examples if _scorable(e) and self._key(e) not in self._results]
        if not pending:
            return
        items = [
            {
                "output": e.raw_output,
                "row": {"inputs": dict(e.inputs), "expected": e.expected, "value": e.value},
            }
            for e in pending
        ]
        results = await asyncio.to_thread(self._run, items)
        self._results.update({self._key(e): r for e, r in zip(pending, results, strict=True)})

    def compute(self, examples: Sequence[Example]) -> MetricValue:
        scores = [s for e in examples if (s := self.per_example(e)) is not None]
        errors = sum(1 for e in examples if "error" in self._results.get(self._key(e), {}))
        value = sum(scores) / len(scores) if scores else None
        return self._result(value, len(scores), metric_errors=errors)

    def per_example(self, example: Example) -> float | None:
        result = self._results.get(self._key(example))
        if result is None or "score" not in result:
            return None
        return float(result["score"])

    def feedback(self, example: Example) -> str | None:
        result = self._results.get(self._key(example))
        if result is None:
            return None
        if "error" in result:
            return f"metric error: {result['error']}"
        return result.get("feedback")

    @staticmethod
    def _key(example: Example) -> tuple[str, str]:
        return example.row_id, example.raw_output

    def _run(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(RUNNER), str(self._path)],
                input=json.dumps(items),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ConfigError(
                f"{self._path.name} did not finish within {self._timeout:g} seconds"
            ) from None
        try:
            payload = json.loads(done.stdout)
        except ValueError:
            tail = redact(done.stderr.strip()[-STDERR_TAIL:])
            raise ConfigError(f"{self._path.name} failed to run: {tail or 'no output'}") from None
        if "fatal" in payload:
            raise ConfigError(f"cannot use {self._path.name}: {payload['fatal']}")
        results: list[dict[str, Any]] = payload["results"]
        return results


def _scorable(example: Example) -> bool:
    return example.error != ErrorCode.PROVIDER_ERROR
