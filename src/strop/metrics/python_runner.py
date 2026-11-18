"""Runs a user's metric file on a batch of examples.

This file is executed as a script in a separate Python process by ``python_metric``. It
imports nothing from Strop. The user's ``score(output, row)`` function may return a number
between 0 and 1, a bool, or a dict ``{"score": number, "feedback": "text"}``.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _normalize(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"score": 1.0 if value else 0.0}
    if isinstance(value, int | float):
        return _checked(float(value), None)
    if isinstance(value, dict) and isinstance(value.get("score"), int | float):
        feedback = value.get("feedback")
        return _checked(float(value["score"]), str(feedback) if feedback else None)
    return {"error": "score() must return a number or a dict with a 'score' key"}


def _checked(score: float, feedback: str | None) -> dict[str, Any]:
    if not 0.0 <= score <= 1.0:
        return {"error": f"score {score} is outside the range 0 to 1"}
    return {"score": score, "feedback": feedback}


def _load(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("strop_user_metric", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(module)
    if not callable(getattr(module, "score", None)):
        raise AttributeError("the metric file must define a function score(output, row)")
    return module.score


def main(path: str) -> int:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # user print() calls must not corrupt the JSON result
    try:
        score = _load(path)
    except Exception as exc:
        sys.stdout = real_stdout
        json.dump({"fatal": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 3
    results = []
    for item in json.load(sys.stdin):
        try:
            results.append(_normalize(score(item["output"], item["row"])))
        except Exception as exc:
            results.append({"error": f"{type(exc).__name__}: {exc}"})
    sys.stdout = real_stdout
    json.dump({"results": results}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
