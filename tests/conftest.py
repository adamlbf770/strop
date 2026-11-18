from typing import Any

import pytest


@pytest.fixture
def config_data() -> dict[str, Any]:
    """A minimal valid strop.yaml as a plain dictionary."""
    return {
        "project": {"name": "demo", "task_description": "Classify a message."},
        "dataset": {
            "path": "data.csv",
            "columns": {"message": "input", "category": "expected"},
        },
        "output": {"type": "label", "labels": ["billing", "refund"]},
        "models": {"target": {"model": "provider/model-a"}},
        "metrics": {"primary": "macro_f1"},
    }
