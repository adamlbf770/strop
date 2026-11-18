"""Persistent cache of model responses."""

import hashlib
import json
import sqlite3

from strop.llm.types import LLMRequest, LLMResponse


def cache_key(request: LLMRequest) -> str:
    """Hash of the model, messages, and generation settings that determine a response."""
    payload = {
        "model": request.model,
        "messages": [[m.role, m.content] for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "params": request.params,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, key: str) -> LLMResponse | None:
        row = self._conn.execute(
            "SELECT response_json FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return LLMResponse.model_validate_json(row["response_json"]).model_copy(
            update={"cached": True, "latency_s": 0.0}
        )

    def put(self, key: str, response: LLMResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, response_json) VALUES (?, ?)",
            (key, response.model_dump_json()),
        )

    def clear(self) -> int:
        """Delete every cached response and return how many there were."""
        return self._conn.execute("DELETE FROM llm_cache").rowcount
