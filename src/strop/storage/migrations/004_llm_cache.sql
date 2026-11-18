-- Cached model responses, keyed by a hash of everything that determines the response.
CREATE TABLE llm_cache (
    key           TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
