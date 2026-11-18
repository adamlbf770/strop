-- One evaluation = one prompt version run over one partition, with per-example results.
CREATE TABLE evaluations (
    id            INTEGER PRIMARY KEY,
    prompt_number INTEGER NOT NULL REFERENCES prompt_versions (number),
    split_id      INTEGER NOT NULL REFERENCES splits (id),
    partition     TEXT NOT NULL CHECK (partition IN ('train', 'validation', 'test')),
    kind          TEXT NOT NULL,
    run_id        INTEGER,
    metrics_json  TEXT NOT NULL,
    usage_json    TEXT NOT NULL,
    details_json  TEXT NOT NULL DEFAULT '{}',
    sample_count  INTEGER NOT NULL,
    error_count   INTEGER NOT NULL,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX evaluations_lookup ON evaluations (prompt_number, partition, kind);

CREATE TABLE example_results (
    evaluation_id     INTEGER NOT NULL REFERENCES evaluations (id) ON DELETE CASCADE,
    row_id            TEXT NOT NULL,
    expected          TEXT,
    raw_output        TEXT NOT NULL,
    value             TEXT,
    parsed_json       TEXT,
    error             TEXT,
    detail            TEXT,
    finish_reason     TEXT,
    latency_s         REAL NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    cost_usd          REAL,
    cached            INTEGER NOT NULL DEFAULT 0,
    scores_json       TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (evaluation_id, row_id)
);
