-- Optimization runs, their turns, and every candidate prompt that was proposed.
CREATE TABLE runs (
    id           INTEGER PRIMARY KEY,
    status       TEXT NOT NULL
                 CHECK (status IN ('running', 'paused', 'completed', 'failed', 'interrupted')),
    start_prompt INTEGER NOT NULL REFERENCES prompt_versions (number),
    settings_json TEXT NOT NULL,
    stale_turns  INTEGER NOT NULL DEFAULT 0,
    stop_reason  TEXT,
    model_calls  INTEGER NOT NULL DEFAULT 0,
    cost_usd     REAL,
    started_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at  TEXT
);

CREATE TABLE turns (
    run_id              INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    number              INTEGER NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('running', 'awaiting_approval', 'done')),
    best_before         INTEGER NOT NULL REFERENCES prompt_versions (number),
    best_after          INTEGER REFERENCES prompt_versions (number),
    train_evaluation_id INTEGER REFERENCES evaluations (id),
    analysis_json       TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, number)
);

CREATE TABLE candidates (
    id              INTEGER PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    turn            INTEGER NOT NULL,
    parent_number   INTEGER NOT NULL REFERENCES prompt_versions (number),
    version_number  INTEGER REFERENCES prompt_versions (number),
    change_summary  TEXT NOT NULL DEFAULT '',
    patterns_json   TEXT NOT NULL DEFAULT '[]',
    evidence_json   TEXT NOT NULL DEFAULT '[]',
    tradeoffs_json  TEXT NOT NULL DEFAULT '[]',
    assumptions_json TEXT NOT NULL DEFAULT '[]',
    raw_response    TEXT NOT NULL DEFAULT '',
    evaluation_id   INTEGER REFERENCES evaluations (id),
    passes          INTEGER,
    primary_delta   REAL,
    wins            INTEGER,
    losses          INTEGER,
    ties            INTEGER,
    status          TEXT NOT NULL
                    CHECK (status IN ('evaluated', 'accepted', 'rejected', 'pending', 'invalid')),
    reason          TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX candidates_by_run ON candidates (run_id, turn);
CREATE INDEX candidates_by_version ON candidates (version_number);
