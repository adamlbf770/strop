-- Dataset versions (identified by file content) and the splits made from them.
CREATE TABLE datasets (
    id           INTEGER PRIMARY KEY,
    content_hash TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    row_count    INTEGER NOT NULL,
    skipped_rows INTEGER NOT NULL DEFAULT 0,
    skip_invalid INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (content_hash, skip_invalid)
);

CREATE TABLE splits (
    id          INTEGER PRIMARY KEY,
    dataset_id  INTEGER NOT NULL REFERENCES datasets (id),
    seed        INTEGER NOT NULL,
    strategy    TEXT NOT NULL CHECK (strategy IN ('stratified', 'random')),
    config_json TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE split_assignments (
    split_id  INTEGER NOT NULL REFERENCES splits (id) ON DELETE CASCADE,
    row_id    TEXT NOT NULL,
    partition TEXT NOT NULL CHECK (partition IN ('train', 'validation', 'test')),
    PRIMARY KEY (split_id, row_id)
);
