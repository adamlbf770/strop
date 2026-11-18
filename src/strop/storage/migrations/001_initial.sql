-- Project-level key/value metadata (project id, creation time, tool version).
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
