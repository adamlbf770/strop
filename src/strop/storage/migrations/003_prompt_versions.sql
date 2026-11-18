-- Immutable prompt versions. A revision never overwrites its parent.
CREATE TABLE prompt_versions (
    id             INTEGER PRIMARY KEY,
    number         INTEGER NOT NULL UNIQUE,
    parent_number  INTEGER REFERENCES prompt_versions (number),
    content_json   TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    change_summary TEXT NOT NULL DEFAULT '',
    reason         TEXT NOT NULL DEFAULT '',
    source         TEXT NOT NULL CHECK (source IN ('user', 'optimizer')),
    created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER prompt_versions_no_update BEFORE UPDATE ON prompt_versions
BEGIN
    SELECT RAISE(ABORT, 'prompt versions are immutable');
END;

CREATE TRIGGER prompt_versions_no_delete BEFORE DELETE ON prompt_versions
BEGIN
    SELECT RAISE(ABORT, 'prompt versions are immutable');
END;
