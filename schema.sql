-- schema.sql
-- Run once to create the database structure.

-- One row per visitor/session. We don't do real login —
-- session_id is a random string stored in the browser cookie.
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- One row per form submission (a user fills 5 forms -> 5 rows).
-- answer_json stores whatever fields that form has, as a JSON string,
-- so we don't need a different table per form.
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    form_number INTEGER NOT NULL,
    answer_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- One row per generated roadmap (the LLM's output, saved as JSON text).
-- profile_id links it to the student_profiles table (new, schema-driven flow).
-- user_id is kept nullable for backward compatibility with the old
-- session-based /api/generate-roadmap stub.
CREATE TABLE IF NOT EXISTS roadmaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    profile_id INTEGER,
    generated_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (profile_id) REFERENCES student_profiles(id)
);

-- One row per full student profile (matches schema/student_profile.schema.json).
-- Stored as one JSON blob because the profile is deeply nested (past/present/
-- situation/goal/skills) -- splitting it into columns would mean redesigning
-- this table every time a form field changes.
-- user_id links it to the users table, so /api/generate-roadmap can look up
-- "this session's latest profile" without the frontend needing to track a
-- separate profile_id.
CREATE TABLE IF NOT EXISTS student_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    profile_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- One row per "activity" event (completing a form, generating a roadmap).
-- P4 uses this to draw the streak/heatmap graph.
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users(id)
);