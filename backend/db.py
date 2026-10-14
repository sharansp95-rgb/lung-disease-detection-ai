"""
SQLite persistence. One file (app.db), three tables (users, tokens, scans).
Uses plain sqlite3 (stdlib) rather than an ORM -- schema is small and
stable enough not to need one, and it keeps the dependency list short.

Connections are opened per-call (check_same_thread=False not needed since
we don't share connections across requests) -- SQLite handles this fine at
our request volume; no connection pool needed.
"""

import sqlite3
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    email_verified INTEGER NOT NULL DEFAULT 0,
    google_id TEXT UNIQUE,
    display_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    token TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('verify', 'reset')),
    expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    modality TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    class_probabilities TEXT NOT NULL,
    heatmap_png_base64 TEXT NOT NULL,
    report TEXT,
    report_source TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA)
