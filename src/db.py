"""
Lightweight SQLite persistence for the AdaptiveShield dashboard.

Replaces the original in-memory `logs = []` list, which lost all
history on every server restart and had no way to distinguish one
user's session from another's. This is intentionally minimal (two
tables) -- the full schema designed for the production dashboard
(adaptiveshield_schema.sql) is the target to grow into once auth,
rules, and online learning are built in later phases.
"""
import sqlite3
import os
import time
import threading

DB_PATH = "models/adaptiveshield.db"
_lock = threading.Lock()  # sqlite3 connections aren't thread-safe by default


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _lock:
        conn = get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id       TEXT NOT NULL,
                    username         TEXT,
                    timestamp        TEXT NOT NULL,
                    packet_id        INTEGER,
                    label            TEXT,
                    is_attack        INTEGER,
                    predicted_action TEXT,
                    reward           REAL,
                    confidence       REAL,
                    accuracy         REAL,
                    precision_pct    REAL,
                    recall_pct       REAL,
                    f1_pct           REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS log_access (
                    access_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    username    TEXT,
                    accessed_at TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    session_id  TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    packet_id  INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    username   TEXT,
                    note_text  TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    review_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    packet_id      INTEGER NOT NULL,
                    session_id     TEXT NOT NULL,
                    label          TEXT,
                    confidence     REAL,
                    default_action TEXT,
                    status         TEXT NOT NULL DEFAULT 'pending',
                    reviewer       TEXT,
                    review_action  TEXT,
                    reason         TEXT,
                    created_at     TEXT NOT NULL,
                    reviewed_at    TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()  # FIX: always close, even if a CREATE TABLE somehow fails


def add_review_item(packet_id, session_id, label, confidence, default_action):
    with _lock:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO review_queue (packet_id, session_id, label, confidence, default_action, created_at) VALUES (?,?,?,?,?,?)",
                (packet_id, session_id, label, confidence, default_action, time.strftime('%Y-%m-%d %H:%M:%S'))
            )
            conn.commit()
        finally:
            conn.close()


def get_pending_reviews(session_id, limit=50):
    with _lock:
        conn = get_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM review_queue WHERE session_id = ? AND status = 'pending' ORDER BY review_id DESC LIMIT ?",
                (session_id, limit)
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


def resolve_review(review_id, reviewer, review_action, reason):
    with _lock:
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE review_queue SET status='resolved', reviewer=?, review_action=?, reason=?, reviewed_at=? WHERE review_id=?",
                (reviewer, review_action, reason, time.strftime('%Y-%m-%d %H:%M:%S'), review_id)
            )
            conn.commit()
        finally:
            conn.close()


def get_audit_log(label_filter=None, decision_filter=None, limit=200):
    with _lock:
        conn = get_connection()
        try:
            query = "SELECT * FROM decisions WHERE 1=1"
            params = []
            if label_filter:
                query += " AND label LIKE ?"
                params.append(f"%{label_filter}%")
            if decision_filter:
                query += " AND predicted_action = ?"
                params.append(decision_filter)
            query += " ORDER BY decision_id DESC LIMIT ?"
            params.append(limit)
            cur = conn.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


def add_note(packet_id, session_id, username, note_text):
    with _lock:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO notes (packet_id, session_id, username, note_text, created_at) VALUES (?,?,?,?,?)",
                (packet_id, session_id, username, note_text, time.strftime('%Y-%m-%d %H:%M:%S'))
            )
            conn.commit()
        finally:
            # FIX: this used to only run on success. A failed INSERT (e.g.
            # session_id was None, violating NOT NULL) left the connection
            # open and holding a lock on the database file -- which then
            # caused EVERY subsequent write (including the live packet
            # stream's log_decision calls) to fail with "database is
            # locked". Closing in `finally` guarantees this can't cascade.
            conn.close()


def get_notes(packet_id):
    with _lock:
        conn = get_connection()
        try:
            cur = conn.execute(
                "SELECT username, note_text, created_at FROM notes WHERE packet_id = ? ORDER BY note_id ASC",
                (packet_id,)
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


def log_decision(session_id, username, row):
    """`row` is the same dict shape already built in the playback loop --
    this persists it instead of only keeping it in an in-memory list
    that vanishes on restart."""
    with _lock:
        conn = get_connection()
        try:
            conn.execute("""
                INSERT INTO decisions (session_id, username, timestamp, packet_id, label,
                    is_attack, predicted_action, reward, confidence, accuracy,
                    precision_pct, recall_pct, f1_pct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (session_id, username, row['timestamp'], row['packet_id'], row['label'],
                  row['is_attack'], row['decision'], row['reward'], row['confidence'],
                  row['accuracy'], row['precision'], row['recall'], row['f1']))
            conn.commit()
        finally:
            conn.close()


def get_session_logs(session_id, limit=5000):
    with _lock:
        conn = get_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM decisions WHERE session_id = ? ORDER BY decision_id ASC LIMIT ?",
                (session_id, limit)
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


def log_access(username, action, session_id=None):
    with _lock:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO log_access (username, accessed_at, action, session_id) VALUES (?,?,?,?)",
                (username, time.strftime('%Y-%m-%d %H:%M:%S'), action, session_id)
            )
            conn.commit()
        finally:
            conn.close()