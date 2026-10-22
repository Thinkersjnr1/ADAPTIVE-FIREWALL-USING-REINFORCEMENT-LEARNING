"""SQLite persistence for everything the dashboard owns: users, settings, cases,
alert rules, saved filters, bookmarks, feedback, reports and more.

This is a separate database from src/db.py (which keeps decisions and notes), so
it can be added without touching that module."""
import json
import os
import sqlite3
import threading
import time

_path = "data/adaptiveshield_app.db"
_local = threading.local()
_settings = {}
_settings_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL,
  must_change INTEGER DEFAULT 0, totp_secret TEXT, totp_enabled INTEGER DEFAULT 0,
  totp_last INTEGER DEFAULT 0, created_at TEXT, last_login TEXT);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS admin_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS login_attempts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, ip TEXT, ts REAL, ok INTEGER);
CREATE INDEX IF NOT EXISTS ix_login_user ON login_attempts(username, ts);
CREATE INDEX IF NOT EXISTS ix_login_ip ON login_attempts(ip, ts);
CREATE TABLE IF NOT EXISTS cases(
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, description TEXT DEFAULT '',
  status TEXT DEFAULT 'open', severity TEXT DEFAULT 'MEDIUM', assignee TEXT,
  created_by TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS case_items(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  packet_id INTEGER, label TEXT, decision TEXT, added_by TEXT, added_at TEXT);
CREATE TABLE IF NOT EXISTS case_comments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  username TEXT, text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS alert_rules(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, metric TEXT NOT NULL,
  threshold REAL NOT NULL, window_s INTEGER DEFAULT 60, severity_min TEXT DEFAULT 'HIGH',
  webhook TEXT DEFAULT '', emails TEXT DEFAULT '', cooldown_s INTEGER DEFAULT 120,
  enabled INTEGER DEFAULT 1, created_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, rule_id INTEGER, rule_name TEXT, level TEXT,
  message TEXT, session_id TEXT, username TEXT, acknowledged INTEGER DEFAULT 0, ack_by TEXT);
CREATE TABLE IF NOT EXISTS saved_filters(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, view TEXT, name TEXT, params TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS bookmarks(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, packet_id INTEGER, label TEXT,
  note TEXT DEFAULT '', created_at TEXT);
CREATE TABLE IF NOT EXISTS prefs(username TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, packet_id INTEGER, model TEXT, label TEXT,
  original TEXT, final TEXT, reviewer TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS schedules(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, recipients TEXT, frequency TEXT DEFAULT 'weekly',
  hour INTEGER DEFAULT 8, enabled INTEGER DEFAULT 1, last_run TEXT, created_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS reports(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, title TEXT, kind TEXT,
  sent_to TEXT DEFAULT '', summary TEXT, html TEXT);
CREATE TABLE IF NOT EXISTS session_summaries(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, username TEXT, model TEXT, total INTEGER,
  score REAL, detection REAL, fpr REAL, tp INTEGER, fp INTEGER, fn INTEGER, summary TEXT);
CREATE TABLE IF NOT EXISTS api_keys(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, key_hash TEXT UNIQUE, prefix TEXT,
  created_by TEXT, created_at TEXT, last_used TEXT, enabled INTEGER DEFAULT 1);
"""

USER_COLUMNS = {"password_hash", "role", "must_change", "totp_secret", "totp_enabled",
                "totp_last", "last_login"}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def conn():
    c = getattr(_local, "conn", None)
    if c is None or getattr(_local, "path", None) != _path:
        c = sqlite3.connect(_path, timeout=15)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        _local.conn, _local.path = c, _path
    return c


def init(path=None, defaults=None):
    global _path
    if path:
        _path = path
    folder = os.path.dirname(_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    c = conn()
    c.executescript(SCHEMA)
    c.commit()
    with _settings_lock:
        _settings.clear()
        _settings.update(defaults or {})
        for r in query("SELECT key, value FROM settings"):
            try:
                _settings[r["key"]] = json.loads(r["value"])
            except ValueError:
                pass


def query(sql, args=()):
    return [dict(r) for r in conn().execute(sql, args).fetchall()]


def one(sql, args=()):
    r = conn().execute(sql, args).fetchone()
    return dict(r) if r else None


def execute(sql, args=()):
    """Run a write and return the new row id."""
    c = conn()
    cur = c.execute(sql, args)
    c.commit()
    return cur.lastrowid


def change(sql, args=()):
    """Run a write and return how many rows it touched."""
    c = conn()
    cur = c.execute(sql, args)
    c.commit()
    return cur.rowcount


# ---- settings (cached in memory: the playback loop reads these per packet) --------
def get_setting(key, default=None):
    with _settings_lock:
        return _settings.get(key, default)


def set_setting(key, value):
    execute("INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
    with _settings_lock:
        _settings[key] = value


# ---- users ---------------------------------------------------------------------------
def get_user(name):
    return one("SELECT * FROM users WHERE username=?", (name,)) if name else None


def list_users():
    return query("SELECT username, role, must_change, totp_enabled, created_at, last_login "
                 "FROM users ORDER BY username")


def count_users():
    return one("SELECT COUNT(*) AS n FROM users")["n"]


def create_user(name, password_hash, role, must_change=False):
    execute("INSERT INTO users(username, password_hash, role, must_change, created_at) VALUES(?,?,?,?,?)",
            (name, password_hash, role, 1 if must_change else 0, now()))


def update_user(name, **fields):
    bad = set(fields) - USER_COLUMNS
    if bad:
        raise ValueError(f"unknown user fields: {bad}")
    if not fields:
        return 0
    sets = ", ".join(f"{k}=?" for k in fields)
    return change(f"UPDATE users SET {sets} WHERE username=?", (*fields.values(), name))


def delete_user(name):
    return change("DELETE FROM users WHERE username=?", (name,))


# ---- admin activity ---------------------------------------------------------------------
def add_event(actor, action, detail=""):
    execute("INSERT INTO admin_events(ts, actor, action, detail) VALUES(?,?,?,?)",
            (now(), actor, action, detail))


def recent_events(limit=30):
    return query("SELECT ts, actor, action, detail FROM admin_events ORDER BY id DESC LIMIT ?", (limit,))


# ---- preferences -------------------------------------------------------------------------
def get_prefs(username):
    r = one("SELECT data FROM prefs WHERE username=?", (username,))
    try:
        return json.loads(r["data"]) if r else {}
    except ValueError:
        return {}


def set_prefs(username, data):
    execute("INSERT INTO prefs(username, data) VALUES(?, ?) "
            "ON CONFLICT(username) DO UPDATE SET data=excluded.data", (username, json.dumps(data)))
