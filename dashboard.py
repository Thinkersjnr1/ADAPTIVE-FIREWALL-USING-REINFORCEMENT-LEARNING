import sys
import os
import re
import csv
import io
import json
import time
import sqlite3
import threading
from datetime import timedelta
from functools import wraps

import joblib
import psutil
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for, session
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src import config, db, store, security, evaluation
from src.preprocess import preprocess
from src.dqn_agent import DQNAgent
from src.registry import ModelRegistry

app = Flask(__name__)
# All of these come from src/config.py and can be overridden with environment
# variables (ADAPTIVESHIELD_*). Set ADAPTIVESHIELD_SECRET before any real deployment.
app.secret_key = config.SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",          # blocks cross-site form posts (CSRF) on the admin routes
    SESSION_COOKIE_SECURE=config.COOKIE_SECURE,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=config.SESSION_LIFETIME_HOURS),
)
# Empty ADAPTIVESHIELD_ORIGINS = same-origin only.
socketio = SocketIO(app, cors_allowed_origins=config.SOCKET_ORIGINS, async_mode='threading')

CACHE_DIR = config.CACHE_DIR

# Finds every .keras/.h5 model (plus Random Forest / XGBoost baselines) and caches
# each one's predictions and evaluation. Filled with data in init_engine().
registry = ModelRegistry(config.MODELS_DIR, lambda n: DQNAgent(state_size=n, action_size=2))

# ---------------------------------------------------------------------------
# Users, settings and admin events now live in SQLite (src/store.py), so they
# survive restarts. Passwords are salted hashes; seeded accounts must change
# their password at first sign-in.
# ---------------------------------------------------------------------------
VALID_ROLES = ('admin', 'analyst', 'viewer')
USERNAME_RE = re.compile(r'^[A-Za-z0-9_.-]{3,32}$')
_DUMMY_HASH = generate_password_hash("not-a-real-password")  # keeps login timing similar for unknown users


def review_threshold():
    """Confidence % below which a decision is held for human review (admin-tunable, persisted)."""
    return store.get_setting('review_threshold', 40)


def record_admin_event(actor, action, detail=""):
    try:
        store.add_event(actor, action, detail)
    except Exception:
        pass
    try:
        db.log_access(actor, f'admin:{action}')
    except Exception:
        pass


def bootstrap():
    """Create the databases and, on first run only, the seed accounts."""
    db.init_db()
    store.init(config.DB_PATH, config.DEFAULT_SETTINGS)
    if store.count_users() == 0:
        store.create_user("admin", generate_password_hash(config.ADMIN_PASSWORD), "admin", must_change=True)
        if config.SEED_DEMO_USERS:
            store.create_user("analyst", generate_password_hash("analyst123"), "analyst", must_change=True)
            store.create_user("viewer", generate_password_hash("viewer123"), "viewer", must_change=True)
        print("[AdaptiveShield] First run: created seed accounts. Each must change its password at first sign-in.")
    if config.SECRET_KEY.startswith("dev-secret"):
        print("[AdaptiveShield] WARNING: using the built-in development secret key. "
              "Set ADAPTIVESHIELD_SECRET before any real deployment.")


LOGIN_HTML = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AdaptiveShield &ndash; Sign in</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'IBM Plex Sans',sans-serif;color:#E8E6E1;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:
 radial-gradient(900px 500px at 10% -10%,rgba(232,163,61,.14),transparent 60%),
 radial-gradient(800px 600px at 100% 110%,rgba(124,147,184,.16),transparent 60%),
 radial-gradient(600px 400px at 90% -5%,rgba(161,139,214,.10),transparent 60%),#0D1014}
.box{width:360px;max-width:92vw;padding:36px 32px 30px;border-radius:20px;
 background:linear-gradient(180deg,rgba(35,42,51,.72),rgba(28,33,40,.72));backdrop-filter:blur(16px);
 border:1px solid rgba(255,255,255,.07);box-shadow:0 30px 80px -30px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,255,255,.05)}
.mark{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#E8A33D,#EBD9B4 45%,#5FBFA0);margin-bottom:22px}
h2{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:28px;letter-spacing:-.02em;margin-bottom:6px}
p.sub{color:#8B9099;font-size:13.5px;margin-bottom:24px}
label{display:block;font-size:12.5px;color:#8B9099;margin:14px 0 6px}
input{width:100%;padding:11px 13px;border-radius:10px;border:1px solid rgba(255,255,255,.08);background:rgba(13,16,20,.55);
 color:#E8E6E1;font-size:14px;font-family:inherit;transition:border-color .15s, box-shadow .15s}
input:focus{outline:none;border-color:#E8A33D;box-shadow:0 0 0 3px rgba(232,163,61,.15)}
button{width:100%;margin-top:22px;padding:12px;border:none;border-radius:10px;cursor:pointer;font-weight:600;font-size:14px;
 font-family:inherit;color:#1a1206;background:linear-gradient(135deg,#E8A33D,#EBD9B4);transition:filter .15s}
button:hover{filter:brightness(1.07)}
.err{color:#D96B5A;font-size:13px;background:rgba(217,107,90,.1);border:1px solid rgba(217,107,90,.25);
 padding:9px 12px;border-radius:9px;margin-bottom:6px}
</style></head><body>
<form class="box" method="POST">
<div class="mark"></div>
<h2>AdaptiveShield</h2>
<p class="sub">Sign in to open the dashboard.</p>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<label for="u">Username</label>
<input id="u" name="username" autocomplete="username" required autofocus>
<label for="p">Password</label>
<input id="p" name="password" type="password" autocomplete="current-password" required>
<button type="submit">Sign in</button>
</form></body></html>
"""


# Shared look for the account pages (password change, two-factor): reuse the sign-in page's styles.
_AUTH_HEAD = LOGIN_HTML.split('<form')[0].replace('AdaptiveShield &ndash; Sign in', '{{ page_title }}')
AUTH_HTML = _AUTH_HEAD + """<form class="box" method="POST">
<div class="mark"></div>
<h2>{{ heading }}</h2>
<p class="sub">{{ sub }}</p>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
{% if extra %}{{ extra|safe }}{% endif %}
{% for name, label, typ, ac in fields %}
<label for="{{ name }}">{{ label }}</label>
<input id="{{ name }}" name="{{ name }}" type="{{ typ }}" autocomplete="{{ ac }}" required{% if loop.first %} autofocus{% endif %}>
{% endfor %}
<button type="submit">{{ button }}</button>
{% for href, text in links %}<p style="margin-top:16px;font-size:13px"><a href="{{ href }}" style="color:#8B9099">{{ text }}</a></p>{% endfor %}
</form></body></html>
"""


def _client_ip():
    return request.remote_addr or ""


def _end_session():
    keep = {k: session[k] for k in ('pending_2fa', 'pending_2fa_at') if k in session}
    session.clear()
    session.update(keep)  # an unfinished two-factor prompt in another tab stays valid
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Session expired. Please sign in again.'}), 401
    return redirect(url_for('login'))


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = store.get_user(session.get('username'))
        if not user:
            return _end_session()
        idle = store.get_setting('idle_timeout_min', config.IDLE_TIMEOUT_MIN) or 0
        now = time.time()
        last = session.get('last_seen')
        if idle and last and now - last > idle * 60:
            return _end_session()
        session['last_seen'] = now
        session['role'] = user['role']  # role changes made by an admin apply immediately
        if user['must_change'] and request.endpoint not in ('account_password', 'logout'):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Change your password before continuing.'}), 403
            return redirect(url_for('account_password'))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get('role') != 'admin':
            return jsonify({'error': 'Admin access required.'}), 403
        return f(*args, **kwargs)
    return login_required(wrapper)


def current_role():
    """Live role lookup for Socket.IO handlers (their session is a copy taken at connect time)."""
    user = store.get_user(session.get('username'))
    return user['role'] if user else None


# ---- Shared engine state: computed ONCE for the whole server, read-only
# after setup. Expensive (data loading + batched inference), so it is
# never redone per browser connection. ----
engine = {
    'ready': False,
    'ready_data': None,
    'X_test': None, 'y_test': None, 'classes': None,
    'all_q_vals': None, 'total_test_size': 0,
}
engine_lock = threading.Lock()
engine_started = False

# ---- Per-session playback/control state, keyed by Socket.IO sid. Each
# session has its own independent playback position, pause/stop flags and
# logs, all running over the SAME shared read-only engine data above. ----
sessions = {}
sessions_lock = threading.Lock()


def current_sid(explicit=None):
    """Find the caller's playback session.

    A session id set inside a Socket.IO handler never reaches the browser cookie, so
    plain HTTP routes can't rely on session['sid']. Use the id the page sends if it is
    the caller's own (or the caller is an admin), otherwise the caller's newest session."""
    me, role = session.get('username'), session.get('role')
    with sessions_lock:
        if explicit and explicit in sessions:
            if sessions[explicit].get('username') == me or role == 'admin':
                return explicit
        for sid in reversed(list(sessions)):
            if sessions[sid].get('username') == me:
                return sid
    return None

CVSS_MAP = {
    'DDoS': ('CRITICAL', 9.8, '#dc2626'),
    'DoS Hulk': ('CRITICAL', 9.1, '#dc2626'),
    'DoS GoldenEye': ('HIGH', 8.6, '#ea580c'),
    'DoS Slowloris': ('HIGH', 7.5, '#ea580c'),
    'DoS Slowhttptest': ('HIGH', 7.5, '#ea580c'),
    'PortScan': ('MEDIUM', 5.3, '#d97706'),
    'FTP-Patator': ('HIGH', 8.1, '#ea580c'),
    'SSH-Patator': ('HIGH', 8.1, '#ea580c'),
    'Web Attack Brute Force': ('HIGH', 7.3, '#ea580c'),
    'Web Attack XSS': ('MEDIUM', 6.1, '#d97706'),
    'Web Attack Sql Injection': ('CRITICAL', 9.8, '#dc2626'),
    'Infiltration': ('CRITICAL', 9.5, '#dc2626'),
    'Bot': ('HIGH', 8.8, '#ea580c'),
    'Heartbleed': ('CRITICAL', 9.8, '#dc2626'),
    'BENIGN': ('NONE', 0.0, '#059669'),
}


def get_cvss(label):
    for k, v in CVSS_MAP.items():
        if k.lower() in label.lower():
            return v
    return ('MEDIUM', 5.0, '#d97706')


# MITRE ATT&CK technique mapping -- verified against attack.mitre.org.
# Public taxonomy/reference data, same category as the CVSS map above.
# Some categories (esp. a few UNSW-NB15 labels like "Analysis"/"Fuzzers")
# don't map cleanly 1:1 onto a single ATT&CK technique; those fall through
# to the generic default below rather than forcing an inaccurate mapping.
MITRE_MAP = {
    'ddos':               ('T1498', 'Network Denial of Service'),
    'dos hulk':           ('T1499', 'Endpoint Denial of Service'),
    'dos slowloris':      ('T1499', 'Endpoint Denial of Service'),
    'dos slowhttptest':   ('T1499', 'Endpoint Denial of Service'),
    'dos goldeneye':      ('T1499', 'Endpoint Denial of Service'),
    'dos ':               ('T1499', 'Endpoint Denial of Service'),
    'portscan':           ('T1046', 'Network Service Discovery'),
    'patator':            ('T1110', 'Brute Force'),
    'bruteforce':         ('T1110', 'Brute Force'),
    'brute force':        ('T1110', 'Brute Force'),
    'ssh-bruteforce':     ('T1110', 'Brute Force'),
    'ftp-bruteforce':     ('T1110', 'Brute Force'),
    'sql injection':      ('T1190', 'Exploit Public-Facing Application'),
    'xss':                ('T1190', 'Exploit Public-Facing Application'),
    'web attack':         ('T1190', 'Exploit Public-Facing Application'),
    'heartbleed':         ('T1190', 'Exploit Public-Facing Application'),
    'exploits':           ('T1210', 'Exploitation of Remote Services'),
    'backdoor':           ('T1071', 'Application Layer Protocol'),
    'bot':                ('T1071', 'Application Layer Protocol'),
    'infilt':             ('T1133', 'External Remote Services'),
    'reconnaissance':     ('T1595', 'Active Scanning'),
    'shellcode':          ('T1203', 'Exploitation for Client Execution'),
    'worms':              ('T1210', 'Exploitation of Remote Services'),
    'generic':            ('T1499', 'Endpoint Denial of Service'),
}


def get_mitre(label):
    for k, v in MITRE_MAP.items():
        if k.lower() in label.lower():
            return v
    return ('--', 'No direct ATT&CK mapping')


def save_cache(X_train, X_test, y_train, y_test, classes):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(f"{CACHE_DIR}/X_train.npy", X_train)
    np.save(f"{CACHE_DIR}/X_test.npy", X_test)
    np.save(f"{CACHE_DIR}/y_train.npy", y_train)
    np.save(f"{CACHE_DIR}/y_test.npy", y_test)
    np.save(f"{CACHE_DIR}/classes.npy", np.array(classes))
    print("[AdaptiveShield] Cache saved.")


def load_cache():
    files = ['X_train', 'X_test', 'y_train', 'y_test', 'classes']
    if all(os.path.exists(f"{CACHE_DIR}/{f}.npy") for f in files):
        X_train = np.load(f"{CACHE_DIR}/X_train.npy")
        X_test = np.load(f"{CACHE_DIR}/X_test.npy")
        y_train = np.load(f"{CACHE_DIR}/y_train.npy")
        y_test = np.load(f"{CACHE_DIR}/y_test.npy")
        classes = list(np.load(f"{CACHE_DIR}/classes.npy", allow_pickle=True))
        print(f"[AdaptiveShield] Cache loaded — {len(X_test):,} packets.")
        return X_train, X_test, y_train, y_test, classes
    return None


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AdaptiveShield</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
:root{
  /* original palette, kept */
  --base:#14181D; --surface:#1C2128; --surface-2:#232A33; --border:#2C3440;
  --signal-amber:#E8A33D; --signal-mint:#5FBFA0; --ledger-ink:#7C93B8; --alert:#D96B5A;
  --text:#E8E6E1; --text-muted:#8B9099;
  /* added depth + accents */
  --base-deep:#0D1014; --surface-3:#2B333E; --text-dim:#5F6672;
  --orchid:#A18BD6; --champagne:#EBD9B4; --teal-deep:#2F8F7B; --copper:#C9794A;
  --hair:rgba(255,255,255,.06); --hair-strong:rgba(255,255,255,.11);
  --r-lg:18px; --r-md:14px; --r-sm:9px;
  --serif:'Fraunces',Georgia,serif;
}
*{box-sizing:border-box; margin:0; padding:0;}
html{color-scheme:dark;}
body{
  color:var(--text); font-family:'IBM Plex Sans',sans-serif; font-size:14px;
  height:100vh; overflow:hidden; display:flex; gap:12px; padding:12px;
  background:
    radial-gradient(900px 520px at 6% -10%, rgba(232,163,61,.09), transparent 60%),
    radial-gradient(800px 600px at 100% 112%, rgba(124,147,184,.11), transparent 60%),
    radial-gradient(600px 420px at 92% -6%, rgba(161,139,214,.07), transparent 60%),
    var(--base-deep);
}
.mono{font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;}
::-webkit-scrollbar{width:10px; height:10px;}
::-webkit-scrollbar-thumb{background:var(--surface-3); border-radius:10px; border:3px solid transparent; background-clip:content-box;}
::-webkit-scrollbar-thumb:hover{background:#3a4452; background-clip:content-box; border:3px solid transparent;}
::-webkit-scrollbar-track{background:transparent;}
:focus-visible{outline:2px solid var(--signal-amber); outline-offset:2px;}
@media (prefers-reduced-motion:reduce){*{animation:none !important; transition:none !important;}}

/* ---------- Surfaces ---------- */
.card, .pulse-card, .metric, .ledger, .timeline-card, .scatter-card, .conf-card, .health-card,
.review-item, .feed-list, .inspector, .x-card{
  background:linear-gradient(180deg, rgba(35,42,51,.55), rgba(28,33,40,.55));
  border:1px solid var(--hair); border-radius:var(--r-md);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
}

/* ---------- Left rail ---------- */
.rail{
  width:232px; flex-shrink:0; display:flex; flex-direction:column; height:calc(100vh - 24px);
  background:rgba(28,33,40,.62); backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
  border:1px solid var(--hair); border-radius:var(--r-lg);
  box-shadow:0 24px 60px -28px rgba(0,0,0,.75), inset 0 1px 0 rgba(255,255,255,.05);
}
.rail-brand{padding:22px 20px 14px; display:flex; align-items:center; gap:12px;}
.rail-brand .mark{
  width:32px; height:32px; border-radius:10px; flex-shrink:0;
  background:linear-gradient(135deg,var(--signal-amber),var(--champagne) 45%,var(--signal-mint));
  box-shadow:0 6px 18px -6px rgba(232,163,61,.55);
}
.rail-brand .name{font-family:var(--serif); font-weight:600; font-size:17px; letter-spacing:-.015em;}
.rail-brand .sub{font-size:11.5px; color:var(--text-muted); margin-top:1px;}
.rail-nav{flex:1; padding:6px 12px 12px; overflow-y:auto;}
.rail-group{font-size:11.5px; color:var(--text-dim); padding:14px 10px 6px; letter-spacing:.01em;}
.rail-item{
  position:relative; display:flex; align-items:center; gap:11px; padding:9px 12px; border-radius:10px;
  color:var(--text-muted); font-size:13.5px; cursor:pointer; margin-bottom:2px;
  transition:background .15s, color .15s; user-select:none;
}
.rail-item .ic{width:17px; height:17px; flex-shrink:0; opacity:.85;}
.rail-item:hover{background:rgba(255,255,255,.04); color:var(--text);}
.rail-item.active{color:var(--text); background:linear-gradient(90deg, rgba(232,163,61,.14), rgba(232,163,61,.03));}
.rail-item.active .ic{color:var(--signal-amber); opacity:1;}
.rail-item.active::before{
  content:""; position:absolute; left:-12px; top:9px; bottom:9px; width:3px; border-radius:0 3px 3px 0;
  background:var(--signal-amber); box-shadow:0 0 12px rgba(232,163,61,.7);
}
.rail-foot{padding:14px 16px; border-top:1px solid var(--hair); display:flex; align-items:center; gap:11px;}
.avatar{
  width:34px; height:34px; border-radius:50%; flex-shrink:0; display:grid; place-items:center;
  font-weight:600; font-size:13px; color:#1a1206; background:linear-gradient(135deg,var(--orchid),var(--ledger-ink));
}
.who{font-weight:500; font-size:13px;}
.role-label{font-size:12px; color:var(--text-muted); text-transform:capitalize;}
.rail-foot a{margin-left:auto; color:var(--ledger-ink); text-decoration:none; font-size:12px;}
.rail-foot a:hover{color:var(--text);}

/* ---------- Main ---------- */
.main{flex:1; min-width:0; display:flex; flex-direction:column; height:calc(100vh - 24px);}
.topbar{padding:8px 10px 14px; display:flex; align-items:flex-end; justify-content:space-between; gap:20px; flex-wrap:wrap;}
.topbar h1{font-family:var(--serif); font-weight:600; font-size:28px; letter-spacing:-.02em; line-height:1.1;}
.topbar .page-sub{color:var(--text-muted); font-size:13px; margin-top:5px; max-width:60ch;}
.controls{display:flex; gap:8px; align-items:center; flex-wrap:wrap;}
.status-pill{
  display:flex; align-items:center; gap:8px; font-size:12.5px; padding:7px 12px; border-radius:999px;
  background:rgba(255,255,255,.04); border:1px solid var(--hair); color:var(--text-muted);
}
.status-pill b{color:var(--text); font-weight:500;}
.led{width:7px; height:7px; border-radius:50%; background:var(--signal-mint); box-shadow:0 0 0 0 rgba(95,191,160,.6); animation:pulse 2s infinite;}
.status-pill.paused .led{background:var(--signal-amber); animation:none;}
.status-pill.done .led{background:var(--ledger-ink); animation:none;}
.status-pill.stopped .led{background:var(--alert); animation:none;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(95,191,160,.55);}70%{box-shadow:0 0 0 7px rgba(95,191,160,0);}100%{box-shadow:0 0 0 0 rgba(95,191,160,0);}}
.progress{height:2px; margin:0 10px 14px; background:rgba(255,255,255,.05); border-radius:2px; overflow:hidden;}
.progress i{display:block; height:100%; width:0; background:linear-gradient(90deg,var(--signal-amber),var(--signal-mint)); transition:width .4s ease;}

.btn{
  background:rgba(255,255,255,.05); border:1px solid var(--hair-strong); color:var(--text);
  padding:7px 14px; border-radius:var(--r-sm); font-size:13px; cursor:pointer; text-decoration:none;
  font-family:'IBM Plex Sans',sans-serif; transition:border-color .15s, background .15s, filter .15s; display:inline-block;
}
.btn:hover{border-color:rgba(232,163,61,.55); background:rgba(255,255,255,.07);}
.btn.primary{background:linear-gradient(135deg,var(--signal-amber),var(--champagne)); color:#1a1206; border-color:transparent; font-weight:600;}
.btn.primary:hover{filter:brightness(1.07);}
.btn-approve{background:rgba(95,191,160,.16); color:var(--signal-mint); border-color:rgba(95,191,160,.35);}
.btn-approve:hover{background:rgba(95,191,160,.26); border-color:var(--signal-mint);}
.btn-override, .btn-danger{background:rgba(217,107,90,.14); color:var(--alert); border-color:rgba(217,107,90,.35);}
.btn-override:hover, .btn-danger:hover{background:rgba(217,107,90,.24); border-color:var(--alert);}
.btn.small{padding:5px 10px; font-size:12px;}
.btn:disabled{opacity:.4; cursor:not-allowed;}
input[type=range]{accent-color:var(--signal-amber); width:96px;}
.speed{display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-muted);}

input[type=text], input[type=password], select, textarea{
  background:rgba(13,16,20,.5); border:1px solid var(--hair-strong); color:var(--text);
  padding:8px 11px; border-radius:var(--r-sm); font-size:13px; font-family:'IBM Plex Sans',sans-serif;
  transition:border-color .15s, box-shadow .15s;
}
input[type=text]:focus, input[type=password]:focus, select:focus, textarea:focus{
  outline:none; border-color:var(--signal-amber); box-shadow:0 0 0 3px rgba(232,163,61,.14);
}

.views{flex:1; min-height:0; position:relative;}
.view{position:absolute; inset:0; overflow-y:auto; padding:2px 10px 24px; display:none;}
.view.active{display:block; animation:viewin .22s ease;}
@keyframes viewin{from{opacity:0;}to{opacity:1;}}
.sub-text{font-size:12px; color:var(--text-muted); line-height:1.5;}

/* ---------- Command Center ---------- */
.pulse-card{
  display:grid; grid-template-columns:minmax(220px,.8fr) 2fr; gap:8px; padding:8px; margin-bottom:16px; overflow:hidden;
  background:
    radial-gradient(500px 220px at 0% 0%, rgba(232,163,61,.10), transparent 70%),
    linear-gradient(180deg, rgba(35,42,51,.6), rgba(28,33,40,.6));
  border-radius:var(--r-lg);
}
.hero-decision{padding:22px 22px; display:flex; flex-direction:column; justify-content:center; gap:6px; min-width:0;}
.hero-decision .hd-note{font-size:12.5px; color:var(--text-muted);}
.hd-word{font-family:var(--serif); font-weight:600; font-size:56px; line-height:1; letter-spacing:-.03em; color:var(--text-dim); transition:color .25s;}
.hd-word.block{color:var(--alert);}
.hd-word.allow{color:var(--signal-mint);}
.hd-sub{font-size:14px; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
canvas#pulse{width:100%; height:176px; display:block;}

.metric-grid{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:16px;}
.metric{padding:16px 18px; position:relative; overflow:hidden;}
.metric::after{content:""; position:absolute; left:18px; right:18px; bottom:0; height:2px; border-radius:2px; background:var(--accent,var(--hair)); opacity:.7;}
.metric .label{font-size:12.5px; color:var(--text-muted); margin-bottom:8px;}
.metric .value{font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:500; font-variant-numeric:tabular-nums; letter-spacing:-.02em;}
.metric .value.mint{color:var(--signal-mint);} .metric .value.amber{color:var(--signal-amber);}
.metric .value.alert{color:var(--alert);} .metric .value.ink{color:var(--ledger-ink);}
.metric .cap{font-size:11.5px; color:var(--text-dim); margin-top:4px;}

.ledger-head{padding:14px 18px; border-bottom:1px solid var(--hair); font-size:13.5px; font-weight:600;}
.ledger-body{max-height:300px; overflow-y:auto;}
.ledger-row{display:flex; align-items:center; gap:12px; padding:9px 18px; border-bottom:1px solid var(--hair); font-size:12.5px;}
.ledger-row:last-child{border-bottom:none;}
.ledger-row .n{font-family:'IBM Plex Mono',monospace; color:var(--text-muted); width:56px;}
.ledger-row .lbl{flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.chip{font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; padding:2px 8px; border-radius:6px; display:inline-block;}
.chip.block{color:var(--alert); background:rgba(217,107,90,.13);}
.chip.allow{color:var(--signal-mint); background:rgba(95,191,160,.12);}
.chip.amber{color:var(--signal-amber); background:rgba(232,163,61,.13);}
.chip.ink{color:var(--ledger-ink); background:rgba(124,147,184,.14);}
.ledger-row .conf{font-family:'IBM Plex Mono',monospace; color:var(--text-muted); width:56px; text-align:right;}
.empty-line{padding:22px 18px; color:var(--text-muted); font-size:13px;}

/* ---------- Live Feed ---------- */
.timeline-card{padding:12px 16px; margin-bottom:14px;}
.timeline-card .t-head{font-size:12px; color:var(--text-muted); margin-bottom:8px;}
.timeline-strip{display:flex; align-items:flex-end; gap:2px; height:30px; overflow:hidden;}
.t-tick{width:4px; border-radius:2px; flex-shrink:0; cursor:pointer;}
.feed-toolbar{display:flex; gap:8px; margin-bottom:12px;}
.feed-toolbar input{flex:1;}
.feed-layout{display:flex; gap:14px; height:540px;}
.feed-list{flex:1.3; overflow-y:auto; min-width:0;}
.feed-table{width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:12.5px;}
.feed-table thead th{
  text-align:left; padding:10px 14px; border-bottom:1px solid var(--hair); color:var(--text-muted);
  font-weight:500; font-size:11.5px; position:sticky; top:0; background:#1d242c; z-index:1;
}
.feed-table tbody td{padding:8px 14px; border-bottom:1px solid var(--hair); cursor:pointer; white-space:nowrap;}
.feed-table tbody tr{transition:background .12s;}
.feed-table tbody tr:hover{background:rgba(255,255,255,.035);}
.feed-table tbody tr.selected{background:rgba(232,163,61,.08); box-shadow:inset 2px 0 0 var(--signal-amber);}
.feed-table .dec-block{color:var(--alert); font-weight:600;}
.feed-table .dec-allow{color:var(--signal-mint); font-weight:600;}
.sev-dot{display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:8px;}

.inspector{flex:1; overflow:hidden; display:flex; flex-direction:column; min-width:0;}
.inspector-head{padding:16px 18px 12px; border-bottom:1px solid var(--hair);}
.inspector-head h3{font-size:15px; font-weight:600; margin-bottom:4px;}
.inspector-head .meta{font-size:11.5px; color:var(--text-muted); font-family:'IBM Plex Mono',monospace;}
.inspector-qvals{display:flex; gap:10px; padding:12px 18px; border-bottom:1px solid var(--hair);}
.inspector-qvals .q{flex:1; background:rgba(255,255,255,.035); border-radius:10px; padding:9px 12px;}
.inspector-qvals .q .l{font-size:11px; color:var(--text-muted); margin-bottom:2px;}
.inspector-qvals .q .v{font-family:'IBM Plex Mono',monospace; font-size:15px;}
.inspector-fields{padding:4px 0;}
.field-row{display:flex; justify-content:space-between; gap:12px; padding:6px 18px; font-family:'IBM Plex Mono',monospace; font-size:11.5px; border-bottom:1px solid var(--hair);}
.field-row:hover{background:rgba(255,255,255,.03);}
.field-row .fname{color:var(--ledger-ink);}
.inspector-empty{flex:1; display:flex; align-items:center; justify-content:center; color:var(--text-muted); font-size:13px; text-align:center; padding:28px; line-height:1.6;}
.mitre-badge{display:inline-flex; align-items:center; margin-top:8px; font-size:11px; font-family:'IBM Plex Mono',monospace; color:var(--ledger-ink); background:rgba(124,147,184,.12); padding:3px 9px; border-radius:6px;}
.insp-export{font-size:11.5px; color:var(--text-muted); background:none; border:1px solid var(--hair-strong); border-radius:7px; padding:4px 10px; cursor:pointer; float:right; font-family:inherit;}
.insp-export:hover{border-color:var(--signal-amber); color:var(--text);}
.insp-tabs{display:flex; padding:0 8px; border-bottom:1px solid var(--hair);}
.insp-tab{padding:10px 12px; font-size:12.5px; color:var(--text-muted); cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-1px;}
.insp-tab:hover{color:var(--text);}
.insp-tab.active{color:var(--text); border-bottom-color:var(--signal-amber);}
.insp-pane{display:none; flex:1; overflow-y:auto; min-height:0;}
.insp-pane.active{display:block;}
.attr-row{padding:9px 18px; border-bottom:1px solid var(--hair);}
.attr-label{display:flex; justify-content:space-between; font-size:11.5px; font-family:'IBM Plex Mono',monospace; margin-bottom:5px;}
.attr-label .fname{color:var(--ledger-ink);}
.attr-bar-track{height:5px; background:rgba(255,255,255,.05); border-radius:3px; overflow:hidden;}
.attr-bar-fill{height:100%; border-radius:3px;}
.attr-bar-fill.pos{background:linear-gradient(90deg,var(--signal-amber),var(--champagne));}
.attr-bar-fill.neg{background:linear-gradient(90deg,var(--ledger-ink),var(--orchid));}
.similar-row{display:flex; justify-content:space-between; padding:10px 18px; border-bottom:1px solid var(--hair); font-size:12.5px; cursor:pointer;}
.similar-row:hover{background:rgba(255,255,255,.035);}
.sim-dist{font-family:'IBM Plex Mono',monospace; color:var(--text-muted); font-size:11px;}
.notes-pane{padding:14px 18px; display:flex; flex-direction:column; gap:10px;}
.notes-list{display:flex; flex-direction:column; gap:8px;}
.note-item{background:rgba(255,255,255,.04); border-radius:10px; padding:9px 12px; font-size:12.5px; line-height:1.5;}
.note-meta{color:var(--text-muted); font-size:11px; margin-bottom:3px; font-family:'IBM Plex Mono',monospace;}
.notes-pane textarea{width:100%; resize:vertical; min-height:56px;}
.scatter-card{padding:16px 18px; margin-top:14px;}
.s-head{font-size:13.5px; font-weight:600; margin-bottom:4px;}
.s-sub{font-size:11.5px; color:var(--text-muted); margin-bottom:10px;}

/* ---------- Confidence & Drift ---------- */
.conf-grid{display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px;}
.conf-card{padding:16px 18px;}
.conf-card h3{font-size:13.5px; font-weight:600; margin-bottom:10px;}
.drift-row{display:flex; justify-content:space-between; font-size:12.5px; font-family:'IBM Plex Mono',monospace; padding:8px 0; border-bottom:1px solid var(--hair);}
.drift-row:last-child{border-bottom:none;}
.drift-row .dz{color:var(--alert);}
.shadow-compare{display:flex; gap:12px; margin-top:6px;}
.shadow-box{flex:1; background:rgba(255,255,255,.035); border-radius:12px; padding:14px; text-align:center; border:1px solid transparent;}
.shadow-box .sl{font-size:11.5px; color:var(--text-muted); margin-bottom:5px;}
.shadow-box .sv{font-family:'IBM Plex Mono',monospace; font-size:17px; font-weight:600;}
.shadow-box.agree{border-color:rgba(95,191,160,.25);} .shadow-box.agree .sv{color:var(--signal-mint);}
.shadow-box.disagree{border-color:rgba(217,107,90,.3);} .shadow-box.disagree .sv{color:var(--alert);}
.narrative-box{background:rgba(232,163,61,.07); border-left:3px solid var(--signal-amber); border-radius:4px 10px 10px 4px; padding:11px 14px; font-size:13px; margin-top:12px; line-height:1.5;}

/* ---------- Model Health ---------- */
.health-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:14px;}
.health-card{padding:18px 20px;}
.health-card .hl{font-size:12.5px; color:var(--text-muted); margin-bottom:8px;}
.health-card .hv{font-family:'IBM Plex Mono',monospace; font-size:21px; font-variant-numeric:tabular-nums; word-break:break-all;}

/* ---------- Review Queue & Audit ---------- */
.review-item{padding:16px 18px; margin-bottom:12px;}
.ri-head{display:flex; justify-content:space-between; margin-bottom:6px; font-size:14px; font-weight:500;}
.ri-meta{font-size:11.5px; color:var(--text-muted); font-family:'IBM Plex Mono',monospace; margin-bottom:12px;}
.review-actions{display:flex; gap:8px; align-items:center;}
.review-actions input{flex:1;}
.audit-toolbar{display:flex; gap:8px; margin-bottom:12px;}
.audit-toolbar input{flex:1;}
.table-wrap{overflow-x:auto; border-radius:var(--r-md);}
.audit-table, .admin-table{width:100%; border-collapse:collapse; font-size:12.5px;}
.audit-table{font-family:'IBM Plex Mono',monospace;}
.audit-table th, .admin-table th{text-align:left; padding:10px 14px; border-bottom:1px solid var(--hair); color:var(--text-muted); font-weight:500; font-size:11.5px;}
.audit-table td, .admin-table td{padding:9px 14px; border-bottom:1px solid var(--hair);}
.admin-table tr:last-child td{border-bottom:none;}

/* ---------- Executive Summary ---------- */
.x-card{padding:20px 22px;}
.x-hero{display:grid; grid-template-columns:1fr auto; gap:30px; align-items:center; margin-bottom:16px; padding:30px 34px; border-radius:var(--r-lg);
  background:
    radial-gradient(600px 300px at 100% 0%, rgba(95,191,160,.10), transparent 65%),
    radial-gradient(500px 260px at 0% 100%, rgba(232,163,61,.09), transparent 65%),
    linear-gradient(180deg, rgba(35,42,51,.62), rgba(28,33,40,.62));}
.x-hero h2{font-family:var(--serif); font-weight:600; font-size:34px; line-height:1.15; letter-spacing:-.025em; max-width:20ch; margin-bottom:14px;}
.x-hero p{color:#c9c7c1; font-size:14.5px; line-height:1.65; max-width:62ch;}
.x-hero .basis{margin-top:14px; font-size:12px; color:var(--text-muted);}
.gauge{position:relative; width:200px; height:200px;}
.gauge canvas{width:200px; height:200px; display:block;}
.gauge-center{position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;}
.gauge-center b{font-family:var(--serif); font-size:44px; font-weight:600; letter-spacing:-.03em; line-height:1;}
.gauge-center span{font-size:13px; color:var(--signal-mint); margin-top:4px;}
.gauge-center small{font-size:11px; color:var(--text-muted); margin-top:2px;}
.kpi-row{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:16px;}
.kpi{padding:16px 20px;}
.kpi .kl{font-size:12.5px; color:var(--text-muted); margin-bottom:6px;}
.kpi .kv{font-family:'IBM Plex Mono',monospace; font-size:28px; font-weight:500; letter-spacing:-.02em;}
.kpi .kc{font-size:11.5px; color:var(--text-dim); margin-top:4px;}
.x-grid{display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:16px;}
.x-card h3{font-size:14px; font-weight:600; margin-bottom:4px;}
.x-card .sub-text{margin-bottom:14px;}
.stack-bar{display:flex; height:12px; border-radius:7px; overflow:hidden; background:rgba(255,255,255,.05); margin-bottom:14px; gap:2px;}
.stack-bar i{display:block; height:100%;}
.legend-row{display:flex; justify-content:space-between; align-items:center; padding:7px 0; font-size:12.5px; border-bottom:1px solid var(--hair);}
.legend-row:last-child{border-bottom:none;}
.legend-row .nm{display:flex; align-items:center; gap:9px;}
.legend-row .nm i{width:9px; height:9px; border-radius:3px; display:inline-block;}
.legend-row .vals{font-family:'IBM Plex Mono',monospace; color:var(--text-muted); font-size:12px;}
.bar-row{margin-bottom:12px;}
.bar-top{display:flex; justify-content:space-between; font-size:12.5px; margin-bottom:5px; gap:10px;}
.bar-top .bt-v{font-family:'IBM Plex Mono',monospace; color:var(--text-muted); font-size:11.5px; white-space:nowrap;}
.bar-track{height:6px; background:rgba(255,255,255,.06); border-radius:4px; overflow:hidden;}
.bar-fill{height:100%; border-radius:4px; transition:width .5s ease;}
.cmp-group{margin-bottom:16px;}
.cmp-group h4{font-size:12.5px; font-weight:500; color:var(--text-muted); margin-bottom:8px;}
.mitre-row{display:flex; gap:12px; align-items:baseline; padding:8px 0; border-bottom:1px solid var(--hair); font-size:12.5px;}
.mitre-row:last-child{border-bottom:none;}
.mitre-row .mid{font-family:'IBM Plex Mono',monospace; color:var(--ledger-ink); width:58px; flex-shrink:0;}
.mitre-row .mn{flex:1;}
.mitre-row .mc{font-family:'IBM Plex Mono',monospace; color:var(--text-muted);}
.exec-actions{display:flex; justify-content:flex-end; gap:8px;}
.empty-state{text-align:center; padding:80px 20px; color:var(--text-muted);}
.empty-state h3{font-family:var(--serif); font-size:22px; color:var(--text); margin-bottom:8px; font-weight:600;}
.empty-state p{max-width:44ch; margin:0 auto; line-height:1.6;}

/* ---------- Admin Controls ---------- */
.admin-stats{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:14px;}
.admin-two{display:grid; grid-template-columns:1fr 1.4fr; gap:14px; margin-bottom:14px;}
.admin-card{padding:18px 20px; margin-bottom:14px;}
.admin-two .admin-card{margin-bottom:0;}
.admin-card h3{font-size:14px; font-weight:600; margin-bottom:4px;}
.admin-card .sub-text{margin-bottom:14px;}
.thr-row{display:flex; align-items:center; gap:14px;}
.thr-row input[type=range]{flex:1; width:auto;}
.thr-val{font-family:'IBM Plex Mono',monospace; font-size:22px; min-width:64px; text-align:right; color:var(--signal-amber);}
.form-row{display:flex; gap:8px; flex-wrap:wrap; align-items:center;}
.form-row input, .form-row select{flex:1; min-width:140px;}
.prog-mini{width:120px; height:5px; background:rgba(255,255,255,.07); border-radius:3px; overflow:hidden; display:inline-block; vertical-align:middle; margin-right:8px;}
.prog-mini i{display:block; height:100%; background:linear-gradient(90deg,var(--signal-amber),var(--signal-mint));}
.act-cell{display:flex; gap:6px; justify-content:flex-end; flex-wrap:wrap;}
.event-row{display:flex; gap:12px; padding:8px 0; border-bottom:1px solid var(--hair); font-size:12.5px;}
.event-row:last-child{border-bottom:none;}
.event-row .et{font-family:'IBM Plex Mono',monospace; color:var(--text-dim); width:64px; flex-shrink:0;}
.event-row .ea{color:var(--text-muted);}

/* ---------- Overlays ---------- */
.loading-screen{position:fixed; inset:0; background:var(--base-deep); display:flex; flex-direction:column; align-items:center; justify-content:center; gap:16px; z-index:100;}
.loading-screen .spin{width:34px; height:34px; border-radius:50%; border:3px solid rgba(255,255,255,.08); border-top-color:var(--signal-amber); border-right-color:var(--signal-mint); animation:spin .9s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.loading-screen p{color:var(--text-muted); font-size:13px;}
#toasts{position:fixed; right:20px; bottom:20px; display:flex; flex-direction:column; gap:8px; z-index:200;}
.toast{background:#232a33; border:1px solid var(--hair-strong); border-left:3px solid var(--ledger-ink); padding:11px 16px; border-radius:10px; font-size:13px; max-width:340px; box-shadow:0 16px 40px -16px rgba(0,0,0,.7); animation:toastin .22s ease;}
.toast.ok{border-left-color:var(--signal-mint);} .toast.err{border-left-color:var(--alert);}
.toast.out{opacity:0; transition:opacity .25s;}
@keyframes toastin{from{opacity:0; transform:translateY(8px);}to{opacity:1; transform:none;}}
#modal{position:fixed; inset:0; background:rgba(8,10,13,.6); backdrop-filter:blur(4px); display:none; align-items:center; justify-content:center; z-index:150;}
#modal.open{display:flex;}
.modal-box{width:380px; max-width:92vw; padding:24px; border-radius:var(--r-lg); background:#1d242c; border:1px solid var(--hair-strong); box-shadow:0 30px 80px -20px rgba(0,0,0,.8);}
.modal-box h3{font-family:var(--serif); font-size:20px; font-weight:600; margin-bottom:8px;}
.modal-box p{color:var(--text-muted); font-size:13px; line-height:1.55; margin-bottom:16px;}
.modal-box input{width:100%; margin-bottom:16px; display:none;}
.modal-actions{display:flex; justify-content:flex-end; gap:8px;}

/* ---------- Responsive ---------- */
@media (max-width:1180px){
  .metric-grid, .kpi-row, .admin-stats{grid-template-columns:repeat(2,1fr);}
  .feed-layout{flex-direction:column; height:auto;}
  .feed-list{max-height:380px;} .inspector{min-height:420px;}
  .x-hero{grid-template-columns:1fr;} .pulse-card{grid-template-columns:1fr;}
  .x-grid, .admin-two, .conf-grid{grid-template-columns:1fr;} .health-grid{grid-template-columns:1fr 1fr;}
}
@media (max-width:860px){
  .rail{width:68px;} .rail-brand div:last-child, .rail-item .label, .rail-group, .rail-foot .who, .rail-foot .role-label, .rail-foot a{display:none;}
  .rail-brand{padding:20px 17px 12px;} .rail-foot{justify-content:center;} .rail-item{justify-content:center;}
  .rail-item.active::before{left:-12px;}
}
@media print{
  body{background:#fff; color:#111; height:auto; overflow:visible; display:block; padding:0;}
  .rail, .topbar .controls, .progress, .exec-actions, #toasts{display:none !important;}
  .main, .views{height:auto; position:static;} .view{position:static; overflow:visible; display:none !important; padding:0;}
  #view-exec{display:block !important;}
  .x-card, .kpi, .x-hero{background:#fff !important; border:1px solid #ddd; box-shadow:none; color:#111;}
  .x-hero p, .gauge-center span, .sub-text, .legend-row .vals, .bar-top .bt-v, .mitre-row .mc, .kpi .kl, .kpi .kc{color:#444 !important;}
  .topbar h1{color:#111;}
}
</style>
</head>
<body>

<div id="loading" class="loading-screen">
  <div class="spin"></div>
  <p id="loading-text">Loading the model and preparing the session replay&hellip;</p>
</div>

<aside class="rail">
  <div class="rail-brand">
    <div class="mark"></div>
    <div><div class="name">AdaptiveShield</div><div class="sub">Session replay</div></div>
  </div>
  <nav class="rail-nav" id="nav">
    <div class="rail-group">Monitor</div>
    <div class="rail-item active" data-view="command" tabindex="0"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7.5" height="7.5" rx="2"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="2"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="2"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="2"/></svg><span class="label">Command Center</span></div>
    <div class="rail-item" data-view="feed" tabindex="0"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16M4 12h16M4 18h10"/></svg><span class="label">Live Feed</span></div>
    <div class="rail-item" data-view="confidence" tabindex="0"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 3 4-8 5 6"/></svg><span class="label">Confidence &amp; Drift</span></div>
    <div class="rail-group">Respond</div>
    <div class="rail-item" data-view="review" tabindex="0"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.5"/></svg><span class="label">Review Queue</span></div>
    <div class="rail-item" data-view="audit" tabindex="0"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h8l4 4v14H7z"/><path d="M15 3v4h4M10 12h6M10 16h6"/></svg><span class="label">Audit Log</span></div>
    <div class="rail-group">Oversight</div>
    <div class="rail-item" data-view="exec" tabindex="0"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg><span class="label">Executive Summary</span></div>
    <div class="rail-item" data-view="health" tabindex="0"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2.5-6 4 12 2.5-6H21"/></svg><span class="label">Model Health</span></div>
    {% if role == 'admin' %}
    <div class="rail-item" data-view="admin" tabindex="0"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></svg><span class="label">Admin Controls</span></div>
    {% endif %}
  </nav>
  <div class="rail-foot">
    <div class="avatar">{{ username[:1]|upper }}</div>
    <div><div class="who">{{ username }}</div><div class="role-label">{{ role }}</div></div>
    <a href="/account/password" title="Password and two-factor">Account</a>
    <a href="/logout">Sign out</a>
  </div>
</aside>

<main class="main">
  <header class="topbar">
    <div>
      <h1 id="view-title">Command Center</h1>
      <p class="page-sub" id="view-sub"></p>
    </div>
    <div class="controls">
      <div class="status-pill" id="status-pill"><span class="led"></span><span id="status-text">Starting</span><b class="mono" id="pkt-count">0</b></div>
      {% if role in ['admin','analyst'] %}
      <button class="btn" id="btn-pause">Pause</button>
      <button class="btn" id="btn-restart">Restart</button>
      <button class="btn" id="btn-stop">Stop</button>
      <label class="speed">Speed <input type="range" id="speed" min="1" max="10" value="5" aria-label="Replay speed"></label>
      {% else %}
      <span class="sub-text">Viewer access is read-only</span>
      {% endif %}
      <a href="/download-logs" class="btn primary">Export CSV</a>
    </div>
  </header>
  <div class="progress"><i id="progress-fill"></i></div>

  <div class="views">

  <!-- ===== Command Center ===== -->
  <section class="view active" id="view-command">
    <div class="pulse-card">
      <div class="hero-decision">
        <span class="hd-note">Latest decision</span>
        <span class="hd-word" id="last-decision">&mdash;</span>
        <span class="hd-sub" id="last-label">Waiting for traffic</span>
      </div>
      <canvas id="pulse"></canvas>
    </div>
    <div class="metric-grid">
      <div class="metric" style="--accent:var(--signal-mint)"><div class="label">Accuracy</div><div class="value mint" id="m-accuracy">&mdash;</div><div class="cap">Correct decisions so far</div></div>
      <div class="metric" style="--accent:var(--signal-amber)"><div class="label">False positive rate</div><div class="value amber" id="m-fpr">&mdash;</div><div class="cap">Normal traffic blocked by mistake</div></div>
      <div class="metric" style="--accent:var(--ledger-ink)"><div class="label">F1 score</div><div class="value ink" id="m-f1">&mdash;</div><div class="cap">Balance of precision and recall</div></div>
      <div class="metric" style="--accent:var(--orchid)"><div class="label">Blocked / allowed</div><div class="value" id="m-ba">&mdash;</div><div class="cap">Decisions enforced</div></div>
    </div>
    <div class="ledger">
      <div class="ledger-head">Recent activity</div>
      <div class="ledger-body" id="ledger-body"><div class="empty-line" id="ledger-empty">Decisions will appear here as the replay runs.</div></div>
    </div>
  </section>

  <!-- ===== Live Feed ===== -->
  <section class="view" id="view-feed">
    <div class="timeline-card">
      <div class="t-head">Recent attack timeline. Taller ticks are higher severity; click one to inspect that packet.</div>
      <div class="timeline-strip" id="timeline-strip"></div>
    </div>
    <div class="feed-toolbar">
      <input type="text" id="feed-search" placeholder="Filter by classification&hellip;">
      <select id="feed-filter-decision">
        <option value="">All decisions</option>
        <option value="BLOCK">Blocked only</option>
        <option value="ALLOW">Allowed only</option>
      </select>
    </div>
    <div class="feed-layout">
      <div class="feed-list">
        <table class="feed-table">
          <thead><tr><th>#</th><th>Time</th><th>Classification</th><th>Decision</th><th>Confidence</th><th>CVSS</th></tr></thead>
          <tbody id="feed-body"></tbody>
        </table>
      </div>
      <div class="inspector" id="inspector">
        <div class="inspector-empty" id="inspector-empty">Select a packet to see every field the model used for that decision, why it decided, and which flows look similar.</div>
        <div id="inspector-content" style="display:none; flex-direction:column; flex:1; overflow:hidden; min-height:0;">
          <div class="inspector-head">
            <button class="insp-export" id="btn-export-report">Export report</button>
            <h3 id="insp-label">&mdash;</h3>
            <div class="meta">Packet #<span id="insp-id">&mdash;</span> &middot; <span id="insp-decision">&mdash;</span></div>
            <div class="mitre-badge" id="insp-mitre">&mdash;</div>
          </div>
          <div class="inspector-qvals">
            <div class="q"><div class="l">Q(ALLOW)</div><div class="v" id="insp-qallow">&mdash;</div></div>
            <div class="q"><div class="l">Q(BLOCK)</div><div class="v" id="insp-qblock">&mdash;</div></div>
          </div>
          <div class="insp-tabs">
            <div class="insp-tab active" data-pane="fields">Fields</div>
            <div class="insp-tab" data-pane="attribution">Why</div>
            <div class="insp-tab" data-pane="similar">Similar</div>
            <div class="insp-tab" data-pane="notes">Notes</div>
          </div>
          <div class="insp-pane active" id="pane-fields"><div class="inspector-fields" id="insp-fields"></div></div>
          <div class="insp-pane" id="pane-attribution"><div id="insp-attribution"></div></div>
          <div class="insp-pane" id="pane-similar"><div id="insp-similar"></div></div>
          <div class="insp-pane" id="pane-notes">
            <div class="notes-pane">
              <div class="notes-list" id="notes-list"></div>
              {% if role in ['admin','analyst'] %}
              <textarea id="note-text" placeholder="Add an investigation note for this packet&hellip;"></textarea>
              <button class="btn" id="btn-save-note" style="align-self:flex-start;">Save note</button>
              {% endif %}
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="scatter-card">
      <div class="s-head">Feature-space scatter</div>
      <div class="s-sub">Flow duration (x) against flow bytes per second (y), both on a log scale. Red points were blocked, green were allowed.</div>
      <canvas id="scatter" style="width:100%; height:190px;"></canvas>
    </div>
  </section>

  <!-- ===== Confidence & Drift ===== -->
  <section class="view" id="view-confidence">
    <div class="conf-grid">
      <div class="conf-card">
        <h3>Confidence margin distribution</h3>
        <canvas id="conf-histogram" style="width:100%; height:150px;"></canvas>
      </div>
      <div class="conf-card">
        <h3>Drift score over time</h3>
        <canvas id="drift-timeline" style="width:100%; height:150px;"></canvas>
        <div class="s-sub" style="margin:6px 0 0;">Features more than 2.5&sigma; from the evaluation-set baseline, per packet.</div>
      </div>
    </div>
    <div class="conf-card" style="margin-bottom:14px;">
      <h3>Shadow comparison on the last packet</h3>
      <div class="s-sub">AdaptiveShield (DQN) next to a simple z-score rule, standing in for a legacy static-rule firewall.</div>
      <div class="shadow-compare">
        <div class="shadow-box" id="shadow-dqn"><div class="sl">AdaptiveShield</div><div class="sv">&mdash;</div></div>
        <div class="shadow-box" id="shadow-rule"><div class="sl">Rule-based (z-score)</div><div class="sv">&mdash;</div></div>
      </div>
      <div class="narrative-box" id="narrative-box">Waiting for traffic&hellip;</div>
    </div>
    <div class="conf-card">
      <h3>Most drifted features on the last packet</h3>
      <div id="drift-list"><div class="sub-text">Waiting for traffic&hellip;</div></div>
    </div>
  </section>

  <!-- ===== Review Queue ===== -->
  <section class="view" id="view-review">
    <p class="sub-text" style="margin-bottom:14px;">Decisions below <b id="rq-threshold">40</b>% confidence are held here instead of being enforced automatically. Approve to confirm the model's default action, or override it with a reason.</p>
    <div id="review-list"><div class="sub-text">Loading&hellip;</div></div>
  </section>

  <!-- ===== Audit Log ===== -->
  <section class="view" id="view-audit">
    <div class="audit-toolbar">
      <input type="text" id="audit-label-filter" placeholder="Filter by label&hellip;">
      <select id="audit-decision-filter">
        <option value="">All decisions</option>
        <option value="BLOCK">Blocked only</option>
        <option value="ALLOW">Allowed only</option>
      </select>
      <button class="btn" id="btn-audit-search">Search</button>
    </div>
    <div class="card table-wrap">
      <table class="audit-table">
        <thead><tr><th>#</th><th>Timestamp</th><th>Session</th><th>User</th><th>Label</th><th>Decision</th><th>Confidence</th></tr></thead>
        <tbody id="audit-body"></tbody>
      </table>
    </div>
  </section>

  <!-- ===== Model Health ===== -->
  <section class="view" id="view-health">
    <div class="health-grid">
      <div class="health-card"><div class="hl">Active model</div><div class="hv" id="health-path" style="font-size:14px;">&mdash;</div></div>
      <div class="health-card"><div class="hl">Parameters</div><div class="hv" id="health-params">&mdash;</div></div>
      <div class="health-card"><div class="hl">Self-check accuracy</div><div class="hv" id="health-accuracy" style="color:var(--signal-mint);">&mdash;</div></div>
    </div>
    <div class="health-grid">
      <div class="health-card"><div class="hl">Evaluation set size</div><div class="hv" id="health-testsize">&mdash;</div></div>
      <div class="health-card"><div class="hl">Training episodes (checkpoint)</div><div class="hv" id="health-episodes">&mdash;</div></div>
      <div class="health-card"><div class="hl">Training steps (checkpoint)</div><div class="hv" id="health-steps">&mdash;</div></div>
    </div>
  </section>

  <!-- ===== Executive Summary ===== -->
  <section class="view" id="view-exec">
    <div class="empty-state" id="exec-empty">
      <h3>Nothing to summarise yet</h3>
      <p>Once the replay has processed some traffic, this page turns the results into a readout you can share with leadership.</p>
    </div>
    <div id="exec-body" style="display:none;">
      <div class="x-hero">
        <div>
          <h2 id="ex-headline">&mdash;</h2>
          <p id="ex-body"></p>
          <div class="basis" id="ex-basis"></div>
        </div>
        <div class="gauge">
          <canvas id="gauge"></canvas>
          <div class="gauge-center"><b id="g-score">&mdash;</b><span id="g-rating">&mdash;</span><small>Protection score</small></div>
        </div>
      </div>
      <div class="kpi-row">
        <div class="x-card kpi"><div class="kl">Attacks stopped</div><div class="kv" id="k-stopped" style="color:var(--signal-mint)">&mdash;</div><div class="kc" id="k-stopped-c"></div></div>
        <div class="x-card kpi"><div class="kl">Attacks missed</div><div class="kv" id="k-missed" style="color:var(--alert)">&mdash;</div><div class="kc" id="k-missed-c"></div></div>
        <div class="x-card kpi"><div class="kl">False alarms</div><div class="kv" id="k-fp" style="color:var(--signal-amber)">&mdash;</div><div class="kc" id="k-fp-c"></div></div>
        <div class="x-card kpi"><div class="kl">Waiting for review</div><div class="kv" id="k-review" style="color:var(--ledger-ink)">&mdash;</div><div class="kc" id="k-review-c"></div></div>
      </div>
      <div class="x-grid">
        <div class="x-card">
          <h3>Threats by severity</h3>
          <div class="sub-text">How serious the attacks in this replay were, and how many were stopped.</div>
          <div class="stack-bar" id="sev-bar"></div>
          <div id="sev-legend"></div>
        </div>
        <div class="x-card">
          <h3>Most common attacks</h3>
          <div class="sub-text">Share of each attack type that was stopped.</div>
          <div id="top-attacks"></div>
        </div>
        <div class="x-card">
          <h3>Compared with a static rule</h3>
          <div class="sub-text" id="cmp-sub">The same traffic judged by a simple threshold rule, like a traditional firewall.</div>
          <div id="cmp-body"></div>
        </div>
        <div class="x-card">
          <h3>Attacker techniques observed</h3>
          <div class="sub-text">Mapped to MITRE ATT&amp;CK so your security team can act on them.</div>
          <div id="mitre-list"></div>
        </div>
      </div>
      <div class="exec-actions">
        <button class="btn" id="btn-exec-copy">Copy summary</button>
        <button class="btn primary" id="btn-exec-print">Save as PDF</button>
      </div>
    </div>
  </section>

  <!-- ===== Admin Controls ===== -->
  {% if role == 'admin' %}
  <section class="view" id="view-admin">
    <div class="admin-stats">
      <div class="x-card kpi"><div class="kl">Processor load</div><div class="kv" id="a-cpu">&mdash;</div></div>
      <div class="x-card kpi"><div class="kl">Memory in use</div><div class="kv" id="a-mem">&mdash;</div></div>
      <div class="x-card kpi"><div class="kl">Engine</div><div class="kv" id="a-engine" style="font-size:20px; padding-top:6px;">&mdash;</div><div class="kc" id="a-engine-c"></div></div>
      <div class="x-card kpi"><div class="kl">Open sessions</div><div class="kv" id="a-sessions">&mdash;</div></div>
    </div>

    <div class="admin-two">
      <div class="x-card admin-card">
        <h3>Review sensitivity</h3>
        <div class="sub-text">Decisions with confidence below this level are held for a person to review. A higher value sends more decisions to the queue.</div>
        <div class="thr-row">
          <input type="range" id="thr-range" min="5" max="90" value="40" aria-label="Review threshold">
          <div class="thr-val" id="thr-val">40%</div>
        </div>
        <div style="margin-top:14px;"><button class="btn primary" id="btn-thr-apply">Apply to new decisions</button></div>
      </div>
      <div class="x-card admin-card">
        <h3>Add a user</h3>
        <div class="sub-text">Admins manage the system, analysts can control the replay and annotate, viewers can only watch. Passwords need at least 8 characters.</div>
        <div class="form-row">
          <input type="text" id="nu-name" placeholder="Username" autocomplete="off">
          <input type="password" id="nu-pass" placeholder="Password" autocomplete="new-password">
          <select id="nu-role"><option value="viewer">Viewer</option><option value="analyst">Analyst</option><option value="admin">Admin</option></select>
          <button class="btn primary" id="btn-add-user">Add user</button>
        </div>
      </div>
    </div>

    <div class="x-card admin-card" style="padding:0;">
      <div style="padding:18px 20px 6px;"><h3>Live sessions</h3><div class="sub-text" style="margin-bottom:8px;">Everyone currently replaying traffic. Pause or stop a session remotely.</div></div>
      <div class="table-wrap"><table class="admin-table">
        <thead><tr><th>User</th><th>Progress</th><th>State</th><th>Speed</th><th></th></tr></thead>
        <tbody id="sess-body"></tbody>
      </table></div>
    </div>

    <div class="x-card admin-card" style="padding:0;">
      <div style="padding:18px 20px 6px;"><h3>User access</h3><div class="sub-text" style="margin-bottom:8px;">Role changes apply on the person's next page request. Your own role can't be changed here, so you can't lock yourself out.</div></div>
      <div class="table-wrap"><table class="admin-table">
        <thead><tr><th>User</th><th>Role</th><th>Open sessions</th><th></th></tr></thead>
        <tbody id="users-body"></tbody>
      </table></div>
    </div>

    <div class="admin-two" style="grid-template-columns:1fr 1fr;">
      <div class="x-card admin-card">
        <h3>Recent admin activity</h3>
        <div class="sub-text">The latest changes made from this page since the server started.</div>
        <div id="events-list"></div>
      </div>
      <div class="x-card admin-card">
        <h3>Records</h3>
        <div class="sub-text">Download the full decision audit trail for compliance or offline analysis.</div>
        <a class="btn" href="/api/admin/audit.csv">Download audit log (CSV)</a>
      </div>
    </div>
  </section>
  {% endif %}

  </div>
</main>

<div id="toasts" aria-live="polite"></div>
<div id="modal" role="dialog" aria-modal="true">
  <div class="modal-box">
    <h3 id="modal-title"></h3>
    <p id="modal-body"></p>
    <input type="password" id="modal-input">
    <div class="modal-actions">
      <button class="btn" id="modal-cancel">Cancel</button>
      <button class="btn primary" id="modal-ok">Confirm</button>
    </div>
  </div>
</div>

<script>
const ROLE = {{ role|tojson }};
const CAN_CONTROL = ROLE === 'admin' || ROLE === 'analyst';
const ME = {{ username|tojson }};
const socket = io();

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n) => Number(n || 0).toLocaleString();

function toast(msg, kind){
  const el = document.createElement('div');
  el.className = 'toast ' + (kind || '');
  el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(()=>{ el.classList.add('out'); setTimeout(()=>el.remove(), 260); }, 3400);
}

async function api(url, opts){
  const r = await fetch(url, opts);
  let d = {};
  try{ d = await r.json(); }catch(e){}
  if(!r.ok) throw new Error(d.error || ('Request failed (' + r.status + ')'));
  return d;
}
const send = (url, body, method) => api(url, {method: method || 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body || {})});

function askModal(opts){
  return new Promise(resolve=>{
    const m = $('#modal'), inp = $('#modal-input'), ok = $('#modal-ok'), cancel = $('#modal-cancel');
    $('#modal-title').textContent = opts.title;
    $('#modal-body').textContent = opts.body || '';
    inp.style.display = opts.input ? 'block' : 'none';
    inp.value = ''; inp.placeholder = opts.input || '';
    ok.textContent = opts.ok || 'Confirm';
    ok.className = 'btn ' + (opts.danger ? 'btn-danger' : 'primary');
    m.classList.add('open');
    (opts.input ? inp : ok).focus();
    const finish = (v)=>{ m.classList.remove('open'); document.removeEventListener('keydown', onKey); resolve(v); };
    const onKey = (e)=>{ if(e.key === 'Escape') finish(null); else if(e.key === 'Enter' && opts.input) ok.click(); };
    document.addEventListener('keydown', onKey);
    ok.onclick = ()=> finish(opts.input ? inp.value : true);
    cancel.onclick = ()=> finish(null);
  });
}

/* ---------- Page chrome ---------- */
const VIEWS = {
  command:['Command Center','Live decisions, accuracy and recent activity for this replay.'],
  feed:['Live Feed','Open any flow to see what the model saw, why it decided, and which traffic looks similar.'],
  confidence:['Confidence & Drift','How sure the model is, and how far live traffic has moved from normal.'],
  review:['Review Queue','Low-confidence decisions waiting for a person to confirm or override.'],
  audit:['Audit Log','Every decision recorded for this deployment, searchable by label and outcome.'],
  health:['Model Health','The model in service and how it performs on its evaluation set.'],
  exec:['Executive Summary','A plain-language readout of how well AdaptiveShield is protecting the network.'],
  admin:['Admin Controls','Manage live sessions, review sensitivity and who can access the dashboard.'],
};
let viewTimer = null;
let totalPackets = 0;
let paused = false;

function setStatus(kind, text){
  const p = $('#status-pill');
  p.className = 'status-pill ' + (kind === 'live' ? '' : kind);
  $('#status-text').textContent = text;
}

function showView(name){
  if(!VIEWS[name] || !$('#view-' + name)) name = 'command';
  $$('.rail-item').forEach(i => i.classList.toggle('active', i.dataset.view === name));
  $$('.view').forEach(v => v.classList.remove('active'));
  $('#view-' + name).classList.add('active');
  $('#view-title').textContent = VIEWS[name][0];
  $('#view-sub').textContent = VIEWS[name][1];
  clearInterval(viewTimer); viewTimer = null;
  if(name === 'health') loadHealth();
  else if(name === 'review') loadReviewQueue();
  else if(name === 'audit') loadAudit();
  else if(name === 'exec'){ loadExec(); viewTimer = setInterval(loadExec, 3000); }
  else if(name === 'admin'){ loadAdmin(); viewTimer = setInterval(loadAdmin, 3000); }
  requestAnimationFrame(()=>{ drawPulse(); drawScatter(); drawConfHistogram(); drawDriftTimeline(); });
}
$$('.rail-item').forEach(item=>{
  item.addEventListener('click', ()=> showView(item.dataset.view));
  item.addEventListener('keydown', e=>{ if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); showView(item.dataset.view); } });
});
$('#view-sub').textContent = VIEWS.command[1];

/* ---------- Pulse chart ---------- */
const pulseCanvas = $('#pulse');
const ctx = pulseCanvas.getContext('2d');
let pulseData = [];
const MAX_POINTS = 80;

function drawPulse(){
  const dpr = window.devicePixelRatio || 1;
  const w = pulseCanvas.offsetWidth * dpr, h = pulseCanvas.offsetHeight * dpr;
  if(!w || !h) return;
  pulseCanvas.width = w; pulseCanvas.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,.05)'; ctx.lineWidth = 1;
  for(let i = 1; i < 4; i++){ const y = h/4*i; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  if(pulseData.length < 2) return;
  const pts = pulseData.map((v, i) => [(i/(MAX_POINTS-1))*w, h/2 - v*(h*0.3)]);
  const trace = ()=>{
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for(let i = 1; i < pts.length; i++){
      const mx = (pts[i-1][0] + pts[i][0]) / 2, my = (pts[i-1][1] + pts[i][1]) / 2;
      ctx.quadraticCurveTo(pts[i-1][0], pts[i-1][1], mx, my);
    }
    ctx.lineTo(pts[pts.length-1][0], pts[pts.length-1][1]);
  };
  trace(); ctx.lineTo(pts[pts.length-1][0], h); ctx.lineTo(pts[0][0], h); ctx.closePath();
  const fill = ctx.createLinearGradient(0, 0, 0, h);
  fill.addColorStop(0, 'rgba(232,163,61,.28)'); fill.addColorStop(1, 'rgba(232,163,61,0)');
  ctx.fillStyle = fill; ctx.fill();
  trace();
  ctx.lineWidth = 2.2 * dpr; ctx.lineJoin = 'round';
  const stroke = ctx.createLinearGradient(0, 0, w, 0);
  stroke.addColorStop(0, '#E8A33D'); stroke.addColorStop(1, '#EBD9B4');
  ctx.strokeStyle = stroke; ctx.shadowColor = 'rgba(232,163,61,.55)'; ctx.shadowBlur = 12 * dpr; ctx.stroke();
  ctx.shadowBlur = 0;
  const last = pts[pts.length-1];
  ctx.beginPath(); ctx.arc(last[0] - 3*dpr, last[1], 4*dpr, 0, Math.PI*2);
  ctx.fillStyle = pulseData[pulseData.length-1] > 0 ? '#5FBFA0' : '#D96B5A'; ctx.fill();
}
window.addEventListener('resize', ()=>{ drawPulse(); drawScatter(); drawConfHistogram(); drawDriftTimeline(); });

function addLedgerRow(label, decision, confidence, n){
  const body = $('#ledger-body');
  const empty = $('#ledger-empty'); if(empty) empty.remove();
  const row = document.createElement('div');
  row.className = 'ledger-row';
  row.innerHTML = `<span class="n">#${n}</span><span class="lbl">${esc(label)}</span><span class="chip ${decision==='BLOCK'?'block':'allow'}">${decision}</span><span class="conf">${confidence}%</span>`;
  body.prepend(row);
  while(body.children.length > 12) body.removeChild(body.lastChild);
}

/* ---------- Live feed ---------- */
const packetCache = new Map();
let currentPacketId = null;

function addFeedRow(pkt){
  const body = $('#feed-body');
  const row = document.createElement('tr');
  row.dataset.pktId = pkt.packet_id;
  row.dataset.label = pkt.last_label;
  row.dataset.decision = pkt.last_action;
  const sevColor = pkt.cvss_info[2] || '#8B9099';
  row.innerHTML = `<td>${pkt.total}</td><td>${new Date().toLocaleTimeString()}</td>
    <td><span class="sev-dot" style="background:${sevColor}"></span>${esc(pkt.last_label)}</td>
    <td class="${pkt.last_action==='BLOCK'?'dec-block':'dec-allow'}">${pkt.last_action}</td>
    <td>${pkt.confidence}%</td><td>${esc(pkt.cvss_info[0])}</td>`;
  row.addEventListener('click', ()=> selectPacket(pkt.packet_id));
  body.prepend(row);
  packetCache.set(pkt.packet_id, pkt);
  while(body.children.length > 200){
    const last = body.lastChild;
    packetCache.delete(parseInt(last.dataset.pktId));
    body.removeChild(last);
  }
  applyFeedFilter();
  addTimelineTick(pkt);
  addScatterPoint(pkt);
}

function addTimelineTick(pkt){
  const strip = $('#timeline-strip');
  const tick = document.createElement('div');
  const isBlock = pkt.last_action === 'BLOCK';
  const color = pkt.cvss_info[2] || '#8B9099';
  tick.className = 't-tick';
  tick.style.height = isBlock ? (10 + (pkt.cvss_info[1]||0)*2) + 'px' : '6px';
  tick.style.background = isBlock ? color : 'var(--signal-mint)';
  tick.title = `#${pkt.packet_id} — ${pkt.last_label} (${pkt.last_action})`;
  tick.addEventListener('click', ()=> selectPacket(pkt.packet_id));
  strip.appendChild(tick);
  while(strip.children.length > 150) strip.removeChild(strip.firstChild);
}

/* Feature-space scatter (log scale: flow durations and byte rates are heavy-tailed) */
const scatterCanvas = $('#scatter');
const scatterCtx = scatterCanvas.getContext('2d');
const scatterPoints = [];
function addScatterPoint(pkt){
  const f = pkt.features || {};
  const x = f['Flow Duration'], y = f['Flow Bytes/s'];
  if(x === undefined || y === undefined) return;
  scatterPoints.push({x, y, block: pkt.last_action === 'BLOCK'});
  if(scatterPoints.length > 300) scatterPoints.shift();
  drawScatter();
}
function drawScatter(){
  const dpr = window.devicePixelRatio || 1;
  const w = scatterCanvas.offsetWidth * dpr, h = scatterCanvas.offsetHeight * dpr;
  if(!w || !h) return;
  scatterCanvas.width = w; scatterCanvas.height = h;
  scatterCtx.clearRect(0, 0, w, h);
  scatterCtx.strokeStyle = 'rgba(255,255,255,.05)';
  for(let i = 1; i < 4; i++){ const y = h/4*i; scatterCtx.beginPath(); scatterCtx.moveTo(0, y); scatterCtx.lineTo(w, y); scatterCtx.stroke(); }
  if(scatterPoints.length < 2) return;
  const lg = v => Math.log1p(Math.max(0, v));
  const xMax = Math.max(...scatterPoints.map(p => lg(p.x)), 1);
  const yMax = Math.max(...scatterPoints.map(p => lg(p.y)), 1);
  scatterPoints.forEach(p=>{
    const px = (lg(p.x)/xMax) * (w*0.94) + w*0.03;
    const py = h*0.94 - (lg(p.y)/yMax) * (h*0.88);
    scatterCtx.beginPath(); scatterCtx.arc(px, py, 3*dpr, 0, Math.PI*2);
    scatterCtx.fillStyle = p.block ? 'rgba(217,107,90,.75)' : 'rgba(95,191,160,.7)';
    scatterCtx.fill();
  });
}

function applyFeedFilter(){
  const text = $('#feed-search').value.toLowerCase();
  const dec = $('#feed-filter-decision').value;
  $$('#feed-body tr').forEach(row=>{
    const okText = !text || row.dataset.label.toLowerCase().includes(text);
    const okDec = !dec || row.dataset.decision === dec;
    row.style.display = (okText && okDec) ? '' : 'none';
  });
}
$('#feed-search').addEventListener('input', applyFeedFilter);
$('#feed-filter-decision').addEventListener('change', applyFeedFilter);

$$('.insp-tab').forEach(tab=>{
  tab.addEventListener('click', ()=>{
    $$('.insp-tab').forEach(t => t.classList.remove('active'));
    $$('.insp-pane').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    $('#pane-' + tab.dataset.pane).classList.add('active');
  });
});

function selectPacket(pktId){
  const pkt = packetCache.get(pktId);
  if(!pkt) return;
  currentPacketId = pktId;
  $$('.feed-table tbody tr').forEach(r => r.classList.remove('selected'));
  const row = document.querySelector(`.feed-table tbody tr[data-pkt-id="${pktId}"]`);
  if(row) row.classList.add('selected');
  $('#inspector-empty').style.display = 'none';
  $('#inspector-content').style.display = 'flex';
  $('#insp-label').textContent = pkt.last_label;
  $('#insp-id').textContent = pkt.packet_id;
  $('#insp-decision').textContent = pkt.last_action;
  $('#insp-qallow').textContent = (pkt.q_allow ?? '—');
  $('#insp-qblock').textContent = (pkt.q_block ?? '—');
  const mitre = pkt.mitre_info || ['--','No mapping'];
  $('#insp-mitre').textContent = mitre[0] === '--' ? 'No direct ATT&CK mapping' : `${mitre[0]} — ${mitre[1]}`;

  const fieldsEl = $('#insp-fields');
  fieldsEl.innerHTML = '';
  const features = pkt.features || {};
  const names = Object.keys(features);
  if(names.length === 0){
    fieldsEl.innerHTML = '<div class="field-row"><span class="fname">Flow statistics unavailable</span></div><div class="field-row" style="color:var(--text-muted)">Run `python -m src.preprocess` once to generate the scaler files this panel needs.</div>';
  } else {
    fieldsEl.innerHTML = names.map(n => `<div class="field-row"><span class="fname">${esc(n)}</span><span>${esc(features[n])}</span></div>`).join('');
  }
  loadAttribution(pktId); loadSimilar(pktId); loadNotes(pktId);
}

const pad = (t) => `<div class="empty-line">${t}</div>`;

function loadAttribution(pktId){
  const el = $('#insp-attribution');
  el.innerHTML = pad('Working out which features drove this decision&hellip;');
  api(`/api/attribution/${pktId}`).then(data=>{
    const maxAbs = Math.max(...data.attribution.map(a => Math.abs(a.attribution)), 0.0001);
    el.innerHTML = data.attribution.map(a=>{
      const pct = Math.min(100, Math.abs(a.attribution)/maxAbs*100);
      return `<div class="attr-row"><div class="attr-label"><span class="fname">${esc(a.feature)}</span><span>${a.attribution.toFixed(4)}</span></div>
        <div class="attr-bar-track"><div class="attr-bar-fill ${a.attribution >= 0 ? 'pos' : 'neg'}" style="width:${pct}%"></div></div></div>`;
    }).join('') + '<div class="empty-line" style="font-size:11.5px;">Amber features pushed toward this decision. Blue features pushed against it but were outweighed.</div>';
  }).catch(e=>{ el.innerHTML = pad(esc(e.message)); });
}

function loadSimilar(pktId){
  const el = $('#insp-similar');
  el.innerHTML = pad('Searching for similar flows&hellip;');
  api(`/api/similar/${pktId}`).then(data=>{
    el.innerHTML = data.similar.map(s =>
      `<div class="similar-row" data-pkt="${s.packet_id}"><span>#${s.packet_id} &middot; ${esc(s.label)}</span><span class="sim-dist">distance ${s.distance}</span></div>`
    ).join('') || pad('No comparable flows found.');
  }).catch(e=>{ el.innerHTML = pad(esc(e.message)); });
}
$('#insp-similar').addEventListener('click', e=>{
  const row = e.target.closest('.similar-row'); if(!row) return;
  const id = parseInt(row.dataset.pkt);
  if(packetCache.has(id)) selectPacket(id);
  else toast(`Packet #${id} is outside the recent window. Only the last 200 packets are kept in the list.`);
});

function loadNotes(pktId){
  const el = $('#notes-list');
  el.innerHTML = '<div class="sub-text">Loading&hellip;</div>';
  api(`/api/notes/${pktId}`).then(data=>{
    el.innerHTML = (data.notes || []).map(n =>
      `<div class="note-item"><div class="note-meta">${esc(n.username)} &middot; ${esc(n.created_at)}</div>${esc(n.note_text)}</div>`
    ).join('') || '<div class="sub-text">No notes on this packet yet.</div>';
  }).catch(()=>{ el.innerHTML = '<div class="sub-text">Notes are unavailable right now.</div>'; });
}
const saveNoteBtn = $('#btn-save-note');
if(saveNoteBtn){
  saveNoteBtn.addEventListener('click', ()=>{
    const text = $('#note-text').value.trim();
    if(!text || !currentPacketId) return;
    send(`/api/notes/${currentPacketId}`, {note_text: text, sid: socket.id}).then(()=>{
      $('#note-text').value = ''; loadNotes(currentPacketId); toast('Note saved', 'ok');
    }).catch(e => toast(e.message, 'err'));
  });
}

$('#btn-export-report').addEventListener('click', ()=>{
  if(!currentPacketId) return;
  const pkt = packetCache.get(currentPacketId); if(!pkt) return;
  const mitre = pkt.mitre_info || ['--','No mapping'];
  const lines = [
    'AdaptiveShield Investigation Report', '='.repeat(40),
    `Packet: #${pkt.packet_id}`, `Classification: ${pkt.last_label}`,
    `Decision: ${pkt.last_action}`, `Confidence: ${pkt.confidence}%`,
    `CVSS: ${pkt.cvss_info[0]} (${pkt.cvss_info[1]})`,
    `MITRE ATT&CK: ${mitre[0]} — ${mitre[1]}`,
    `Q(ALLOW): ${pkt.q_allow}    Q(BLOCK): ${pkt.q_block}`,
    '', 'Flow statistics:', '-'.repeat(40),
  ];
  Object.entries(pkt.features || {}).forEach(([k, v]) => lines.push(`${k}: ${v}`));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n')], {type:'text/plain'}));
  a.download = `adaptiveshield_packet_${pkt.packet_id}_report.txt`;
  a.click();
});

/* ---------- Confidence & drift ---------- */
const confCanvas = $('#conf-histogram'), confCtx = confCanvas.getContext('2d');
const confMargins = [];
function drawConfHistogram(){
  const dpr = window.devicePixelRatio || 1;
  const w = confCanvas.offsetWidth * dpr, h = confCanvas.offsetHeight * dpr;
  if(!w || !h) return;
  confCanvas.width = w; confCanvas.height = h;
  confCtx.clearRect(0, 0, w, h);
  if(confMargins.length < 2) return;
  const bins = new Array(10).fill(0);
  confMargins.forEach(m => { bins[Math.min(9, Math.floor(m/10))]++; });
  const maxBin = Math.max(...bins, 1), bw = w/bins.length;
  const g = confCtx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, '#EBD9B4'); g.addColorStop(1, '#E8A33D');
  confCtx.fillStyle = g;
  bins.forEach((count, i)=>{
    const bh = (count/maxBin)*(h*0.88);
    confCtx.beginPath(); confCtx.roundRect(i*bw + 3, h - bh, bw - 6, bh, 4*dpr); confCtx.fill();
  });
}
const driftCanvas = $('#drift-timeline'), driftCtx = driftCanvas.getContext('2d');
const driftScores = [];
function drawDriftTimeline(){
  const dpr = window.devicePixelRatio || 1;
  const w = driftCanvas.offsetWidth * dpr, h = driftCanvas.offsetHeight * dpr;
  if(!w || !h) return;
  driftCanvas.width = w; driftCanvas.height = h;
  driftCtx.clearRect(0, 0, w, h);
  if(driftScores.length < 2) return;
  const pts = driftScores.map((v, i) => [(i/(driftScores.length-1))*w, h*0.92 - (v/6)*(h*0.85)]);
  driftCtx.beginPath(); pts.forEach((p, i) => i ? driftCtx.lineTo(p[0], p[1]) : driftCtx.moveTo(p[0], p[1]));
  driftCtx.lineTo(w, h); driftCtx.lineTo(0, h); driftCtx.closePath();
  const f = driftCtx.createLinearGradient(0, 0, 0, h);
  f.addColorStop(0, 'rgba(161,139,214,.30)'); f.addColorStop(1, 'rgba(124,147,184,0)');
  driftCtx.fillStyle = f; driftCtx.fill();
  driftCtx.beginPath(); pts.forEach((p, i) => i ? driftCtx.lineTo(p[0], p[1]) : driftCtx.moveTo(p[0], p[1]));
  driftCtx.strokeStyle = '#A18BD6'; driftCtx.lineWidth = 2*dpr; driftCtx.lineJoin = 'round'; driftCtx.stroke();
}

function updateShadowAndDrift(d){
  confMargins.push(Math.abs((d.q_block ?? 0) - (d.q_allow ?? 0))); if(confMargins.length > 100) confMargins.shift();
  drawConfHistogram();
  driftScores.push(d.drift_score || 0); if(driftScores.length > 150) driftScores.shift();
  drawDriftTimeline();
  const dqn = $('#shadow-dqn'), rule = $('#shadow-rule');
  dqn.querySelector('.sv').textContent = d.last_action;
  rule.querySelector('.sv').textContent = d.rule_decision || '—';
  const agree = d.rule_decision === d.last_action;
  dqn.className = rule.className = 'shadow-box ' + (agree ? 'agree' : 'disagree');
  $('#narrative-box').textContent = d.narrative || '—';
  const top = d.top_drift || [];
  $('#drift-list').innerHTML = top.length
    ? top.map(t => `<div class="drift-row"><span>${esc(t.feature)}</span><span class="dz">z = ${t.zscore}</span></div>`).join('')
    : '<div class="sub-text">No significant drift on this packet.</div>';
}

/* ---------- Model health ---------- */
function loadHealth(){
  api('/api/health').then(h=>{
    $('#health-path').textContent = h.model_path || '—';
    $('#health-params').textContent = fmt(h.params);
    $('#health-accuracy').textContent = h.accuracy != null ? (h.accuracy*100).toFixed(2) + '%' : '—';
    $('#health-testsize').textContent = fmt(h.total_test);
    const cp = h.checkpoint || {};
    $('#health-episodes').textContent = cp.episode ?? '—';
    $('#health-steps').textContent = cp.train_step_count != null ? fmt(cp.train_step_count) : '—';
  }).catch(()=>{});
}

/* ---------- Review queue ---------- */
function loadReviewQueue(){
  const el = $('#review-list');
  api('/api/config').then(c => { $('#rq-threshold').textContent = c.review_threshold; }).catch(()=>{});
  api(`/api/review-queue?sid=${encodeURIComponent(socket.id || '')}`).then(data=>{
    const items = data.items || [];
    if(!items.length){ el.innerHTML = '<div class="card"><div class="empty-line">Nothing is waiting for review.</div></div>'; return; }
    el.innerHTML = items.map(it => `
      <div class="review-item" id="rq-${it.review_id}">
        <div class="ri-head"><span>${esc(it.label)}</span><span class="chip ${it.default_action==='BLOCK'?'block':'allow'}">Model default: ${esc(it.default_action)}</span></div>
        <div class="ri-meta">Packet #${it.packet_id} &middot; Confidence ${it.confidence}% &middot; ${esc(it.created_at)}</div>
        ${CAN_CONTROL ? `<div class="review-actions">
          <input type="text" id="reason-${it.review_id}" placeholder="Reason (required)&hellip;">
          <button class="btn btn-approve" data-id="${it.review_id}" data-act="approve" data-final="${esc(it.default_action)}">Approve</button>
          <button class="btn btn-override" data-id="${it.review_id}" data-act="override" data-final="${it.default_action==='BLOCK'?'ALLOW':'BLOCK'}">Override</button>
        </div>` : '<div class="sub-text">Only analysts and admins can resolve review items.</div>'}
      </div>`).join('');
  }).catch(e=>{ el.innerHTML = '<div class="sub-text">' + esc(e.message) + '</div>'; });
}
$('#review-list').addEventListener('click', e=>{
  const b = e.target.closest('button[data-act]'); if(!b) return;
  const id = b.dataset.id;
  const reason = ($('#reason-' + id).value || '').trim();
  if(!reason){ toast('Add a reason before resolving this item.', 'err'); $('#reason-' + id).focus(); return; }
  send(`/api/review/${id}`, {action: b.dataset.act, reason, final_action: b.dataset.final})
    .then(()=>{ toast(b.dataset.act === 'approve' ? 'Decision approved' : 'Decision overridden', 'ok'); loadReviewQueue(); })
    .catch(err => toast(err.message, 'err'));
});

/* ---------- Audit log ---------- */
function loadAudit(){
  const params = new URLSearchParams();
  const label = $('#audit-label-filter').value, decision = $('#audit-decision-filter').value;
  if(label) params.set('label', label);
  if(decision) params.set('decision', decision);
  api(`/api/audit?${params}`).then(data=>{
    const rows = data.rows || [];
    $('#audit-body').innerHTML = rows.map(r => `
      <tr><td>${r.decision_id}</td><td>${esc(r.timestamp)}</td><td>${esc((r.session_id||'').slice(0,8))}</td>
      <td>${esc(r.username)}</td><td>${esc(r.label)}</td><td>${esc(r.predicted_action)}</td><td>${r.confidence}%</td></tr>
    `).join('') || '<tr><td colspan="7" class="sub-text" style="padding:20px;">No matching records.</td></tr>';
  }).catch(e => toast(e.message, 'err'));
}
$('#btn-audit-search').addEventListener('click', loadAudit);
$('#audit-label-filter').addEventListener('keydown', e => { if(e.key === 'Enter') loadAudit(); });

/* ---------- Executive summary ---------- */
let lastExec = null;
function drawGauge(score){
  const c = $('#gauge'), dpr = window.devicePixelRatio || 1;
  const size = c.offsetWidth * dpr; if(!size) return;
  c.width = size; c.height = size;
  const g = c.getContext('2d'), cx = size/2, cy = size/2, r = size*0.41, lw = size*0.07;
  g.lineCap = 'round'; g.lineWidth = lw;
  g.strokeStyle = 'rgba(255,255,255,.07)';
  g.beginPath(); g.arc(cx, cy, r, 0.75*Math.PI, 2.25*Math.PI); g.stroke();
  const frac = Math.max(0, Math.min(1, score/100));
  const grad = g.createLinearGradient(0, size, size, 0);
  grad.addColorStop(0, '#E8A33D'); grad.addColorStop(1, '#5FBFA0');
  g.strokeStyle = grad; g.shadowColor = 'rgba(95,191,160,.45)'; g.shadowBlur = 14*dpr;
  g.beginPath(); g.arc(cx, cy, r, 0.75*Math.PI, 0.75*Math.PI + frac*1.5*Math.PI); g.stroke();
}
const SEV_COLORS = {CRITICAL:'#dc2626', HIGH:'#ea580c', MEDIUM:'#d97706', LOW:'#7C93B8'};
const rateColor = (p) => p >= 90 ? 'var(--signal-mint)' : (p >= 70 ? 'var(--signal-amber)' : 'var(--alert)');
const barRow = (name, valText, pct, color) =>
  `<div class="bar-row"><div class="bar-top"><span>${esc(name)}</span><span class="bt-v">${valText}</span></div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, pct))}%; background:${color}"></div></div></div>`;

function loadExec(){
  api(`/api/executive-summary?sid=${encodeURIComponent(socket.id || '')}`).then(renderExec).catch(()=>{});
}
function renderExec(d){
  lastExec = d;
  $('#exec-empty').style.display = d.empty ? 'block' : 'none';
  $('#exec-body').style.display = d.empty ? 'none' : 'block';
  if(d.empty) return;
  $('#ex-headline').textContent = d.headline;
  $('#ex-body').textContent = d.summary;
  $('#ex-basis').textContent = `Based on ${fmt(d.total)} flows, ${d.progress}% of the evaluation set.`;
  $('#g-score').textContent = d.score.toFixed(1);
  $('#g-rating').textContent = d.rating;
  drawGauge(d.score);
  $('#k-stopped').textContent = fmt(d.tp);   $('#k-stopped-c').textContent = `${d.detection}% of attacks`;
  $('#k-missed').textContent = fmt(d.fn);    $('#k-missed-c').textContent = d.fn ? 'Attacks allowed through' : 'None got through';
  $('#k-fp').textContent = fmt(d.fp);        $('#k-fp-c').textContent = `${d.fpr}% of normal traffic`;
  $('#k-review').textContent = fmt(d.pending_reviews); $('#k-review-c').textContent = `Below ${d.review_threshold}% confidence`;

  const sevTotal = d.severity.reduce((s, x) => s + x.seen, 0);
  $('#sev-bar').innerHTML = d.severity.map(s => `<i style="width:${sevTotal ? s.seen/sevTotal*100 : 0}%; background:${SEV_COLORS[s.name] || '#7C93B8'}"></i>`).join('');
  $('#sev-legend').innerHTML = d.severity.length ? d.severity.map(s =>
    `<div class="legend-row"><span class="nm"><i style="background:${SEV_COLORS[s.name] || '#7C93B8'}"></i>${esc(s.name.charAt(0) + s.name.slice(1).toLowerCase())}</span><span class="vals">${fmt(s.caught)} of ${fmt(s.seen)} stopped</span></div>`
  ).join('') : '<div class="sub-text">No attacks seen yet.</div>';

  $('#top-attacks').innerHTML = d.top_attacks.length ? d.top_attacks.map(a => {
    const p = a.seen ? a.caught/a.seen*100 : 0;
    return barRow(a.label, `${fmt(a.caught)} / ${fmt(a.seen)}`, p, rateColor(p));
  }).join('') : '<div class="sub-text">No attacks seen yet.</div>';

  if(d.rule_available){
    $('#cmp-body').innerHTML =
      `<div class="cmp-group"><h4>Attacks detected</h4>${barRow('AdaptiveShield', d.detection + '%', d.detection, 'var(--signal-mint)')}${barRow('Static rule', d.rule_detection + '%', d.rule_detection, 'var(--ledger-ink)')}</div>` +
      `<div class="cmp-group"><h4>Normal traffic wrongly blocked (lower is better)</h4>${barRow('AdaptiveShield', d.fpr + '%', d.fpr, 'var(--signal-amber)')}${barRow('Static rule', d.rule_fpr + '%', d.rule_fpr, 'var(--ledger-ink)')}</div>` +
      `<div class="sub-text" style="margin:0;">The two agreed on ${d.rule_agreement}% of decisions.</div>`;
  } else {
    $('#cmp-body').innerHTML = '<div class="sub-text">The rule comparison needs the scaler files. Run <code>python -m src.preprocess</code> once to enable it.</div>';
  }
  $('#mitre-list').innerHTML = d.mitre.length ? d.mitre.map(m =>
    `<div class="mitre-row"><span class="mid">${esc(m.id)}</span><span class="mn">${esc(m.name)}</span><span class="mc">${fmt(m.seen)}</span></div>`
  ).join('') : '<div class="sub-text">No mapped techniques observed yet.</div>';
}
$('#btn-exec-print').addEventListener('click', ()=> window.print());
$('#btn-exec-copy').addEventListener('click', ()=>{
  if(!lastExec || lastExec.empty) return;
  const d = lastExec;
  const text = `AdaptiveShield summary\n${d.headline}\n\n${d.summary}\n\nProtection score: ${d.score} (${d.rating})\nAttacks stopped: ${d.tp}\nAttacks missed: ${d.fn}\nFalse alarms: ${d.fp}\nFlows analysed: ${d.total}`;
  (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject())
    .then(()=> toast('Summary copied', 'ok')).catch(()=> toast('Copy is blocked in this browser. Use Save as PDF instead.', 'err'));
});

/* ---------- Admin controls ---------- */
let thrDirty = false;
const thrRange = $('#thr-range');
if(thrRange){
  thrRange.addEventListener('input', ()=>{ thrDirty = true; $('#thr-val').textContent = thrRange.value + '%'; });
  $('#btn-thr-apply').addEventListener('click', ()=>{
    send('/api/admin/threshold', {value: parseInt(thrRange.value)}).then(r=>{
      thrDirty = false; toast(`Review threshold set to ${r.review_threshold}%`, 'ok'); loadAdmin();
    }).catch(e => toast(e.message, 'err'));
  });
  $('#btn-add-user').addEventListener('click', ()=>{
    const username = $('#nu-name').value.trim(), password = $('#nu-pass').value, role = $('#nu-role').value;
    send('/api/admin/users', {username, password, role}).then(()=>{
      toast(`Added ${username}`, 'ok'); $('#nu-name').value = ''; $('#nu-pass').value = ''; loadAdmin();
    }).catch(e => toast(e.message, 'err'));
  });
  $('#sess-body').addEventListener('click', e=>{
    const b = e.target.closest('button[data-act]'); if(!b) return;
    const act = b.dataset.act;
    const run = ()=> send('/api/admin/session', {sid: b.dataset.sid, action: act})
      .then(()=>{ toast('Session ' + (act === 'stop' ? 'stopped' : act === 'pause' ? 'paused' : 'resumed'), 'ok'); loadAdmin(); })
      .catch(err => toast(err.message, 'err'));
    if(act === 'stop') askModal({title:'Stop this session?', body:'The replay for this person ends immediately. They can restart it from their dashboard.', ok:'Stop session', danger:true}).then(v => { if(v) run(); });
    else run();
  });
  $('#users-body').addEventListener('click', async e=>{
    const b = e.target.closest('button[data-act]'); if(!b) return;
    const user = b.dataset.user;
    if(b.dataset.act === 'password'){
      const pw = await askModal({title:`Reset password for ${user}`, body:'Choose a temporary password (10+ characters, with a letter and a number). They will be asked to change it at their next sign-in.', input:'New password', ok:'Reset password'});
      if(pw === null) return;
      send(`/api/admin/users/${encodeURIComponent(user)}`, {password: pw}).then(()=> toast('Password updated', 'ok')).catch(err => toast(err.message, 'err'));
    } else if(b.dataset.act === 'remove'){
      const ok = await askModal({title:`Remove ${user}?`, body:'They lose access immediately and any open session of theirs is stopped.', ok:'Remove user', danger:true});
      if(!ok) return;
      send(`/api/admin/users/${encodeURIComponent(user)}`, {}, 'DELETE').then(()=>{ toast(`Removed ${user}`, 'ok'); loadAdmin(); }).catch(err => toast(err.message, 'err'));
    }
  });
  $('#users-body').addEventListener('change', e=>{
    const s = e.target.closest('select[data-user]'); if(!s) return;
    send(`/api/admin/users/${encodeURIComponent(s.dataset.user)}`, {role: s.value})
      .then(()=> toast(`${s.dataset.user} is now ${s.value}`, 'ok')).catch(err => { toast(err.message, 'err'); loadAdmin(); });
  });
}
function stateChip(s){
  if(s.stopped) return '<span class="chip block">Stopped</span>';
  if(s.paused) return '<span class="chip amber">Paused</span>';
  return '<span class="chip allow">Running</span>';
}
function loadAdmin(){
  if(!thrRange) return;
  api('/api/admin/overview').then(o=>{
    $('#a-cpu').textContent = o.system.cpu + '%';
    $('#a-mem').textContent = o.system.mem + '%';
    $('#a-engine').textContent = o.engine.ready ? 'Ready' : 'Loading';
    $('#a-engine').style.color = o.engine.ready ? 'var(--signal-mint)' : 'var(--signal-amber)';
    $('#a-engine-c').textContent = o.engine.ready ? `${fmt(o.engine.total)} packets, ${fmt(o.engine.params)} parameters` : 'Preparing the model';
    $('#a-sessions').textContent = o.sessions.length;
    if(!thrDirty){ thrRange.value = o.review_threshold; $('#thr-val').textContent = o.review_threshold + '%'; }

    $('#sess-body').innerHTML = o.sessions.map(s=>{
      const pct = o.engine.total ? Math.min(100, s.position/o.engine.total*100) : 0;
      return `<tr><td>${esc(s.user)}${s.user === ME ? ' <span class="sub-text">(you)</span>' : ''}</td>
        <td><span class="prog-mini"><i style="width:${pct}%"></i></span><span class="mono sub-text">${fmt(s.position)}</span></td>
        <td>${stateChip(s)}</td><td class="mono">${s.speed}&times;</td>
        <td><div class="act-cell">${s.stopped ? '' : (s.paused
          ? `<button class="btn small" data-act="resume" data-sid="${esc(s.sid)}">Resume</button>`
          : `<button class="btn small" data-act="pause" data-sid="${esc(s.sid)}">Pause</button>`) +
          `<button class="btn small btn-danger" data-act="stop" data-sid="${esc(s.sid)}">Stop</button>`}</div></td></tr>`;
    }).join('') || '<tr><td colspan="5" class="sub-text" style="padding:20px;">No open sessions.</td></tr>';

    const active = document.activeElement;
    if(!(active && active.closest && active.closest('#users-body'))){
      $('#users-body').innerHTML = o.users.map(u=>{
        const self = u.username === ME;
        const roleCell = self
          ? `<span class="chip ink">${esc(u.role)}</span> <span class="sub-text">(you)</span>`
          : `<select data-user="${esc(u.username)}">${['viewer','analyst','admin'].map(r => `<option value="${r}" ${u.role === r ? 'selected' : ''}>${r.charAt(0).toUpperCase() + r.slice(1)}</option>`).join('')}</select>`;
        return `<tr><td>${esc(u.username)}</td><td>${roleCell}</td><td class="mono">${u.sessions}</td>
          <td><div class="act-cell"><button class="btn small" data-act="password" data-user="${esc(u.username)}">Reset password</button>
          ${self ? '' : `<button class="btn small btn-danger" data-act="remove" data-user="${esc(u.username)}">Remove</button>`}</div></td></tr>`;
      }).join('');
    }
    $('#events-list').innerHTML = o.events.map(ev =>
      `<div class="event-row"><span class="et">${esc(ev.time)}</span><span><b style="font-weight:500">${esc(ev.actor)}</b> <span class="ea">${esc(ev.action)}${ev.detail ? ': ' + esc(ev.detail) : ''}</span></span></div>`
    ).join('') || '<div class="sub-text">No changes yet.</div>';
  }).catch(()=>{});
}

/* ---------- Live socket events ---------- */
function resetUI(){
  pulseData = []; scatterPoints.length = 0; confMargins.length = 0; driftScores.length = 0;
  packetCache.clear();
  $('#feed-body').innerHTML = ''; $('#timeline-strip').innerHTML = '';
  $('#ledger-body').innerHTML = '<div class="empty-line" id="ledger-empty">Decisions will appear here as the replay runs.</div>';
  ['m-accuracy','m-fpr','m-f1','m-ba'].forEach(id => $('#' + id).textContent = '—');
  $('#last-decision').textContent = '—'; $('#last-decision').className = 'hd-word';
  $('#last-label').textContent = 'Waiting for traffic';
  $('#progress-fill').style.width = '0%';
  $('#inspector-empty').style.display = 'flex'; $('#inspector-content').style.display = 'none';
  drawPulse(); drawScatter(); drawConfHistogram(); drawDriftTimeline();
}
function setPaused(p){
  paused = p;
  const b = $('#btn-pause'); if(b) b.textContent = p ? 'Resume' : 'Pause';
}

socket.on('ready', (data)=>{
  $('#loading').style.display = 'none';
  totalPackets = data.total_packets;
  $('#pkt-count').textContent = fmt(totalPackets);
  setStatus('live', 'Replaying');
});
socket.on('load_error', (data)=>{
  $('#loading-text').textContent = 'The model could not be loaded: ' + data.message;
});
socket.on('session_state', (s)=>{
  if(s.state === 'paused'){ setPaused(true); setStatus('paused', 'Paused'); }
  else if(s.state === 'running'){ setPaused(false); setStatus('live', 'Replaying'); }
  else if(s.state === 'stopped'){ setPaused(false); setStatus('stopped', 'Stopped'); }
});
socket.on('update', (d)=>{
  pulseData.push(d.last_reward > 0 ? 1 : -1);
  if(pulseData.length > MAX_POINTS) pulseData.shift();
  drawPulse();
  $('#last-label').textContent = d.last_label;
  const word = $('#last-decision');
  word.textContent = d.last_action; word.className = 'hd-word ' + (d.last_action === 'BLOCK' ? 'block' : 'allow');
  $('#m-accuracy').textContent = d.accuracy + '%';
  $('#m-fpr').textContent = d.fpr + '%';
  $('#m-f1').textContent = d.f1 + '%';
  $('#m-ba').textContent = fmt(d.blocked) + ' / ' + fmt(d.allowed);
  if(totalPackets) $('#progress-fill').style.width = Math.min(100, d.total/totalPackets*100) + '%';
  addLedgerRow(d.last_label, d.last_action, d.confidence, d.total);
  addFeedRow(d);
  updateShadowAndDrift(d);
});
socket.on('done', (stats)=>{
  $('#m-fpr').textContent = stats.fpr + '%';
  $('#progress-fill').style.width = '100%';
  setStatus('done', 'Replay complete');
});

/* ---------- Playback controls ---------- */
const pauseBtn = $('#btn-pause');
if(pauseBtn){
  pauseBtn.addEventListener('click', ()=>{ socket.emit('control', paused ? 'resume' : 'pause'); });
  $('#btn-stop').addEventListener('click', ()=> socket.emit('control', 'stop'));
  $('#btn-restart').addEventListener('click', ()=>{
    setPaused(false); resetUI(); setStatus('live', 'Replaying');
    socket.emit('control', 'restart');
  });
  $('#speed').addEventListener('input', (e)=> socket.emit('set_speed', parseInt(e.target.value)));
}

setStatus('live', 'Starting');
drawPulse();
</script>
</body>
</html>
"""


def _finish_login(username):
    user = store.get_user(username)
    session.clear()
    session.permanent = True
    session['username'] = username
    session['role'] = user['role']
    session['last_seen'] = time.time()
    store.update_user(username, last_login=store.now())
    db.log_access(username, "login")
    if user['must_change']:
        return redirect(url_for('account_password'))
    return redirect(url_for("index"))


def _locked_message(wait):
    mins = max(1, round(wait / 60))
    return f"Too many failed attempts. Try again in {mins} minute{'s' if mins != 1 else ''}."


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()[:64]
        password = request.form.get("password") or ""
        ip = _client_ip()
        wait = security.lockout_seconds(username, ip, config.LOCKOUT_MAX_FAILS, config.LOCKOUT_WINDOW_MIN)
        if wait:
            return render_template_string(LOGIN_HTML, error=_locked_message(wait)), 429
        user = store.get_user(username)
        ok = check_password_hash(user["password_hash"] if user else _DUMMY_HASH, password) and user is not None
        if not ok:
            security.record_attempt(username, ip, False)
            return render_template_string(LOGIN_HTML, error="That username or password is incorrect.")
        if user["totp_enabled"]:
            # Password is right; the failure count is NOT cleared until the code is right too.
            session.clear()
            session["pending_2fa"] = username
            session["pending_2fa_at"] = time.time()
            return redirect(url_for("login_2fa"))
        security.record_attempt(username, ip, True)
        return _finish_login(username)
    return render_template_string(LOGIN_HTML, error=None)


@app.route("/login/2fa", methods=["GET", "POST"])
def login_2fa():
    username = session.get("pending_2fa")
    user = store.get_user(username) if username else None
    if not user or not user["totp_enabled"] or time.time() - session.get("pending_2fa_at", 0) > 300:
        session.pop("pending_2fa", None)
        return redirect(url_for("login"))
    error = None
    if request.method == "POST":
        ip = _client_ip()
        wait = security.lockout_seconds(username, ip, config.LOCKOUT_MAX_FAILS, config.LOCKOUT_WINDOW_MIN)
        if wait:
            error = _locked_message(wait)
        else:
            counter = security.verify_totp(user["totp_secret"], request.form.get("code"), user["totp_last"] or 0)
            if counter is None:
                security.record_attempt(username, ip, False)
                error = "That code isn't right. Check your authenticator app and try again."
            else:
                store.update_user(username, totp_last=counter)
                security.record_attempt(username, ip, True)
                return _finish_login(username)
    return render_template_string(
        AUTH_HTML, page_title="AdaptiveShield – Two-factor", heading="Two-factor code",
        sub="Enter the 6-digit code from your authenticator app.", error=error, extra=None,
        fields=[("code", "Authentication code", "text", "one-time-code")], button="Verify",
        links=[("/login", "Back to sign in")])


@app.route("/account/password", methods=["GET", "POST"])
@login_required
def account_password():
    me = session["username"]
    user = store.get_user(me)
    forced = bool(user["must_change"])
    error = None
    if request.method == "POST":
        current = request.form.get("current") or ""
        new = request.form.get("new") or ""
        confirm = request.form.get("confirm") or ""
        if not check_password_hash(user["password_hash"], current):
            error = "Your current password is incorrect."
        elif new != confirm:
            error = "The two new passwords don't match."
        elif check_password_hash(user["password_hash"], new):
            error = "Choose a password different from the current one."
        else:
            error = security.check_password(new, me)
        if not error:
            store.update_user(me, password_hash=generate_password_hash(new), must_change=0)
            store.add_event(me, "changed own password", "")
            db.log_access(me, "password-change")
            return redirect(url_for("index"))
    return render_template_string(
        AUTH_HTML, page_title="AdaptiveShield – Change password", heading="Change password",
        sub=("Your password was set by an administrator. Choose your own to continue." if forced
             else "Use at least 10 characters, with a letter and a number."),
        error=error, extra=None,
        fields=[("current", "Current password", "password", "current-password"),
                ("new", "New password", "password", "new-password"),
                ("confirm", "Confirm new password", "password", "new-password")],
        button="Save password",
        links=[("/logout", "Sign out")] if forced else [("/account/2fa", "Two-factor sign-in"), ("/", "Back to dashboard")])


@app.route("/account/2fa", methods=["GET", "POST"])
@login_required
def account_2fa():
    me = session["username"]
    user = store.get_user(me)
    error = None
    back = [("/account/password", "Change password"), ("/", "Back to dashboard")]
    if user["totp_enabled"]:
        if request.method == "POST":
            if check_password_hash(user["password_hash"], request.form.get("password") or ""):
                store.update_user(me, totp_enabled=0, totp_secret=None, totp_last=0)
                store.add_event(me, "turned off two-factor", "")
                return redirect(url_for("index"))
            error = "That password is incorrect."
        return render_template_string(
            AUTH_HTML, page_title="AdaptiveShield – Two-factor", heading="Two-factor is on",
            sub="Enter your password to turn it off.", error=error, extra=None,
            fields=[("password", "Password", "password", "current-password")],
            button="Turn off two-factor", links=back)
    secret = session.get("totp_setup") or security.new_secret()
    session["totp_setup"] = secret
    if request.method == "POST":
        counter = security.verify_totp(secret, request.form.get("code"))
        if counter is not None:
            store.update_user(me, totp_secret=secret, totp_enabled=1, totp_last=counter)
            store.add_event(me, "turned on two-factor", "")
            session.pop("totp_setup", None)
            return redirect(url_for("index"))
        error = "That code isn't right. Check the time on your phone and try again."
    svg = security.qr_svg(security.provisioning_uri(secret, me))
    extra = ((f'<div style="background:#fff;padding:10px;border-radius:12px;margin-bottom:12px">{svg}</div>' if svg else "")
             + f'<p class="sub" style="margin-bottom:0">Secret key: <code>{secret}</code></p>')
    return render_template_string(
        AUTH_HTML, page_title="AdaptiveShield – Two-factor", heading="Set up two-factor",
        sub="Scan the QR code (or type the key) into an authenticator app, then enter the 6-digit code it shows.",
        error=error, extra=extra, fields=[("code", "Authentication code", "text", "one-time-code")],
        button="Turn on two-factor", links=back)


@app.route("/logout")
def logout():
    if "username" in session:
        db.log_access(session["username"], "logout")
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template_string(HTML, role=session.get('role'), username=session.get('username'))


@app.route("/download-logs")
@login_required
def download_logs():
    sid = current_sid()
    with sessions_lock:
        st = sessions.get(sid)
    logs = st['logs'] if st else []
    db.log_access(session.get("username"), "export", sid)
    if not logs:
        return "No logs yet.", 400

    def generate():
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=[
            'packet_id', 'timestamp', 'label', 'is_attack',
            'action', 'decision', 'reward', 'accuracy', 'precision', 'recall', 'f1',
            'confidence', 'cvss_severity', 'cvss_score'])
        w.writeheader(); yield out.getvalue(); out.truncate(0); out.seek(0)
        for row in list(logs):
            w.writerow(row); yield out.getvalue(); out.truncate(0); out.seek(0)
    return Response(generate(), headers={
        'Content-Disposition': 'attachment; filename=adaptiveshield_logs.csv',
        'Content-Type': 'text/csv'})


@app.route("/stats")
@login_required
def stats():
    sid = current_sid()
    with sessions_lock:
        st = sessions.get(sid)
    return jsonify(st['stats'] if st else {})


@app.route("/api/config")
@login_required
def api_config():
    return jsonify({'review_threshold': review_threshold(), 'role': session.get('role')})


@app.route("/api/attribution/<int:packet_id>")
@login_required
def api_attribution(packet_id):
    """Gradient x input feature attribution for one packet, computed
    on-demand (not in the hot playback loop -- only run when a viewer
    actually clicks a packet). Returns the features that most pushed the
    Q-value gap toward the model's chosen action, with sign: positive
    means it pushed TOWARD the decision made, negative means it pushed
    AGAINST it (but was outweighed by other features)."""
    if not engine.get('ready') or engine.get('model') is None:
        return jsonify({'error': 'The engine is not ready yet.'}), 503
    feature_names = engine.get('feature_names')
    if feature_names is None:
        return jsonify({'error': 'Feature names are unavailable. Run python -m src.preprocess once.'}), 503

    idx = packet_id - 1
    X_test = engine['X_test']
    if idx < 0 or idx >= len(X_test):
        return jsonify({'error': 'Packet id is out of range.'}), 404

    model = engine['model']
    x = tf.convert_to_tensor(X_test[idx:idx + 1], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        q_vals = model(x, training=False)
        action = int(tf.argmax(q_vals[0]))
        target_q = q_vals[0][action]
    grads = tape.gradient(target_q, x).numpy()[0]
    # gradient x input: weights each feature's gradient by its own actual
    # value, so a feature that's zero can't appear falsely influential.
    attribution = grads * X_test[idx]
    ranked = sorted(zip(feature_names, attribution.tolist()), key=lambda p: abs(p[1]), reverse=True)
    top = [{'feature': name, 'attribution': round(float(val), 5)} for name, val in ranked[:8]]
    return jsonify({'packet_id': packet_id, 'action': action, 'attribution': top})


@app.route("/api/similar/<int:packet_id>")
@login_required
def api_similar(packet_id):
    """Nearest-neighbor lookup in scaled feature space -- 'pivot on this
    packet' to find other flows in the test set that look statistically
    similar, the way a SOC analyst would manually cross-reference related
    traffic. Distance is computed in the same normalized 0-1 space the
    model itself uses."""
    if not engine.get('ready'):
        return jsonify({'error': 'The engine is not ready yet.'}), 503
    idx = packet_id - 1
    X_test = engine['X_test']; y_test = engine['y_test']; classes = engine['classes']
    if idx < 0 or idx >= len(X_test):
        return jsonify({'error': 'Packet id is out of range.'}), 404

    target = X_test[idx]
    dists = np.linalg.norm(X_test - target, axis=1)
    dists[idx] = np.inf  # exclude the packet itself
    nearest = np.argsort(dists)[:5]
    results = []
    for n_idx in nearest:
        label_idx = int(y_test[n_idx])
        label = classes[label_idx] if label_idx < len(classes) else "Unknown"
        results.append({
            'packet_id': int(n_idx) + 1,
            'label': label,
            'distance': round(float(dists[n_idx]), 4),
        })
    return jsonify({'packet_id': packet_id, 'similar': results})


@app.route("/api/notes/<int:packet_id>", methods=["GET", "POST"])
@login_required
def api_notes(packet_id):
    if request.method == "POST":
        # Read-only viewers can't write investigation annotations, same
        # restriction as playback controls.
        if session.get('role') not in ('admin', 'analyst'):
            return jsonify({'error': 'You do not have permission to add notes.'}), 403
        data = request.json or {}
        text = data.get('note_text', '').strip()
        # The frontend sends socket.id explicitly because session['sid'] set
        # inside a Socket.IO handler isn't reliably synced to the browser cookie.
        sid = current_sid(data.get('sid'))
        if not text:
            return jsonify({'error': 'The note is empty.'}), 400
        db.add_note(packet_id, sid, session.get('username'), text)
        return jsonify({'ok': True})
    return jsonify({'notes': db.get_notes(packet_id)})


@app.route("/api/health")
@login_required
def api_health():
    if not engine.get('ready'):
        return jsonify({'error': 'The engine is not ready yet.'}), 503
    return jsonify(engine.get('health', {}))


@app.route("/api/review-queue")
@login_required
def api_review_queue():
    sid = current_sid(request.args.get('sid'))
    return jsonify({'items': db.get_pending_reviews(sid)})


@app.route("/api/review/<int:review_id>", methods=["POST"])
@login_required
def api_review_resolve(review_id):
    if session.get('role') not in ('admin', 'analyst'):
        return jsonify({'error': 'You do not have permission to resolve reviews.'}), 403
    data = request.json or {}
    action = data.get('action')  # 'approve' or 'override'
    reason = data.get('reason', '').strip()
    final_action = data.get('final_action')
    if action not in ('approve', 'override') or not reason:
        return jsonify({'error': 'An action and a reason are required.'}), 400
    db.resolve_review(review_id, session.get('username'), final_action, reason)
    return jsonify({'ok': True})


@app.route("/api/audit")
@login_required
def api_audit():
    label = request.args.get('label') or None
    decision = request.args.get('decision') or None
    return jsonify({'rows': db.get_audit_log(label, decision)})


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------
def pct(n, d, default=0.0):
    return round(n / d * 100, 2) if d else default


def build_summary(st, sid):
    """Turns one session's running counters into a leadership-friendly readout.
    The protection score is a simple composite, spelled out here so it is
    never a black box: 50% attack detection rate, 30% precision, and 20%
    the share of normal traffic left alone."""
    a = st['agg']
    total = a['total']
    if total == 0:
        return {'empty': True}

    tp, fp, fn, tn = a['tp'], a['fp'], a['fn'], a['tn']
    attacks, benign = tp + fn, fp + tn
    detection = pct(tp, attacks, 100.0)
    precision = pct(tp, tp + fp, 100.0)
    fpr = pct(fp, benign, 0.0)
    score = round(0.5 * detection + 0.3 * precision + 0.2 * (100 - fpr), 1)
    if score >= 95:
        rating = 'Strong'
    elif score >= 85:
        rating = 'Good'
    elif score >= 70:
        rating = 'Fair'
    else:
        rating = 'At risk'

    labels = list(a['by_label'].items())
    attack_rows = sorted([(n, v) for n, v in labels if v['attack']], key=lambda p: p[1]['seen'], reverse=True)
    top_attacks = [{'label': n, 'seen': v['seen'], 'caught': v['seen'] - v['missed']} for n, v in attack_rows[:6]]
    missed_rows = sorted([(n, v) for n, v in attack_rows if v['missed'] > 0], key=lambda p: p[1]['missed'], reverse=True)

    sev_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    sev_items = list(a['sev'].items())
    severity = [{'name': k, 'seen': v['seen'], 'caught': v['caught']}
                for k in sev_order for (kk, v) in sev_items if kk == k]
    severity += [{'name': k, 'seen': v['seen'], 'caught': v['caught']} for k, v in sev_items if k not in sev_order]

    mitre = sorted([{'id': k, 'name': v['name'], 'seen': v['seen']} for k, v in list(a['mitre'].items())],
                   key=lambda m: m['seen'], reverse=True)[:6]

    if attacks == 0:
        headline = f"No attacks seen yet; {tn:,} normal flows were allowed through."
    else:
        headline = (f"Stopped {tp:,} of {attacks:,} attacks, with {fp:,} false "
                    f"{'alarm' if fp == 1 else 'alarms'}.")

    parts = [f"AdaptiveShield has analysed {total:,} network flows and caught {detection:.1f}% of the attacks among them."]
    if benign:
        parts.append(f"It blocked {fp:,} legitimate {'flow' if fp == 1 else 'flows'} by mistake, {fpr:.2f}% of normal traffic.")
    if fn:
        worst = missed_rows[0][0] if missed_rows else None
        parts.append(f"{fn:,} {'attack' if fn == 1 else 'attacks'} got through" +
                     (f", most often {worst}." if worst else "."))
    elif attacks:
        parts.append("No attacks got through.")
    rule_available = engine.get('feature_names') is not None and engine.get('baseline_mean') is not None
    if rule_available and attacks:
        rd = pct(a['rule_tp'], attacks, 0.0)
        parts.append(f"A static threshold rule on the same traffic would have detected {rd:.1f}% of attacks.")

    try:
        pending = len(db.get_pending_reviews(sid))
    except Exception:
        pending = a['held']

    return {
        'empty': False, 'total': total, 'progress': round(min(100, total / (engine.get('total_test_size') or total) * 100), 1),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'detection': detection, 'precision': precision, 'fpr': fpr,
        'score': score, 'rating': rating,
        'headline': headline, 'summary': ' '.join(parts),
        'top_attacks': top_attacks, 'severity': severity, 'mitre': mitre,
        'rule_available': rule_available,
        'rule_detection': pct(a['rule_tp'], attacks, 0.0), 'rule_fpr': pct(a['rule_fp'], benign, 0.0),
        'rule_agreement': pct(a['rule_agree'], total, 0.0),
        'pending_reviews': pending, 'review_threshold': review_threshold(),
        'avg_confidence': round(a['conf_sum'] / total, 1),
    }


@app.route("/api/executive-summary")
@login_required
def api_executive_summary():
    sid = current_sid(request.args.get('sid'))
    with sessions_lock:
        st = sessions.get(sid)
    if not st:
        return jsonify({'empty': True})
    return jsonify(build_summary(st, sid))


# ---------------------------------------------------------------------------
# Admin Controls
# ---------------------------------------------------------------------------
def speed_from_delay(delay):
    return max(1, min(10, round(1 + (0.2 - delay) * 9 / 0.195)))


def emit_state(sid, st):
    state = 'stopped' if st['is_stopped'] else ('paused' if st['is_paused'] else 'running')
    socketio.emit('session_state', {'state': state}, room=sid)


@app.route("/api/admin/overview")
@admin_required
def api_admin_overview():
    with sessions_lock:
        sess = [{
            'sid': sid, 'user': st.get('username') or '',
            'position': st['agg']['total'], 'paused': st['is_paused'], 'stopped': st['is_stopped'],
            'speed': speed_from_delay(st['packet_delay']),
        } for sid, st in sessions.items()]
    users = [{'username': u['username'], 'role': u['role'],
              'sessions': sum(1 for x in sess if x['user'] == u['username']),
              'two_factor': bool(u['totp_enabled']), 'must_change': bool(u['must_change'])}
             for u in store.list_users()]
    health = engine.get('health') or {}
    return jsonify({
        'system': {'cpu': round(psutil.cpu_percent(), 1), 'mem': round(psutil.virtual_memory().percent, 1)},
        'engine': {'ready': bool(engine.get('ready')), 'total': engine.get('total_test_size', 0),
                   'params': health.get('params', 0), 'model_path': health.get('model_path')},
        'review_threshold': review_threshold(),
        'sessions': sess, 'users': users, 'events': [{'time': (e['ts'] or '')[11:], 'actor': e['actor'], 'action': e['action'], 'detail': e['detail']}
                   for e in store.recent_events(30)],
        'models': registry.list(), 'active_model': engine.get('model_id'),
    })


@app.route("/api/admin/threshold", methods=["POST"])
@admin_required
def api_admin_threshold():
    try:
        value = int((request.json or {}).get('value'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Enter a whole number between 5 and 90.'}), 400
    if not 5 <= value <= 90:
        return jsonify({'error': 'Enter a whole number between 5 and 90.'}), 400
    old = review_threshold()
    store.set_setting('review_threshold', value)
    record_admin_event(session['username'], 'changed review threshold', f'{old}% to {value}%')
    return jsonify({'ok': True, 'review_threshold': value})


@app.route("/api/admin/session", methods=["POST"])
@admin_required
def api_admin_session():
    data = request.json or {}
    sid, action = data.get('sid'), data.get('action')
    if action not in ('pause', 'resume', 'stop'):
        return jsonify({'error': 'Unknown action.'}), 400
    with sessions_lock:
        st = sessions.get(sid)
    if not st:
        return jsonify({'error': 'That session has already ended.'}), 404
    if action == 'pause':
        st['is_paused'] = True
    elif action == 'resume':
        st['is_paused'] = False
    else:
        st['is_stopped'] = True
        st['is_paused'] = False
    emit_state(sid, st)
    record_admin_event(session['username'], f'{action} session', st.get('username') or sid[:8])
    return jsonify({'ok': True})


@app.route("/api/admin/users", methods=["POST"])
@admin_required
def api_admin_add_user():
    data = request.json or {}
    name = (data.get('username') or '').strip()
    password = data.get('password') or ''
    role = data.get('role')
    if not USERNAME_RE.match(name):
        return jsonify({'error': 'Usernames use 3 to 32 letters, numbers, dots, dashes or underscores.'}), 400
    problem = security.check_password(password, name)
    if problem:
        return jsonify({'error': problem}), 400
    if role not in VALID_ROLES:
        return jsonify({'error': 'Choose a valid role.'}), 400
    try:
        store.create_user(name, generate_password_hash(password), role, must_change=True)
    except sqlite3.IntegrityError:
        return jsonify({'error': f'{name} already exists.'}), 409
    record_admin_event(session['username'], 'added user', f'{name} as {role} (must change password at first sign-in)')
    return jsonify({'ok': True})


@app.route("/api/admin/users/<name>", methods=["POST", "DELETE"])
@admin_required
def api_admin_user(name):
    me = session['username']
    if not store.get_user(name):
        return jsonify({'error': 'That user no longer exists.'}), 404

    if request.method == "DELETE":
        if name == me:
            return jsonify({'error': 'You cannot remove your own account.'}), 400
        store.delete_user(name)
        security.clear_lockout(name)
        with sessions_lock:
            for sid, st in sessions.items():
                if st.get('username') == name:
                    st['is_stopped'] = True
                    st['is_paused'] = False
        record_admin_event(me, 'removed user', name)
        return jsonify({'ok': True})

    data = request.json or {}
    # Validate everything first so a bad field can't leave a half-applied change.
    if 'role' in data:
        if name == me:
            return jsonify({'error': 'You cannot change your own role.'}), 400
        if data['role'] not in VALID_ROLES:
            return jsonify({'error': 'Choose a valid role.'}), 400
    if 'password' in data:
        problem = security.check_password(data['password'], name)
        if problem:
            return jsonify({'error': problem}), 400
    if 'role' in data:
        store.update_user(name, role=data['role'])
        record_admin_event(me, 'changed role', f"{name} to {data['role']}")
    if 'password' in data:
        store.update_user(name, password_hash=generate_password_hash(data['password']), must_change=1)
        security.clear_lockout(name)
        record_admin_event(me, 'reset password', f'{name} (must change at next sign-in)')
    if data.get('reset_2fa'):
        store.update_user(name, totp_enabled=0, totp_secret=None, totp_last=0)
        record_admin_event(me, 'reset two-factor', name)
    if data.get('unlock'):
        security.clear_lockout(name)
        record_admin_event(me, 'unlocked account', name)
    return jsonify({'ok': True})


@app.route("/api/admin/audit.csv")
@admin_required
def api_admin_audit_csv():
    rows = db.get_audit_log(None, None)
    fields = ['decision_id', 'timestamp', 'session_id', 'username', 'label', 'predicted_action', 'confidence']

    def generate():
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); yield out.getvalue(); out.truncate(0); out.seek(0)
        for r in rows:
            w.writerow(r); yield out.getvalue(); out.truncate(0); out.seek(0)
    record_admin_event(session['username'], 'exported audit log')
    return Response(generate(), headers={
        'Content-Disposition': 'attachment; filename=adaptiveshield_audit_log.csv',
        'Content-Type': 'text/csv'})


# ---------------------------------------------------------------------------
# Models: list, load, evaluate, compare, robustness, and switching the live model
# ---------------------------------------------------------------------------
def pick_default_model():
    """The admin's saved choice if it still exists, otherwise the first DQN model on disk."""
    dqn = [m['id'] for m in registry.list() if m['kind'] == 'dqn']
    wanted = store.get_setting('default_model', config.DEFAULT_MODEL_FILE)
    return wanted if wanted in dqn else (dqn[0] if dqn else None)


def activate_model(mid):
    """Make a DQN model the one new and restarted replays use. Sessions already running
    keep the predictions they started with. Baselines are compare-only: the attribution and
    similar-flow panels need a differentiable Keras model."""
    entry = registry.get(mid)
    if entry is None:
        raise KeyError(f"Unknown model '{mid}'.")
    if entry['kind'] != 'dqn':
        raise ValueError("Only DQN models can drive the live replay. Baselines are for comparison.")
    entry = registry.load(mid)
    health = dict(engine.get('health') or {})
    health.update(model_path=entry['path'], params=entry['params'],
                  accuracy=entry['health']['accuracy'], checkpoint=None)
    with engine_lock:
        engine.update(all_q_vals=entry['q_vals'], model=entry['model'], model_id=mid, health=health)
    store.set_setting('default_model', mid)


@app.route("/api/models")
@login_required
def api_models():
    return jsonify({'models': registry.list(), 'active': engine.get('model_id')})


@app.route("/api/models/<mid>/load", methods=["POST"])
@login_required
def api_model_load(mid):
    if session.get('role') not in ('admin', 'analyst'):
        return jsonify({'error': 'You do not have permission to load models.'}), 403
    if registry.get(mid) is None:
        return jsonify({'error': 'Unknown model.'}), 404
    if not engine.get('ready'):
        return jsonify({'error': 'The engine is not ready yet.'}), 503
    entry = registry.load_async(mid)
    status = 'loading' if entry['status'] in ('available', 'error') else entry['status']
    return jsonify({'ok': True, 'status': status})


@app.route("/api/models/<mid>/evaluation")
@login_required
def api_model_evaluation(mid):
    entry = registry.get(mid)
    if entry is None:
        return jsonify({'error': 'Unknown model.'}), 404
    if entry['status'] != 'ready':
        return jsonify({'error': 'Load this model first.', 'status': entry['status']}), 409
    return jsonify({'id': mid, 'name': entry['name'], 'evaluation': entry['eval']})


@app.route("/api/models/compare")
@login_required
def api_model_compare():
    a, b = registry.ready(request.args.get('a', '')), registry.ready(request.args.get('b', ''))
    if not a or not b:
        return jsonify({'error': 'Load both models first.'}), 409
    d = registry.data
    return jsonify({'a': {'id': a['id'], 'name': a['name']}, 'b': {'id': b['id'], 'name': b['name']},
                    **evaluation.compare(a['q_vals'], b['q_vals'], d['y_test'], d['classes'])})


@app.route("/api/models/<mid>/robustness", methods=["GET", "POST"])
@login_required
def api_model_robustness(mid):
    entry = registry.ready(mid)
    if entry is None:
        return jsonify({'error': 'Load this model first.'}), 409
    if request.method == "POST":
        if session.get('role') not in ('admin', 'analyst'):
            return jsonify({'error': 'You do not have permission to run robustness tests.'}), 403
        return jsonify(registry.run_robustness(mid))
    return jsonify(entry.get('robust') or {'status': 'idle'})


@app.route("/api/admin/model", methods=["POST"])
@admin_required
def api_admin_model():
    mid = (request.json or {}).get('id')
    try:
        activate_model(mid)
    except (KeyError, ValueError, RuntimeError) as exc:
        return jsonify({'error': str(exc).strip("'\"")}), 400
    except Exception as exc:
        return jsonify({'error': f'Could not load that model: {str(exc)[:200]}'}), 500
    record_admin_event(session['username'], 'changed live model', str(mid))
    return jsonify({'ok': True, 'active': mid})


# ---------------------------------------------------------------------------
# Engine + playback
# ---------------------------------------------------------------------------
def new_agg():
    """Running totals that feed the Executive Summary. Updated once per
    packet, so the summary never has to re-scan the full log."""
    return {
        'total': 0, 'blocked': 0, 'allowed': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
        'rule_agree': 0, 'rule_tp': 0, 'rule_fp': 0, 'held': 0, 'conf_sum': 0.0,
        'by_label': {}, 'sev': {}, 'mitre': {},
    }


def new_session_state(username=None, delay=0.1133):
    return {
        'is_paused': False, 'is_stopped': False,
        'logs': [], 'stats': {}, 'agg': new_agg(),
        'username': username,
        'packet_delay': delay,  # 0.1133 matches the default 5x speed-slider value
    }


def init_engine():
    """Loads data + model + runs batched inference ONCE for the whole
    server. Every session's playback reads this same array afterward."""
    global engine_started
    with engine_lock:
        if engine_started:
            return
        engine_started = True

    print("[AdaptiveShield] Initializing shared engine...")
    cached = load_cache(); from_cache = False
    if cached:
        X_train, X_test, y_train, y_test, classes = cached; from_cache = True
    else:
        print("[AdaptiveShield] Running full preprocessing...")
        X_train, X_test, y_train, y_test, scaler, le = preprocess()
        classes = list(le.classes_); save_cache(X_train, X_test, y_train, y_test, classes)

    registry.set_data({'X_train': X_train, 'X_test': X_test, 'y_train': y_train,
                       'y_test': y_test, 'classes': classes})
    registry.discover()
    model_id = pick_default_model()
    if model_id is None:
        msg = f"No DQN model (.keras / .h5) found in '{config.MODELS_DIR}'. Add one or run python src/train.py."
        print(f"[AdaptiveShield] ERROR: {msg}")
        socketio.emit('load_error', {'message': msg})
        with engine_lock:
            engine_started = False
        return

    total_test_size = len(X_test)
    print(f"[AdaptiveShield] Loading model '{model_id}' and running batched inference on {total_test_size:,} packets...")
    t0 = time.time()
    try:
        entry = registry.load(model_id)
    except Exception as e:
        print(f"[AdaptiveShield] ERROR loading model '{model_id}': {e}")
        socketio.emit('load_error', {'message': str(e)})
        with engine_lock:
            engine_started = False
        return
    all_q_vals = entry['q_vals']
    print(f"[AdaptiveShield] Batched inference done in {time.time()-t0:.2f}s")

    # Flow inspector data: recover approximate human-readable (pre-scaling)
    # feature values via the fitted scaler's inverse_transform. Falls back
    # gracefully if the scaler/feature-order artifacts aren't present yet.
    feature_names = None
    X_test_raw = None
    baseline_mean = None
    baseline_std = None
    try:
        scaler = joblib.load(config.SCALER_PATH)
        with open(config.FEATURE_ORDER_PATH) as f:
            feature_names = json.load(f)
        X_test_raw = scaler.inverse_transform(X_test)
        # Baseline distribution for drift detection and the rule-based
        # shadow comparison: mean/std per feature over the evaluation set.
        baseline_mean = X_test_raw.mean(axis=0)
        baseline_std = X_test_raw.std(axis=0)
        baseline_std[baseline_std == 0] = 1  # avoid divide-by-zero for constant features
        print(f"[AdaptiveShield] Flow inspector: recovered raw feature values for {len(feature_names)} fields")
    except Exception as e:
        print(f"[AdaptiveShield] Flow inspector unavailable (scaler/feature-order not found): {e}")

    # Lightweight model health check: accuracy of the already-computed
    # batched predictions, plus training metadata from the checkpoint file.
    predicted = np.argmax(all_q_vals, axis=1)
    true_binary = (y_test != 0).astype(int)
    health_accuracy = float(np.mean(predicted == true_binary))
    checkpoint_info = None
    try:
        with open(os.path.join(config.MODELS_DIR, "checkpoint.json")) as f:
            checkpoint_info = json.load(f)
    except Exception:
        pass

    with engine_lock:
        engine.update({
            'ready': True,
            'ready_data': {'total_packets': total_test_size, 'from_cache': from_cache},
            'X_test': X_test, 'y_test': y_test, 'classes': classes,
            'all_q_vals': all_q_vals, 'total_test_size': total_test_size,
            'feature_names': feature_names, 'X_test_raw': X_test_raw,
            'baseline_mean': baseline_mean, 'baseline_std': baseline_std,
            'model': entry['model'],  # kept for on-demand attribution lookups
            'model_id': model_id,
            'health': {
                'model_path': entry['path'],
                'params': entry['params'],
                'accuracy': health_accuracy,
                'total_test': total_test_size,
                'checkpoint': checkpoint_info,
            },
        })
    print(f"[AdaptiveShield] Engine ready — {total_test_size:,} packets")


def analyze_flow(features, baseline_mean, baseline_std, feature_names):
    """Computes three related things from ONE pass over a flow's features:
    a rule-based shadow decision (a simple z-score threshold heuristic,
    representative of the static thresholds a legacy firewall would use),
    a drift score (how many features are statistically unusual vs the
    baseline), and the specific features driving that unusualness.
    Dataset-agnostic -- works from statistics, not hardcoded feature names."""
    if not features or baseline_mean is None or feature_names is None:
        return {'rule_decision': 'ALLOW', 'drift_score': 0, 'top_drift': []}
    zscores = []
    for i, name in enumerate(feature_names):
        val = features.get(name)
        if val is None:
            continue
        z = (val - baseline_mean[i]) / baseline_std[i]
        zscores.append((name, float(z)))
    zscores.sort(key=lambda p: abs(p[1]), reverse=True)
    anomalous = sum(1 for _, z in zscores if abs(z) > 2.5)
    rule_decision = 'BLOCK' if anomalous >= 2 else 'ALLOW'
    return {
        'rule_decision': rule_decision,
        'drift_score': anomalous,
        'top_drift': [{'feature': n, 'zscore': round(z, 2)} for n, z in zscores[:3]],
    }


def build_narrative(label, decision, top_drift):
    """Plain-English one-liner built from the same drift analysis."""
    if decision == 'ALLOW':
        return "Allowed — flow statistics fall within the normal range for benign traffic."
    if top_drift:
        reasons = " and ".join(d['feature'].lower() for d in top_drift[:2])
        return f"Blocked — unusually high {reasons}, consistent with {label}."
    return f"Blocked — classified as {label}."


def run_session(sid, username):
    """Independent playback loop for ONE connected client, over the
    shared read-only engine data. Each session has its own pause/stop
    flags, own logs, and own speed."""
    st = sessions[sid]
    agg = st['agg']
    classes = engine['classes']; y_test = engine['y_test']
    all_q_vals = engine['all_q_vals']; total_test_size = engine['total_test_size']
    feature_names = engine.get('feature_names'); X_test_raw = engine.get('X_test_raw')
    baseline_mean = engine.get('baseline_mean'); baseline_std = engine.get('baseline_std')

    total = blocked = allowed = fp = fn = correct = tp = tn = 0
    accuracy = precision = recall = f1 = 0.0

    for i in range(total_test_size):
        if st['is_stopped']:
            break
        while st['is_paused'] and not st['is_stopped']:
            time.sleep(0.2)
        if st['is_stopped']:
            break

        true_label = int(y_test[i]); is_attack = int(true_label != 0)
        q_vals = all_q_vals[i]
        action = int(np.argmax(q_vals))
        raw_conf = abs(float(q_vals[1] - q_vals[0])) / (abs(float(q_vals[0])) + abs(float(q_vals[1])) + 1e-9)
        confidence = min(99, max(1, round(raw_conf * 100, 1)))
        label_name = classes[true_label] if true_label < len(classes) else "Unknown"
        cvss_sev, cvss_score, cvss_clr = get_cvss(label_name)
        mitre_id, mitre_name = get_mitre(label_name)

        if action == 1 and is_attack == 1:   reward = +1.0; tp += 1
        elif action == 0 and is_attack == 0: reward = +0.5; tn += 1
        elif action == 0 and is_attack == 1: reward = -1.0; fn += 1
        else:                                reward = -0.5; fp += 1

        total += 1
        blocked += 1 if action == 1 else 0
        allowed += 1 if action == 0 else 0
        if reward > 0: correct += 1
        accuracy = round((correct / total) * 100, 2)
        precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0.0
        fpr_live = round(fp / (fp + tn) * 100, 2) if (fp + tn) > 0 else 0.0
        cpu = round(psutil.cpu_percent(), 1); mem = round(psutil.virtual_memory().percent, 1)

        # Flow inspector payload: real (inverse-scaled) feature values for
        # this packet. Empty when the scaler artifacts weren't available.
        features = {}
        if feature_names is not None and X_test_raw is not None:
            features = {name: round(float(val), 4) for name, val in zip(feature_names, X_test_raw[i])}

        analysis = analyze_flow(features, baseline_mean, baseline_std, feature_names)
        decision_str = 'BLOCK' if action == 1 else 'ALLOW'
        narrative = build_narrative(label_name, decision_str, analysis['top_drift'])

        # Tiered confidence routing: below the review threshold, the decision
        # is held for human review instead of auto-enforced.
        if confidence < review_threshold():
            db.add_review_item(total, sid, label_name, confidence, decision_str)
            agg['held'] += 1

        # Running totals for the Executive Summary.
        lab = agg['by_label'].setdefault(label_name, {'seen': 0, 'blocked': 0, 'missed': 0, 'attack': bool(is_attack)})
        lab['seen'] += 1
        lab['blocked'] += 1 if action == 1 else 0
        lab['missed'] += 1 if (is_attack and action == 0) else 0
        if is_attack:
            sv = agg['sev'].setdefault(cvss_sev, {'seen': 0, 'caught': 0})
            sv['seen'] += 1
            sv['caught'] += 1 if action == 1 else 0
            if mitre_id != '--':
                mt = agg['mitre'].setdefault(mitre_id, {'name': mitre_name, 'seen': 0})
                mt['seen'] += 1
            if analysis['rule_decision'] == 'BLOCK':
                agg['rule_tp'] += 1
        elif analysis['rule_decision'] == 'BLOCK':
            agg['rule_fp'] += 1
        agg['rule_agree'] += 1 if analysis['rule_decision'] == decision_str else 0
        agg['conf_sum'] += confidence
        agg.update(total=total, blocked=blocked, allowed=allowed, tp=tp, fp=fp, fn=fn, tn=tn)

        row = {'packet_id': total, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
               'label': label_name, 'is_attack': is_attack, 'action': action,
               'decision': decision_str, 'reward': reward,
               'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
               'confidence': confidence, 'cvss_severity': cvss_sev, 'cvss_score': cvss_score}
        st['logs'].append(row)
        db.log_decision(sid, username, row)  # persisted -- survives restarts
        # NOTE: `features` is intentionally NOT persisted to SQLite -- it would
        # bloat the database for what is a demo-inspection feature.

        if total % 10 == 0:
            socketio.emit('update', {
                'total': total, 'blocked': blocked, 'allowed': allowed, 'fp': fp, 'fn': fn, 'tp': tp,
                'fpr': fpr_live,
                'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
                'last_action': decision_str,
                'last_label': label_name, 'last_reward': reward,
                'confidence': confidence, 'cvss_info': [cvss_sev, cvss_score, cvss_clr],
                'mitre_info': [mitre_id, mitre_name],
                'cpu': cpu, 'mem': mem, 'packet_id': total, 'features': features,
                'q_allow': round(float(q_vals[0]), 4), 'q_block': round(float(q_vals[1]), 4),
                'rule_decision': analysis['rule_decision'], 'drift_score': analysis['drift_score'],
                'top_drift': analysis['top_drift'], 'narrative': narrative}, room=sid)
        time.sleep(st['packet_delay'])

    if not st['is_stopped']:
        tb = sum(1 for y in y_test if y == 0)
        fpr = round((fp / tb) * 100, 2) if tb > 0 else 0.0
        st['stats'] = {'total': total, 'blocked': blocked, 'allowed': allowed, 'tp': tp, 'fp': fp, 'fn': fn,
                       'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr}
        socketio.emit('done', st['stats'], room=sid)
        print(f"[AdaptiveShield] Session {sid[:8]} done — Acc:{accuracy}% F1:{f1}% FPR:{fpr}% Logs:{len(st['logs']):,}")


@socketio.on('connect')
def on_connect():
    if not store.get_user(session.get('username')):
        return False  # reject unauthenticated socket connections
    sid = request.sid
    username = session.get('username')  # capture NOW, while still in a valid
                                        # request context -- session isn't
                                        # accessible from a spawned thread
    with sessions_lock:
        sessions[sid] = new_session_state(username)
    db.log_access(username, 'connect', sid)

    def start_when_ready():
        if not engine['ready']:
            threading.Thread(target=init_engine, daemon=True).start()
        while not engine['ready']:
            time.sleep(0.2)
        socketio.emit('ready', engine['ready_data'], room=sid)
        threading.Thread(target=run_session, args=(sid, username), daemon=True).start()
    threading.Thread(target=start_when_ready, daemon=True).start()


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    with sessions_lock:
        st = sessions.pop(sid, None)  # drop finished/abandoned sessions so memory doesn't grow
    if st:
        st['is_stopped'] = True  # cleanly stop this client's own playback thread
    db.log_access(session.get('username'), 'disconnect', sid)


@socketio.on('control')
def on_control(cmd):
    sid = request.sid
    # Viewers are read-only -- only admin/analyst roles may pause, stop, or restart.
    if current_role() not in ('admin', 'analyst'):
        return
    with sessions_lock:
        st = sessions.get(sid)
    if not st:
        return
    if cmd == 'pause':
        st['is_paused'] = True; print(f"[AdaptiveShield] {sid[:8]} paused")
    elif cmd == 'resume':
        st['is_paused'] = False; print(f"[AdaptiveShield] {sid[:8]} resumed")
    elif cmd == 'stop':
        st['is_stopped'] = True; st['is_paused'] = False; print(f"[AdaptiveShield] {sid[:8]} stopped")
    elif cmd == 'restart':
        username = session.get('username')
        with sessions_lock:
            old = sessions.get(sid)
            if old:
                old['is_stopped'] = True  # make sure the previous playback thread exits
                old['is_paused'] = False
            sessions[sid] = new_session_state(username, delay=old['packet_delay'] if old else 0.1133)
            st = sessions[sid]
        print(f"[AdaptiveShield] {sid[:8]} restarting...")
        threading.Thread(target=run_session, args=(sid, username), daemon=True).start()
    emit_state(sid, st)
    db.log_access(session.get('username'), f'control:{cmd}', sid)


@socketio.on('set_speed')
def on_set_speed(val):
    sid = request.sid
    with sessions_lock:
        st = sessions.get(sid)
    if st:
        st['packet_delay'] = round(0.2 - (val - 1) * (0.195) / 9, 4)
        print(f"[AdaptiveShield] {sid[:8]} speed {val}× — delay {st['packet_delay']}s")


if __name__ == "__main__":
    bootstrap()
    print(f"[AdaptiveShield] Serving on http://{config.HOST}:{config.PORT}")
    socketio.run(app, host=config.HOST, port=config.PORT, debug=False, allow_unsafe_werkzeug=True)
