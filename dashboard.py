import sys
import os
import threading
import time
import csv
import io
import json
import joblib
import psutil
import numpy as np
import tensorflow as tf
from functools import wraps
from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for, session
from flask_socketio import SocketIO

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.preprocess import preprocess
from src.dqn_agent import DQNAgent
from src import db

app = Flask(__name__)
# FIX (foundation phase): a real secret key is required for Flask's signed
# session cookies (used for login) to work at all. Set ADAPTIVESHIELD_SECRET
# in your environment before any real deployment -- this fallback is fine
# for local demo use only.
app.secret_key = os.environ.get("ADAPTIVESHIELD_SECRET", "dev-secret-change-before-real-deployment")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

SAVE_PATH = "models/dqn_firewall.keras"
CACHE_DIR  = "data/cache"
REVIEW_THRESHOLD = 40  # confidence % below which a decision is held for human review

# FIX (foundation phase): placeholder credential store. Replace with a real
# hashed-password user table before any real deployment -- this exists so
# role-based access control (admin/analyst/viewer) has something to check
# against during this build phase.
USERS = {
    "admin":   {"password": "adaptiveshield", "role": "admin"},
    "analyst": {"password": "analyst123",     "role": "analyst"},
    "viewer":  {"password": "viewer123",      "role": "viewer"},
}

LOGIN_HTML = """
<!DOCTYPE html><html><head><title>AdaptiveShield Login</title>
<style>body{font-family:sans-serif;background:#0f172a;color:#e2e8f0;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0}
.box{background:#1e293b;padding:2rem;border-radius:8px;width:300px}
input{width:100%;padding:0.5rem;margin:0.5rem 0;border-radius:4px;border:none;box-sizing:border-box}
button{width:100%;padding:0.6rem;background:#2563eb;color:white;border:none;
border-radius:4px;cursor:pointer;font-weight:bold}
.err{color:#f87171;font-size:0.9rem}
</style></head><body>
<form class="box" method="POST">
<h2>AdaptiveShield</h2>
{% if error %}<p class="err">{{ error }}</p>{% endif %}
<input name="username" placeholder="Username" required>
<input name="password" type="password" placeholder="Password" required>
<button type="submit">Sign In</button>
</form></body></html>
"""


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper

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

# ---- Per-session playback/control state, keyed by Socket.IO sid. FIX
# (foundation phase): this replaces the original global is_paused/
# is_stopped/logs/session_stats/packet_delay variables, which were
# shared across every connected client -- meaning two people watching
# the dashboard at once saw and controlled the exact same simulation.
# Each session below gets its own independent playback position,
# pause/stop flags, and logs, all running over the SAME shared
# read-only engine data above. ----
sessions = {}
sessions_lock = threading.Lock()

# FIX: tracks whether the model/dataset are loaded and the 'ready' event has
# already been broadcast once, so any client that connects/reconnects late
# (e.g. a page refresh mid-demo) gets caught up instead of hanging forever
# on the loading screen.
engine_ready       = False
engine_ready_data  = {}

CVSS_MAP = {
    'DDoS':('CRITICAL',9.8,'#dc2626'),
    'DoS Hulk':('CRITICAL',9.1,'#dc2626'),
    'DoS GoldenEye':('HIGH',8.6,'#ea580c'),
    'DoS Slowloris':('HIGH',7.5,'#ea580c'),
    'DoS Slowhttptest':('HIGH',7.5,'#ea580c'),
    'PortScan':('MEDIUM',5.3,'#d97706'),
    'FTP-Patator':('HIGH',8.1,'#ea580c'),
    'SSH-Patator':('HIGH',8.1,'#ea580c'),
    'Web Attack Brute Force':('HIGH',7.3,'#ea580c'),
    'Web Attack XSS':('MEDIUM',6.1,'#d97706'),
    'Web Attack Sql Injection':('CRITICAL',9.8,'#dc2626'),
    'Infiltration':('CRITICAL',9.5,'#dc2626'),
    'Bot':('HIGH',8.8,'#ea580c'),
    'Heartbleed':('CRITICAL',9.8,'#dc2626'),
    'BENIGN':('NONE',0.0,'#059669'),
}
def get_cvss(label):
    for k,v in CVSS_MAP.items():
        if k.lower() in label.lower(): return v
    return ('MEDIUM',5.0,'#d97706')

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
        if k.lower() in label.lower(): return v
    return ('--', 'No direct ATT&CK mapping')

def save_cache(X_train,X_test,y_train,y_test,classes):
    os.makedirs(CACHE_DIR,exist_ok=True)
    np.save(f"{CACHE_DIR}/X_train.npy",X_train)
    np.save(f"{CACHE_DIR}/X_test.npy",X_test)
    np.save(f"{CACHE_DIR}/y_train.npy",y_train)
    np.save(f"{CACHE_DIR}/y_test.npy",y_test)
    np.save(f"{CACHE_DIR}/classes.npy",np.array(classes))
    print("[AdaptiveShield] Cache saved.")

def load_cache():
    files=['X_train','X_test','y_train','y_test','classes']
    if all(os.path.exists(f"{CACHE_DIR}/{f}.npy") for f in files):
        X_train=np.load(f"{CACHE_DIR}/X_train.npy")
        X_test=np.load(f"{CACHE_DIR}/X_test.npy")
        y_train=np.load(f"{CACHE_DIR}/y_train.npy")
        y_test=np.load(f"{CACHE_DIR}/y_test.npy")
        classes=list(np.load(f"{CACHE_DIR}/classes.npy",allow_pickle=True))
        print(f"[AdaptiveShield] Cache loaded — {len(X_test):,} packets.")
        return X_train,X_test,y_train,y_test,classes
    return None

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AdaptiveShield</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
:root{
  --base:#14181D;
  --surface:#1C2128;
  --surface-2:#232A33;
  --border:#2C3440;
  --signal-amber:#E8A33D;
  --signal-mint:#5FBFA0;
  --ledger-ink:#7C93B8;
  --alert:#D96B5A;
  --text:#E8E6E1;
  --text-muted:#8B9099;
  --radius:6px;
}
*{box-sizing:border-box; margin:0; padding:0;}
body{
  background:var(--base); color:var(--text);
  font-family:'IBM Plex Sans',sans-serif;
  height:100vh; overflow:hidden;
  display:flex;
}
.mono{font-family:'IBM Plex Mono',monospace;}

/* ---------- Left rail ---------- */
.rail{
  width:220px; flex-shrink:0; background:var(--surface);
  border-right:1px solid var(--border);
  display:flex; flex-direction:column; height:100vh;
}
.rail-brand{
  padding:20px 18px 16px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:10px;
}
.rail-brand .mark{
  width:26px; height:26px; border-radius:7px;
  background:linear-gradient(135deg,var(--signal-amber),var(--signal-mint));
  flex-shrink:0;
}
.rail-brand .name{font-weight:600; font-size:15px; letter-spacing:-0.01em;}
.rail-brand .sub{font-size:11px; color:var(--text-muted); margin-top:1px;}
.rail-nav{flex:1; padding:14px 10px; overflow-y:auto;}
.rail-item{
  display:flex; align-items:center; gap:10px;
  padding:9px 12px; border-radius:var(--radius);
  color:var(--text-muted); font-size:13.5px; cursor:pointer;
  margin-bottom:2px; transition:background .15s, color .15s;
  border:1px solid transparent;
}
.rail-item:hover{background:var(--surface-2); color:var(--text);}
.rail-item.active{background:var(--surface-2); color:var(--text); border-color:var(--border);}
.rail-item .dot{width:6px; height:6px; border-radius:50%; background:var(--text-muted); flex-shrink:0;}
.rail-item.active .dot{background:var(--signal-amber);}
.rail-item .soon{font-size:10px; color:var(--text-muted); margin-left:auto; font-family:'IBM Plex Mono',monospace;}
.rail-foot{
  padding:14px 18px; border-top:1px solid var(--border);
  font-size:12.5px; color:var(--text-muted);
}
.rail-foot .who{color:var(--text); font-weight:500;}
.rail-foot a{color:var(--ledger-ink); text-decoration:none; font-size:12px;}

/* ---------- Main area ---------- */
.main{flex:1; display:flex; flex-direction:column; height:100vh; overflow:hidden;}
.topbar{
  padding:16px 28px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; justify-content:space-between;
}
.topbar h1{font-size:18px; font-weight:600; letter-spacing:-0.01em;}
.mode-pill{
  font-size:12px; padding:5px 10px; border-radius:20px;
  background:var(--surface-2); border:1px solid var(--border); color:var(--text-muted);
  display:flex; align-items:center; gap:6px;
}
.mode-pill .led{width:6px;height:6px;border-radius:50%; background:var(--signal-mint);}
.controls{display:flex; gap:8px; align-items:center;}
.btn{
  background:var(--surface-2); border:1px solid var(--border); color:var(--text);
  padding:7px 14px; border-radius:var(--radius); font-size:13px; cursor:pointer;
  font-family:'IBM Plex Sans',sans-serif; transition:border-color .15s;
}
.btn:hover{border-color:var(--signal-amber);}
.btn.primary{background:var(--signal-amber); color:#1a1206; border-color:var(--signal-amber); font-weight:600;}
.btn:disabled{opacity:.4; cursor:not-allowed;}
input[type=range]{accent-color:var(--signal-amber); width:90px;}

.view{flex:1; overflow-y:auto; padding:24px 28px; display:none;}
.view.active{display:block;}

/* ---------- Command Center ---------- */
.pulse-card{
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:20px 22px; margin-bottom:20px;
}
.pulse-head{display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px;}
.pulse-head h2{font-size:14px; font-weight:600; color:var(--text);}
.pulse-head .last{font-size:12px; color:var(--text-muted);}
.pulse-head .last b{font-family:'IBM Plex Mono',monospace; color:var(--signal-amber); font-weight:500;}
canvas#pulse{width:100%; height:140px; display:block;}

.metric-grid{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:20px;}
.metric{
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:16px 18px;
}
.metric .label{font-size:12px; color:var(--text-muted); margin-bottom:6px;}
.metric .value{font-family:'IBM Plex Mono',monospace; font-size:24px; font-weight:500;}
.metric .value.mint{color:var(--signal-mint);}
.metric .value.amber{color:var(--signal-amber);}
.metric .value.alert{color:var(--alert);}

.ledger{background:var(--surface); border:1px solid var(--border); border-radius:8px;}
.ledger-head{padding:14px 18px; border-bottom:1px solid var(--border); font-size:13px; font-weight:600;}
.ledger-body{max-height:280px; overflow-y:auto;}
.ledger-row{
  display:flex; align-items:center; gap:12px; padding:9px 18px;
  border-bottom:1px solid var(--border); font-size:12.5px;
}
.ledger-row:last-child{border-bottom:none;}
.ledger-row .n{font-family:'IBM Plex Mono',monospace; color:var(--text-muted); width:44px;}
.ledger-row .lbl{flex:1; color:var(--text);}
.ledger-row .dec{font-family:'IBM Plex Mono',monospace; font-weight:500; width:60px;}
.ledger-row .dec.block{color:var(--alert);}
.ledger-row .dec.allow{color:var(--signal-mint);}
.ledger-row .conf{font-family:'IBM Plex Mono',monospace; color:var(--text-muted); width:50px; text-align:right;}

/* ---------- Live Feed: two-pane, Wireshark-style ---------- */
.feed-layout{display:flex; gap:16px; height:calc(100vh - 130px);}
.feed-list{flex:1.3; overflow-y:auto; border:1px solid var(--border); border-radius:8px; background:var(--surface);}
.feed-table{width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:12.5px;}
.feed-table thead th{
  text-align:left; padding:9px 14px; border-bottom:1px solid var(--border);
  color:var(--text-muted); font-weight:500; font-size:11.5px; position:sticky; top:0; background:var(--surface);
}
.feed-table tbody td{padding:7px 14px; border-bottom:1px solid var(--border); cursor:pointer;}
.feed-table tbody tr:hover{background:var(--surface-2);}
.feed-table tbody tr.selected{background:var(--surface-2); box-shadow:inset 2px 0 0 var(--signal-amber);}
.feed-table .dec-block{color:var(--alert); font-weight:600;}
.feed-table .dec-allow{color:var(--signal-mint); font-weight:600;}
.feed-table .sev-dot{display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px;}

.inspector{flex:1; overflow-y:auto; border:1px solid var(--border); border-radius:8px; background:var(--surface); display:flex; flex-direction:column;}
.inspector-head{padding:14px 16px; border-bottom:1px solid var(--border);}
.inspector-head h3{font-size:13.5px; font-weight:600; margin-bottom:4px;}
.inspector-head .meta{font-size:11.5px; color:var(--text-muted); font-family:'IBM Plex Mono',monospace;}
.inspector-qvals{display:flex; gap:10px; padding:12px 16px; border-bottom:1px solid var(--border);}
.inspector-qvals .q{flex:1; background:var(--surface-2); border-radius:6px; padding:8px 10px;}
.inspector-qvals .q .l{font-size:10.5px; color:var(--text-muted); margin-bottom:2px;}
.inspector-qvals .q .v{font-family:'IBM Plex Mono',monospace; font-size:15px;}
.inspector-fields{flex:1; overflow-y:auto; padding:4px 0;}
.field-row{
  display:flex; justify-content:space-between; padding:5px 16px;
  font-family:'IBM Plex Mono',monospace; font-size:11.5px; border-bottom:1px solid var(--border);
}
.field-row:hover{background:var(--surface-2);}
.field-row .fname{color:var(--ledger-ink);}
.field-row .fval{color:var(--text);}
.inspector-empty{
  flex:1; display:flex; align-items:center; justify-content:center;
  color:var(--text-muted); font-size:13px; text-align:center; padding:20px;
}
.mitre-badge{
  display:inline-flex; align-items:center; gap:5px; margin-top:6px;
  font-size:11px; font-family:'IBM Plex Mono',monospace; color:var(--ledger-ink);
  background:var(--surface-2); padding:3px 8px; border-radius:4px;
}
.insp-export{
  font-size:11.5px; color:var(--text-muted); background:none; border:1px solid var(--border);
  border-radius:5px; padding:4px 9px; cursor:pointer; float:right; margin-top:-2px;
}
.insp-export:hover{border-color:var(--signal-amber); color:var(--text);}

/* ---------- Feed search/filter toolbar ---------- */
.feed-toolbar{display:flex; gap:8px; margin-bottom:12px;}
.feed-toolbar input, .feed-toolbar select{
  background:var(--surface); border:1px solid var(--border); color:var(--text);
  padding:7px 10px; border-radius:6px; font-size:12.5px; font-family:'IBM Plex Sans',sans-serif;
}
.feed-toolbar input{flex:1;}

/* ---------- Inspector sub-tabs ---------- */
.insp-tabs{display:flex; border-bottom:1px solid var(--border);}
.insp-tab{
  padding:9px 14px; font-size:12px; color:var(--text-muted); cursor:pointer;
  border-bottom:2px solid transparent;
}
.insp-tab:hover{color:var(--text);}
.insp-tab.active{color:var(--text); border-bottom-color:var(--signal-amber);}
.insp-pane{display:none; flex:1; overflow-y:auto;}
.insp-pane.active{display:block;}

/* ---------- Attribution bars ---------- */
.attr-row{padding:8px 16px; border-bottom:1px solid var(--border);}
.attr-row .attr-label{
  display:flex; justify-content:space-between; font-size:11.5px;
  font-family:'IBM Plex Mono',monospace; margin-bottom:4px;
}
.attr-row .attr-label .fname{color:var(--ledger-ink);}
.attr-bar-track{height:5px; background:var(--surface-2); border-radius:3px; overflow:hidden;}
.attr-bar-fill{height:100%; border-radius:3px;}
.attr-bar-fill.pos{background:var(--signal-amber);}
.attr-bar-fill.neg{background:var(--ledger-ink);}

/* ---------- Similar flows ---------- */
.similar-row{
  display:flex; justify-content:space-between; padding:9px 16px;
  border-bottom:1px solid var(--border); font-size:12px; cursor:pointer;
}
.similar-row:hover{background:var(--surface-2);}
.similar-row .sim-dist{font-family:'IBM Plex Mono',monospace; color:var(--text-muted); font-size:11px;}

/* ---------- Notes ---------- */
.notes-pane{padding:12px 16px; display:flex; flex-direction:column; gap:10px;}
.notes-list{display:flex; flex-direction:column; gap:8px;}
.note-item{background:var(--surface-2); border-radius:6px; padding:8px 10px; font-size:12px;}
.note-item .note-meta{color:var(--text-muted); font-size:10.5px; margin-bottom:3px; font-family:'IBM Plex Mono',monospace;}
.notes-input{display:flex; gap:6px;}
.notes-input textarea{
  flex:1; background:var(--surface-2); border:1px solid var(--border); color:var(--text);
  border-radius:6px; padding:7px 9px; font-size:12px; font-family:'IBM Plex Sans',sans-serif; resize:vertical; min-height:50px;
}

/* ---------- Attack timeline strip ---------- */
.timeline-card{background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px 16px; margin-bottom:16px;}
.timeline-card .t-head{font-size:12px; color:var(--text-muted); margin-bottom:8px;}
.timeline-strip{display:flex; align-items:flex-end; gap:2px; height:28px; overflow:hidden;}
.t-tick{width:4px; border-radius:1px; flex-shrink:0; cursor:pointer;}

/* ---------- Feature-space scatter ---------- */
.scatter-card{background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; margin-top:16px;}
.scatter-card .s-head{font-size:13px; font-weight:600; margin-bottom:8px;}
.scatter-card .s-sub{font-size:11px; color:var(--text-muted); margin-bottom:8px; font-family:'IBM Plex Mono',monospace;}

/* ---------- Confidence & Drift page ---------- */
.conf-grid{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px;}
.conf-card{background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px;}
.conf-card h3{font-size:13px; font-weight:600; margin-bottom:8px;}
.drift-list{display:flex; flex-direction:column; gap:6px;}
.drift-row{display:flex; justify-content:space-between; font-size:12px; font-family:'IBM Plex Mono',monospace; padding:6px 0; border-bottom:1px solid var(--border);}
.drift-row .dz{color:var(--alert);}
.shadow-compare{display:flex; gap:10px; margin-top:6px;}
.shadow-box{flex:1; background:var(--surface-2); border-radius:6px; padding:10px; text-align:center;}
.shadow-box .sl{font-size:10.5px; color:var(--text-muted); margin-bottom:4px;}
.shadow-box .sv{font-family:'IBM Plex Mono',monospace; font-size:15px; font-weight:600;}
.shadow-box.agree .sv{color:var(--signal-mint);}
.shadow-box.disagree .sv{color:var(--alert);}
.narrative-box{background:var(--surface-2); border-left:3px solid var(--signal-amber); border-radius:4px; padding:10px 12px; font-size:12.5px; margin-top:10px;}

/* ---------- Model Health page ---------- */
.health-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:14px;}
.health-card{background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px 18px;}
.health-card .hl{font-size:12px; color:var(--text-muted); margin-bottom:6px;}
.health-card .hv{font-family:'IBM Plex Mono',monospace; font-size:20px;}

/* ---------- Review Queue & Audit Log ---------- */
.review-item{background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; margin-bottom:10px;}
.review-item .ri-head{display:flex; justify-content:space-between; margin-bottom:8px; font-size:13px;}
.review-item .ri-meta{font-size:11.5px; color:var(--text-muted); font-family:'IBM Plex Mono',monospace; margin-bottom:10px;}
.review-actions{display:flex; gap:8px; align-items:center;}
.review-actions input{flex:1; background:var(--surface-2); border:1px solid var(--border); color:var(--text); border-radius:5px; padding:6px 9px; font-size:12px;}
.btn-approve{background:var(--signal-mint); color:#0a1f18; border:none;}
.btn-override{background:var(--alert); color:#2a0a06; border:none;}
.audit-toolbar{display:flex; gap:8px; margin-bottom:12px;}
.audit-table{width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:12px;}
.audit-table th{text-align:left; padding:8px 12px; border-bottom:1px solid var(--border); color:var(--text-muted); font-weight:500; font-size:11px;}
.audit-table td{padding:6px 12px; border-bottom:1px solid var(--border);}

/* ---------- Empty / building states ---------- */
.building{
  height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center;
  color:var(--text-muted); text-align:center; gap:8px;
}
.building .icon{width:40px; height:40px; border-radius:8px; background:var(--surface); border:1px dashed var(--border); margin-bottom:6px;}
.building h3{color:var(--text); font-size:15px; font-weight:600;}
.building p{font-size:13px; max-width:340px; line-height:1.5;}

.loading-screen{
  position:fixed; inset:0; background:var(--base); display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:14px; z-index:100;
}
.loading-screen .spin{
  width:32px; height:32px; border-radius:50%; border:3px solid var(--border);
  border-top-color:var(--signal-amber); animation:spin 0.8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg);}}
.loading-screen p{color:var(--text-muted); font-size:13px;}
</style>
</head>
<body>

<div id="loading" class="loading-screen">
  <div class="spin"></div>
  <p id="loading-text">Loading model and preparing session replay&hellip;</p>
</div>

<div class="rail">
  <div class="rail-brand">
    <div class="mark"></div>
    <div><div class="name">AdaptiveShield</div><div class="sub">Session Replay</div></div>
  </div>
  <div class="rail-nav" id="nav">
    <div class="rail-item active" data-view="command"><span class="dot"></span>Command Center</div>
    <div class="rail-item" data-view="feed"><span class="dot"></span>Live Feed</div>
    <div class="rail-item" data-view="confidence"><span class="dot"></span>Confidence & Drift</div>
    <div class="rail-item" data-view="review"><span class="dot"></span>Review Queue</div>
    <div class="rail-item" data-view="audit"><span class="dot"></span>Audit Log</div>
    <div class="rail-item" data-view="health"><span class="dot"></span>Model Health</div>
    <div class="rail-item" data-view="soon-admin"><span class="dot"></span>Admin Controls<span class="soon">building</span></div>
    <div class="rail-item" data-view="soon-exec"><span class="dot"></span>Executive Summary<span class="soon">building</span></div>
  </div>
  <div class="rail-foot">
    <div class="who">{{ username }}</div>
    <div style="margin-bottom:8px; text-transform:capitalize;">{{ role }}</div>
    <a href="/logout">Sign out</a>
  </div>
</div>

<div class="main">
  <div class="topbar">
    <div style="display:flex; align-items:center; gap:14px;">
      <h1 id="view-title">Command Center</h1>
      <div class="mode-pill"><span class="led"></span>Session replay &middot; <span id="pkt-count" class="mono">0</span> packets</div>
    </div>
    <div class="controls">
      {% if role in ['admin','analyst'] %}
      <button class="btn" id="btn-pause">Pause</button>
      <button class="btn" id="btn-restart">Restart</button>
      <button class="btn primary" id="btn-stop">Stop</button>
      <input type="range" id="speed" min="1" max="10" value="5">
      {% else %}
      <span style="font-size:12px;color:var(--text-muted);">Viewer &mdash; read only</span>
      {% endif %}
      <a href="/download-logs" class="btn">Export CSV</a>
    </div>
  </div>

  <div class="view active" id="view-command">
    <div class="pulse-card">
      <div class="pulse-head">
        <h2>Live decision pulse</h2>
        <div class="last">last: <b id="last-label">&mdash;</b> &rarr; <b id="last-decision">&mdash;</b></div>
      </div>
      <canvas id="pulse"></canvas>
    </div>
    <div class="metric-grid">
      <div class="metric"><div class="label">Accuracy</div><div class="value mint" id="m-accuracy">&mdash;</div></div>
      <div class="metric"><div class="label">False positive rate</div><div class="value amber" id="m-fpr">&mdash;</div></div>
      <div class="metric"><div class="label">F1-score</div><div class="value" id="m-f1">&mdash;</div></div>
      <div class="metric"><div class="label">Blocked / allowed</div><div class="value" id="m-ba">&mdash;</div></div>
    </div>
    <div class="ledger">
      <div class="ledger-head">Recent activity</div>
      <div class="ledger-body" id="ledger-body"></div>
    </div>
  </div>

  <div class="view" id="view-feed">
    <div class="timeline-card">
      <div class="t-head">Recent attack timeline &mdash; colored by severity, click a tick to jump to that packet</div>
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
        <div class="inspector-empty" id="inspector-empty">Select a packet from the list to inspect its full flow statistics &mdash; every field the model actually used to make that decision.</div>
        <div id="inspector-content" style="display:none; flex-direction:column; flex:1; overflow:hidden;">
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
              <div class="notes-input">
                <textarea id="note-text" placeholder="Add an investigation note for this packet&hellip;"></textarea>
              </div>
              <button class="btn" id="btn-save-note" style="align-self:flex-start;">Save note</button>
              {% endif %}
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="scatter-card">
      <div class="s-head">Feature-space scatter</div>
      <div class="s-sub">Flow Duration (x) vs Flow Bytes/s (y) &mdash; colored by decision</div>
      <canvas id="scatter" style="width:100%; height:180px;"></canvas>
    </div>
  </div>

  <div class="view" id="view-confidence">
    <div class="conf-grid">
      <div class="conf-card">
        <h3>Confidence margin distribution</h3>
        <canvas id="conf-histogram" style="width:100%; height:140px;"></canvas>
      </div>
      <div class="conf-card">
        <h3>Drift score over time</h3>
        <canvas id="drift-timeline" style="width:100%; height:140px;"></canvas>
        <div class="s-sub" style="margin-top:4px;">Count of features &gt;2.5&sigma; from the evaluation-set baseline, per packet</div>
      </div>
    </div>
    <div class="conf-card" style="margin-bottom:16px;">
      <h3>Shadow-mode comparison &mdash; last packet</h3>
      <div class="s-sub">AdaptiveShield (DQN) vs. a simple z-score threshold heuristic, representative of a legacy static-rule firewall</div>
      <div class="shadow-compare">
        <div class="shadow-box" id="shadow-dqn"><div class="sl">AdaptiveShield</div><div class="sv">&mdash;</div></div>
        <div class="shadow-box" id="shadow-rule"><div class="sl">Rule-based (z-score)</div><div class="sv">&mdash;</div></div>
      </div>
      <div class="narrative-box" id="narrative-box">Waiting for traffic&hellip;</div>
    </div>
    <div class="conf-card">
      <h3>Most drifted features &mdash; last packet</h3>
      <div class="drift-list" id="drift-list"><div style="color:var(--text-muted); font-size:12px;">Waiting for traffic&hellip;</div></div>
    </div>
  </div>

  <div class="view" id="view-health">
    <div class="health-grid" style="margin-bottom:16px;">
      <div class="health-card"><div class="hl">Active model</div><div class="hv" id="health-path" style="font-size:13px;">&mdash;</div></div>
      <div class="health-card"><div class="hl">Parameters</div><div class="hv" id="health-params">&mdash;</div></div>
      <div class="health-card"><div class="hl">Self-check accuracy</div><div class="hv mint" id="health-accuracy" style="color:var(--signal-mint);">&mdash;</div></div>
    </div>
    <div class="health-grid">
      <div class="health-card"><div class="hl">Evaluation set size</div><div class="hv" id="health-testsize">&mdash;</div></div>
      <div class="health-card"><div class="hl">Training episodes (checkpoint)</div><div class="hv" id="health-episodes">&mdash;</div></div>
      <div class="health-card"><div class="hl">Training steps (checkpoint)</div><div class="hv" id="health-steps">&mdash;</div></div>
    </div>
  </div>

  <div class="view" id="view-review">
    <div class="s-sub" style="margin-bottom:12px;">Decisions below {{ 40 }}% confidence are held here instead of auto-enforced &mdash; approve to confirm the model's default action, or override with a reason.</div>
    <div id="review-list"><div style="color:var(--text-muted); font-size:12px;">Loading&hellip;</div></div>
  </div>

  <div class="view" id="view-audit">
    <div class="audit-toolbar">
      <input type="text" id="audit-label-filter" placeholder="Filter by label&hellip;" style="flex:1;">
      <select id="audit-decision-filter">
        <option value="">All decisions</option>
        <option value="BLOCK">Blocked only</option>
        <option value="ALLOW">Allowed only</option>
      </select>
      <button class="btn" id="btn-audit-search">Search</button>
    </div>
    <div style="overflow-x:auto;">
      <table class="audit-table">
        <thead><tr><th>#</th><th>Timestamp</th><th>Session</th><th>User</th><th>Label</th><th>Decision</th><th>Confidence</th></tr></thead>
        <tbody id="audit-body"></tbody>
      </table>
    </div>
  </div>

  <div class="view" id="view-building">
    <div class="building">
      <div class="icon"></div>
      <h3>This page is still being built</h3>
      <p>This section is part of the full AdaptiveShield build and isn't wired up yet. Everything except Admin Controls and Executive Summary is now functional &mdash; those two are coming next.</p>
    </div>
  </div>
</div>

<script>
const socket = io();
const pulseCanvas = document.getElementById('pulse');
const ctx = pulseCanvas.getContext('2d');
let pulseData = [];
const MAX_POINTS = 80;

function resizeCanvas(){
  pulseCanvas.width = pulseCanvas.offsetWidth * devicePixelRatio;
  pulseCanvas.height = pulseCanvas.offsetHeight * devicePixelRatio;
}
window.addEventListener('resize', resizeCanvas);

function drawPulse(){
  resizeCanvas();
  const w = pulseCanvas.width, h = pulseCanvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#2C3440'; ctx.lineWidth = 1;
  for(let i=1;i<4;i++){ const y = h/4*i; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
  if(pulseData.length < 2) return;
  ctx.beginPath(); ctx.lineWidth = 2 * devicePixelRatio; ctx.strokeStyle = '#E8A33D';
  pulseData.forEach((v,i)=>{
    const x = (i/(MAX_POINTS-1))*w;
    const y = h/2 - v*(h*0.4);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
}

function addLedgerRow(label, decision, confidence, n){
  const body = document.getElementById('ledger-body');
  const row = document.createElement('div');
  row.className = 'ledger-row';
  row.innerHTML = `<span class="n">#${n}</span><span class="lbl">${label}</span><span class="dec ${decision==='BLOCK'?'block':'allow'}">${decision}</span><span class="conf">${confidence}%</span>`;
  body.prepend(row);
  while(body.children.length > 12) body.removeChild(body.lastChild);
}

const packetCache = new Map(); // packet_id -> full packet data, for the inspector panel
let currentPacketId = null;

function addFeedRow(pkt){
  const body = document.getElementById('feed-body');
  const row = document.createElement('tr');
  row.dataset.pktId = pkt.packet_id;
  row.dataset.label = pkt.last_label;
  row.dataset.decision = pkt.last_action;
  const sevColor = pkt.cvss_info[2] || '#8B9099';
  row.innerHTML = `<td>${pkt.total}</td><td>${new Date().toLocaleTimeString()}</td>
    <td><span class="sev-dot" style="background:${sevColor}"></span>${pkt.last_label}</td>
    <td class="${pkt.last_action==='BLOCK'?'dec-block':'dec-allow'}">${pkt.last_action}</td>
    <td>${pkt.confidence}%</td><td>${pkt.cvss_info[0]}</td>`;
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
  const strip = document.getElementById('timeline-strip');
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

// ---------- Feature-space scatter ----------
const scatterCanvas = document.getElementById('scatter');
const scatterCtx = scatterCanvas.getContext('2d');
const scatterPoints = [];
function resizeScatter(){
  scatterCanvas.width = scatterCanvas.offsetWidth * devicePixelRatio;
  scatterCanvas.height = scatterCanvas.offsetHeight * devicePixelRatio;
}
window.addEventListener('resize', resizeScatter);
function addScatterPoint(pkt){
  const f = pkt.features || {};
  const x = f['Flow Duration']; const y = f['Flow Bytes/s'];
  if(x === undefined || y === undefined) return; // dataset may not have these exact field names
  scatterPoints.push({x, y, block: pkt.last_action==='BLOCK'});
  if(scatterPoints.length > 300) scatterPoints.shift();
  drawScatter();
}
function drawScatter(){
  resizeScatter();
  const w = scatterCanvas.width, h = scatterCanvas.height;
  scatterCtx.clearRect(0,0,w,h);
  if(scatterPoints.length < 2) return;
  const xs = scatterPoints.map(p=>p.x), ys = scatterPoints.map(p=>p.y);
  const xMax = Math.max(...xs,1), yMax = Math.max(...ys,1);
  scatterPoints.forEach(p=>{
    const px = (p.x/xMax) * (w*0.94) + w*0.03;
    const py = h*0.95 - (p.y/yMax) * (h*0.9);
    scatterCtx.beginPath();
    scatterCtx.arc(px, py, 2.5*devicePixelRatio, 0, Math.PI*2);
    scatterCtx.fillStyle = p.block ? '#D96B5A' : '#5FBFA0';
    scatterCtx.globalAlpha = 0.7;
    scatterCtx.fill();
  });
  scatterCtx.globalAlpha = 1;
}

// ---------- Search / filter ----------
function applyFeedFilter(){
  const text = document.getElementById('feed-search').value.toLowerCase();
  const decisionFilter = document.getElementById('feed-filter-decision').value;
  document.querySelectorAll('#feed-body tr').forEach(row=>{
    const matchesText = !text || row.dataset.label.toLowerCase().includes(text);
    const matchesDecision = !decisionFilter || row.dataset.decision === decisionFilter;
    row.style.display = (matchesText && matchesDecision) ? '' : 'none';
  });
}
document.getElementById('feed-search').addEventListener('input', applyFeedFilter);
document.getElementById('feed-filter-decision').addEventListener('change', applyFeedFilter);

// ---------- Inspector tabs ----------
document.querySelectorAll('.insp-tab').forEach(tab=>{
  tab.addEventListener('click', ()=>{
    document.querySelectorAll('.insp-tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.insp-pane').forEach(p=>p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('pane-' + tab.dataset.pane).classList.add('active');
  });
});

function selectPacket(pktId){
  const pkt = packetCache.get(pktId);
  if(!pkt) return;
  currentPacketId = pktId;
  document.querySelectorAll('.feed-table tbody tr').forEach(r=>r.classList.remove('selected'));
  const row = document.querySelector(`.feed-table tbody tr[data-pkt-id="${pktId}"]`);
  if(row) row.classList.add('selected');

  document.getElementById('inspector-empty').style.display = 'none';
  document.getElementById('inspector-content').style.display = 'flex';
  document.getElementById('insp-label').textContent = pkt.last_label;
  document.getElementById('insp-id').textContent = pkt.packet_id;
  document.getElementById('insp-decision').textContent = pkt.last_action;
  document.getElementById('insp-qallow').textContent = (pkt.q_allow ?? '—');
  document.getElementById('insp-qblock').textContent = (pkt.q_block ?? '—');
  const mitre = pkt.mitre_info || ['--','No mapping'];
  document.getElementById('insp-mitre').textContent = mitre[0] === '--' ? 'No direct ATT&CK mapping' : `${mitre[0]} — ${mitre[1]}`;

  const fieldsEl = document.getElementById('insp-fields');
  fieldsEl.innerHTML = '';
  const features = pkt.features || {};
  const names = Object.keys(features);
  if(names.length === 0){
    fieldsEl.innerHTML = '<div class="field-row"><span class="fname">Flow statistics unavailable</span></div><div class="field-row" style="color:var(--text-muted)">Run `python -m src.preprocess` once to generate the scaler artifacts this panel needs.</div>';
  } else {
    names.forEach(name=>{
      const div = document.createElement('div');
      div.className = 'field-row';
      div.innerHTML = `<span class="fname">${name}</span><span class="fval">${features[name]}</span>`;
      fieldsEl.appendChild(div);
    });
  }

  loadAttribution(pktId);
  loadSimilar(pktId);
  loadNotes(pktId);
}

function loadAttribution(pktId){
  const el = document.getElementById('insp-attribution');
  el.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:12px;">Computing&hellip;</div>';
  fetch(`/api/attribution/${pktId}`).then(r=>r.json()).then(data=>{
    if(data.error){ el.innerHTML = `<div style="padding:16px;color:var(--text-muted);font-size:12px;">${data.error}</div>`; return; }
    const maxAbs = Math.max(...data.attribution.map(a=>Math.abs(a.attribution)), 0.0001);
    el.innerHTML = data.attribution.map(a=>{
      const pct = Math.min(100, Math.abs(a.attribution)/maxAbs*100);
      const cls = a.attribution >= 0 ? 'pos' : 'neg';
      return `<div class="attr-row"><div class="attr-label"><span class="fname">${a.feature}</span><span>${a.attribution.toFixed(4)}</span></div>
        <div class="attr-bar-track"><div class="attr-bar-fill ${cls}" style="width:${pct}%"></div></div></div>`;
    }).join('') + '<div style="padding:10px 16px;color:var(--text-muted);font-size:11px;">Amber = pushed toward this decision. Blue = pushed against it (outweighed by other features).</div>';
  }).catch(()=>{ el.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:12px;">Attribution unavailable.</div>'; });
}

function loadSimilar(pktId){
  const el = document.getElementById('insp-similar');
  el.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:12px;">Searching&hellip;</div>';
  fetch(`/api/similar/${pktId}`).then(r=>r.json()).then(data=>{
    if(data.error){ el.innerHTML = `<div style="padding:16px;color:var(--text-muted);font-size:12px;">${data.error}</div>`; return; }
    el.innerHTML = data.similar.map(s=>
      `<div class="similar-row" onclick="jumpToPacket(${s.packet_id})"><span>#${s.packet_id} &middot; ${s.label}</span><span class="sim-dist">dist ${s.distance}</span></div>`
    ).join('') || '<div style="padding:16px;color:var(--text-muted);font-size:12px;">No comparable flows found.</div>';
  }).catch(()=>{ el.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:12px;">Similarity lookup unavailable.</div>'; });
}
function jumpToPacket(pktId){
  // Similar flows are looked up across the FULL test set, but the feed
  // table only keeps the most recent 200 rows -- a similar packet from
  // earlier in the session may not have a visible row to select. Always
  // safe to show its data in the inspector even if it can't be
  // highlighted in the table.
  if(packetCache.has(pktId)) selectPacket(pktId);
  else alert(`Packet #${pktId} is outside the recent feed window (only the last 200 are kept client-side) -- its data is still in the model's evaluation set, just not currently visible in this list.`);
}

function loadNotes(pktId){
  const el = document.getElementById('notes-list');
  el.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">Loading&hellip;</div>';
  fetch(`/api/notes/${pktId}`).then(r=>r.json()).then(data=>{
    el.innerHTML = (data.notes||[]).map(n=>
      `<div class="note-item"><div class="note-meta">${n.username} &middot; ${n.created_at}</div>${n.note_text}</div>`
    ).join('') || '<div style="color:var(--text-muted);font-size:12px;">No notes yet on this packet.</div>';
  });
}
const saveNoteBtn = document.getElementById('btn-save-note');
if(saveNoteBtn){
  saveNoteBtn.addEventListener('click', ()=>{
    const text = document.getElementById('note-text').value.trim();
    if(!text || !currentPacketId) return;
    fetch(`/api/notes/${currentPacketId}`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({note_text: text, sid: socket.id})
    }).then(r=>r.json()).then(()=>{
      document.getElementById('note-text').value = '';
      loadNotes(currentPacketId);
    });
  });
}

document.getElementById('btn-export-report').addEventListener('click', ()=>{
  if(!currentPacketId) return;
  const pkt = packetCache.get(currentPacketId);
  if(!pkt) return;
  const mitre = pkt.mitre_info || ['--','No mapping'];
  let lines = [
    'AdaptiveShield Investigation Report', '='.repeat(40),
    `Packet: #${pkt.packet_id}`, `Classification: ${pkt.last_label}`,
    `Decision: ${pkt.last_action}`, `Confidence: ${pkt.confidence}%`,
    `CVSS: ${pkt.cvss_info[0]} (${pkt.cvss_info[1]})`,
    `MITRE ATT&CK: ${mitre[0]} — ${mitre[1]}`,
    `Q(ALLOW): ${pkt.q_allow}    Q(BLOCK): ${pkt.q_block}`,
    '', 'Flow statistics:', '-'.repeat(40),
  ];
  Object.entries(pkt.features||{}).forEach(([k,v])=> lines.push(`${k}: ${v}`));
  const blob = new Blob([lines.join('\n')], {type:'text/plain'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `adaptiveshield_packet_${pkt.packet_id}_report.txt`;
  a.click();
});

socket.on('ready', (data)=>{
  document.getElementById('loading').style.display = 'none';
  document.getElementById('pkt-count').textContent = data.total_packets.toLocaleString();
});

socket.on('load_error', (data)=>{
  document.getElementById('loading-text').textContent = 'Error loading model: ' + data.message;
});

socket.on('update', (d)=>{
  pulseData.push(d.last_reward > 0 ? 1 : -1);
  if(pulseData.length > MAX_POINTS) pulseData.shift();
  drawPulse();
  document.getElementById('last-label').textContent = d.last_label;
  document.getElementById('last-decision').textContent = d.last_action;
  document.getElementById('m-accuracy').textContent = d.accuracy + '%';
  document.getElementById('m-f1').textContent = d.f1 + '%';
  document.getElementById('m-ba').textContent = d.blocked + ' / ' + d.allowed;
  addLedgerRow(d.last_label, d.last_action, d.confidence, d.total);
  addFeedRow(d);
  updateShadowAndDrift(d);
});

socket.on('done', (stats)=>{
  document.getElementById('m-fpr').textContent = stats.fpr + '%';
});

document.querySelectorAll('.rail-item').forEach(item=>{
  item.addEventListener('click', ()=>{
    document.querySelectorAll('.rail-item').forEach(i=>i.classList.remove('active'));
    item.classList.add('active');
    document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
    const view = item.dataset.view;
    document.getElementById('view-title').textContent = item.textContent.replace('building','').trim();
    if(view === 'command') document.getElementById('view-command').classList.add('active');
    else if(view === 'feed') document.getElementById('view-feed').classList.add('active');
    else if(view === 'confidence') document.getElementById('view-confidence').classList.add('active');
    else if(view === 'health'){ document.getElementById('view-health').classList.add('active'); loadHealth(); }
    else if(view === 'review'){ document.getElementById('view-review').classList.add('active'); loadReviewQueue(); }
    else if(view === 'audit'){ document.getElementById('view-audit').classList.add('active'); loadAudit(); }
    else document.getElementById('view-building').classList.add('active');
  });
});

function loadHealth(){
  fetch('/api/health').then(r=>r.json()).then(h=>{
    if(h.error) return;
    document.getElementById('health-path').textContent = h.model_path || '—';
    document.getElementById('health-params').textContent = (h.params||0).toLocaleString();
    document.getElementById('health-accuracy').textContent = h.accuracy!=null ? (h.accuracy*100).toFixed(2)+'%' : '—';
    document.getElementById('health-testsize').textContent = (h.total_test||0).toLocaleString();
    const cp = h.checkpoint || {};
    document.getElementById('health-episodes').textContent = cp.episode ?? '—';
    document.getElementById('health-steps').textContent = cp.train_step_count!=null ? cp.train_step_count.toLocaleString() : '—';
  });
}

function loadReviewQueue(){
  const el = document.getElementById('review-list');
  el.innerHTML = '<div style="color:var(--text-muted); font-size:12px;">Loading&hellip;</div>';
  fetch(`/api/review-queue?sid=${socket.id}`).then(r=>r.json()).then(data=>{
    const items = data.items || [];
    if(items.length === 0){
      el.innerHTML = '<div style="color:var(--text-muted); font-size:12px;">No decisions currently pending review.</div>';
      return;
    }
    el.innerHTML = items.map(it => `
      <div class="review-item" id="rq-${it.review_id}">
        <div class="ri-head"><span>${it.label}</span><span>Model default: ${it.default_action}</span></div>
        <div class="ri-meta">Packet #${it.packet_id} &middot; Confidence ${it.confidence}% &middot; ${it.created_at}</div>
        <div class="review-actions">
          <input type="text" id="reason-${it.review_id}" placeholder="Reason (required)&hellip;">
          <button class="btn btn-approve" onclick="resolveReview(${it.review_id}, 'approve', '${it.default_action}')">Approve</button>
          <button class="btn btn-override" onclick="resolveReview(${it.review_id}, 'override', '${it.default_action==='BLOCK'?'ALLOW':'BLOCK'}')">Override</button>
        </div>
      </div>`).join('');
  });
}

function resolveReview(reviewId, action, finalAction){
  const reason = document.getElementById(`reason-${reviewId}`).value.trim();
  if(!reason){ alert('A reason is required before resolving this item.'); return; }
  fetch(`/api/review/${reviewId}`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action, reason, final_action: finalAction})
  }).then(r=>r.json()).then(()=> loadReviewQueue());
}

function loadAudit(){
  const label = document.getElementById('audit-label-filter').value;
  const decision = document.getElementById('audit-decision-filter').value;
  const params = new URLSearchParams();
  if(label) params.set('label', label);
  if(decision) params.set('decision', decision);
  fetch(`/api/audit?${params}`).then(r=>r.json()).then(data=>{
    const rows = data.rows || [];
    document.getElementById('audit-body').innerHTML = rows.map(r => `
      <tr><td>${r.decision_id}</td><td>${r.timestamp}</td><td>${(r.session_id||'').slice(0,8)}</td>
      <td>${r.username||''}</td><td>${r.label}</td><td>${r.predicted_action}</td><td>${r.confidence}%</td></tr>
    `).join('') || '<tr><td colspan="7" style="color:var(--text-muted);">No matching records.</td></tr>';
  });
}
const auditSearchBtn = document.getElementById('btn-audit-search');
if(auditSearchBtn) auditSearchBtn.addEventListener('click', loadAudit);

// ---------- Confidence histogram ----------
const confCanvas = document.getElementById('conf-histogram');
const confCtx = confCanvas.getContext('2d');
const confMargins = [];
function drawConfHistogram(){
  confCanvas.width = confCanvas.offsetWidth*devicePixelRatio; confCanvas.height = confCanvas.offsetHeight*devicePixelRatio;
  const w=confCanvas.width, h=confCanvas.height;
  confCtx.clearRect(0,0,w,h);
  if(confMargins.length<2) return;
  const bins = new Array(10).fill(0);
  confMargins.forEach(m=>{ const b=Math.min(9,Math.floor(m/10)); bins[b]++; });
  const maxBin = Math.max(...bins,1);
  const bw = w/bins.length;
  bins.forEach((count,i)=>{
    const bh = (count/maxBin)*(h*0.85);
    confCtx.fillStyle = '#E8A33D';
    confCtx.fillRect(i*bw+2, h-bh, bw-4, bh);
  });
}
// ---------- Drift timeline ----------
const driftCanvas = document.getElementById('drift-timeline');
const driftCtx = driftCanvas.getContext('2d');
const driftScores = [];
function drawDriftTimeline(){
  driftCanvas.width = driftCanvas.offsetWidth*devicePixelRatio; driftCanvas.height = driftCanvas.offsetHeight*devicePixelRatio;
  const w=driftCanvas.width, h=driftCanvas.height;
  driftCtx.clearRect(0,0,w,h);
  if(driftScores.length<2) return;
  driftCtx.beginPath(); driftCtx.strokeStyle='#7C93B8'; driftCtx.lineWidth=2*devicePixelRatio;
  driftScores.forEach((v,i)=>{
    const x=(i/(driftScores.length-1))*w, y=h*0.92-(v/6)*(h*0.85);
    if(i===0) driftCtx.moveTo(x,y); else driftCtx.lineTo(x,y);
  });
  driftCtx.stroke();
}

function updateShadowAndDrift(d){
  const margin = Math.abs((d.q_block??0)-(d.q_allow??0));
  confMargins.push(margin); if(confMargins.length>100) confMargins.shift();
  drawConfHistogram();
  driftScores.push(d.drift_score||0); if(driftScores.length>150) driftScores.shift();
  drawDriftTimeline();

  const dqnBox = document.getElementById('shadow-dqn'), ruleBox = document.getElementById('shadow-rule');
  dqnBox.querySelector('.sv').textContent = d.last_action;
  ruleBox.querySelector('.sv').textContent = d.rule_decision || '—';
  const agree = d.rule_decision === d.last_action;
  dqnBox.className = 'shadow-box ' + (agree?'agree':'disagree');
  ruleBox.className = 'shadow-box ' + (agree?'agree':'disagree');
  document.getElementById('narrative-box').textContent = d.narrative || '—';

  const dl = document.getElementById('drift-list');
  const top = d.top_drift || [];
  dl.innerHTML = top.length ? top.map(t=>`<div class="drift-row"><span>${t.feature}</span><span class="dz">z=${t.zscore}</span></div>`).join('')
    : '<div style="color:var(--text-muted); font-size:12px;">No significant drift detected.</div>';
}

const pauseBtn = document.getElementById('btn-pause');
if(pauseBtn){
  let paused = false;
  pauseBtn.addEventListener('click', ()=>{
    paused = !paused;
    socket.emit('control', paused ? 'pause' : 'resume');
    pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  });
  document.getElementById('btn-stop').addEventListener('click', ()=> socket.emit('control','stop'));
  document.getElementById('btn-restart').addEventListener('click', ()=>{
    pulseData = []; document.getElementById('ledger-body').innerHTML=''; document.getElementById('feed-body').innerHTML='';
    socket.emit('control','restart');
  });
  document.getElementById('speed').addEventListener('input', (e)=> socket.emit('set_speed', parseInt(e.target.value)));
}

drawPulse();
</script>
</body>
</html>
"""

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        user = USERS.get(username)
        if user and user["password"] == password:
            session["username"] = username
            session["role"] = user["role"]
            db.log_access(username, "login")
            return redirect(url_for("index"))
        return render_template_string(LOGIN_HTML, error="Invalid credentials")
    return render_template_string(LOGIN_HTML, error=None)


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
    sid = session.get("sid")
    with sessions_lock:
        st = sessions.get(sid)
    logs = st['logs'] if st else []
    db.log_access(session.get("username"), "export", sid)
    if not logs:
        return "No logs yet.", 400
    def generate():
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=[
            'packet_id','timestamp','label','is_attack',
            'action','decision','reward','accuracy','precision','recall','f1',
            'confidence','cvss_severity','cvss_score'])
        w.writeheader(); yield out.getvalue(); out.truncate(0); out.seek(0)
        for row in logs:
            w.writerow(row); yield out.getvalue(); out.truncate(0); out.seek(0)
    return Response(generate(), headers={
        'Content-Disposition': 'attachment; filename=adaptiveshield_logs.csv',
        'Content-Type': 'text/csv'})

@app.route("/stats")
@login_required
def stats():
    sid = session.get("sid")
    with sessions_lock:
        st = sessions.get(sid)
    return jsonify(st['stats'] if st else {})


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
        return jsonify({'error': 'engine not ready'}), 503
    feature_names = engine.get('feature_names')
    if feature_names is None:
        return jsonify({'error': 'feature names unavailable -- run python -m src.preprocess once'}), 503

    idx = packet_id - 1
    X_test = engine['X_test']
    if idx < 0 or idx >= len(X_test):
        return jsonify({'error': 'packet_id out of range'}), 404

    model = engine['model']
    x = tf.convert_to_tensor(X_test[idx:idx+1], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        q_vals = model(x, training=False)
        action = int(tf.argmax(q_vals[0]))
        target_q = q_vals[0][action]
    grads = tape.gradient(target_q, x).numpy()[0]
    # gradient x input: a standard attribution refinement over raw gradient
    # alone -- weights each feature's gradient by its own actual value, so
    # a feature that's zero can't appear falsely influential.
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
    traffic. Distance computed in the SAME normalized 0-1 space the model
    itself uses, so it's a fair, consistent measure across features with
    very different natural units (durations vs byte counts vs flag
    counts)."""
    if not engine.get('ready'):
        return jsonify({'error': 'engine not ready'}), 503
    idx = packet_id - 1
    X_test = engine['X_test']; y_test = engine['y_test']; classes = engine['classes']
    if idx < 0 or idx >= len(X_test):
        return jsonify({'error': 'packet_id out of range'}), 404

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
        # Read-only viewers shouldn't be able to write investigation
        # annotations, same restriction as playback controls.
        if session.get('role') not in ('admin', 'analyst'):
            return jsonify({'error': 'insufficient permissions'}), 403
        data = request.json or {}
        text = data.get('note_text', '').strip()
        sid = data.get('sid') or session.get('sid')  # FIX: session.get('sid') set inside a
        # SocketIO handler isn't reliably synced back to the browser's cookie for a plain
        # HTTP fetch() -- frontend now sends socket.id explicitly instead.
        if not text:
            return jsonify({'error': 'empty note'}), 400
        db.add_note(packet_id, sid, session.get('username'), text)
        return jsonify({'ok': True})
    return jsonify({'notes': db.get_notes(packet_id)})


@app.route("/api/health")
@login_required
def api_health():
    if not engine.get('ready'):
        return jsonify({'error': 'engine not ready'}), 503
    return jsonify(engine.get('health', {}))


@app.route("/api/review-queue")
@login_required
def api_review_queue():
    sid = request.args.get('sid') or session.get('sid')
    return jsonify({'items': db.get_pending_reviews(sid)})


@app.route("/api/review/<int:review_id>", methods=["POST"])
@login_required
def api_review_resolve(review_id):
    if session.get('role') not in ('admin', 'analyst'):
        return jsonify({'error': 'insufficient permissions'}), 403
    data = request.json or {}
    action = data.get('action')  # 'approve' or 'override'
    reason = data.get('reason', '').strip()
    final_action = data.get('final_action')
    if action not in ('approve', 'override') or not reason:
        return jsonify({'error': 'action and reason are required'}), 400
    db.resolve_review(review_id, session.get('username'), final_action, reason)
    return jsonify({'ok': True})


@app.route("/api/audit")
@login_required
def api_audit():
    label = request.args.get('label') or None
    decision = request.args.get('decision') or None
    return jsonify({'rows': db.get_audit_log(label, decision)})


def new_session_state():
    return {
        'is_paused': False, 'is_stopped': False,
        'logs': [], 'stats': {},
        'packet_delay': 0.1133,  # matches the default 5x speed-slider value
    }


def init_engine():
    """Loads data + model + runs batched inference ONCE for the whole
    server. Every session's playback reads this same array afterward --
    see run_session below."""
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

    agent = DQNAgent(state_size=X_test.shape[1], action_size=2)
    try:
        agent.load(SAVE_PATH)
    except Exception as e:
        print(f"[AdaptiveShield] ERROR loading model from '{SAVE_PATH}': {e}")
        socketio.emit('load_error', {'message': str(e)})
        return
    agent.epsilon = 0.0

    total_test_size = len(X_test)
    print(f"[AdaptiveShield] Running batched inference on {total_test_size:,} packets...")
    t0 = time.time()
    all_q_vals = agent.model.predict(X_test, batch_size=1024, verbose=0)
    print(f"[AdaptiveShield] Batched inference done in {time.time()-t0:.2f}s")

    # Flow inspector data: recover approximate human-readable (pre-scaling)
    # feature values via the fitted scaler's inverse_transform, rather than
    # showing the model's raw 0-1 normalized inputs. This is what makes the
    # inspector read like real flow statistics (actual packet counts,
    # actual durations) instead of confusing normalized floats. Falls back
    # gracefully if the scaler/feature-order artifacts aren't present yet
    # (e.g. `python -m src.preprocess` hasn't been run to persist them).
    feature_names = None
    X_test_raw = None
    baseline_mean = None
    baseline_std = None
    try:
        scaler = joblib.load("models/cicids2017_scaler.pkl")
        with open("models/cicids2017_feature_order.json") as f:
            feature_names = json.load(f)
        X_test_raw = scaler.inverse_transform(X_test)
        # Baseline distribution for drift detection and the rule-based
        # shadow comparison below: mean/std per feature over the
        # evaluation set, used as "what normal traffic looks like".
        baseline_mean = X_test_raw.mean(axis=0)
        baseline_std = X_test_raw.std(axis=0)
        baseline_std[baseline_std == 0] = 1  # avoid divide-by-zero for constant features
        print(f"[AdaptiveShield] Flow inspector: recovered raw feature values for {len(feature_names)} fields")
    except Exception as e:
        print(f"[AdaptiveShield] Flow inspector unavailable (scaler/feature-order not found): {e}")

    # Lightweight model health check: accuracy of the already-computed
    # batched predictions against ground truth, plus training metadata
    # from the matching checkpoint file if one exists. Cheap (reuses
    # all_q_vals, no new inference) -- shown on the Model Health page.
    predicted = np.argmax(all_q_vals, axis=1)
    true_binary = (y_test != 0).astype(int)
    health_accuracy = float(np.mean(predicted == true_binary))
    checkpoint_info = None
    try:
        with open("models/checkpoint.json") as f:
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
            'model': agent.model,  # kept for on-demand attribution/similarity lookups below
            'health': {
                'model_path': SAVE_PATH,
                'params': int(agent.model.count_params()),
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
    baseline distribution), and the specific features driving that
    unusualness (used for both the drift page and the plain-English
    narrative). Dataset-agnostic -- works from statistics, not
    hardcoded feature names, so it doesn't break across datasets with
    different schemas."""
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
    """Plain-English one-liner built from the same drift analysis, so a
    non-technical viewer gets a readable explanation alongside the raw
    attribution/field data."""
    if decision == 'ALLOW':
        return "Allowed — flow statistics fall within the normal range for benign traffic."
    if top_drift:
        reasons = " and ".join(d['feature'].lower() for d in top_drift[:2])
        return f"Blocked — unusually high {reasons}, consistent with {label}."
    return f"Blocked — classified as {label}."


def run_session(sid, username):
    """Independent playback loop for ONE connected client, over the
    shared read-only engine data. Each session has its own pause/stop
    flags, own logs, and own speed -- multiple people can watch,
    pause, and scrub through the demo independently without
    interfering with each other."""
    st = sessions[sid]
    classes = engine['classes']; y_test = engine['y_test']
    all_q_vals = engine['all_q_vals']; total_test_size = engine['total_test_size']
    feature_names = engine.get('feature_names'); X_test_raw = engine.get('X_test_raw')
    baseline_mean = engine.get('baseline_mean'); baseline_std = engine.get('baseline_std')

    total = blocked = allowed = fp = fn = correct = tp = 0
    accuracy = precision = recall = f1 = 0.0

    for i in range(total_test_size):
        if st['is_stopped']:
            break
        while st['is_paused'] and not st['is_stopped']:
            time.sleep(0.2)

        true_label = int(y_test[i]); is_attack = int(true_label != 0)
        q_vals = all_q_vals[i]
        action = int(np.argmax(q_vals))
        raw_conf = abs(float(q_vals[1]-q_vals[0])) / (abs(float(q_vals[0])) + abs(float(q_vals[1])) + 1e-9)
        confidence = min(99, max(1, round(raw_conf*100, 1)))
        label_name = classes[true_label] if true_label < len(classes) else "Unknown"
        cvss_sev, cvss_score, cvss_clr = get_cvss(label_name)
        mitre_id, mitre_name = get_mitre(label_name)

        if   action==1 and is_attack==1: reward=+1.0; tp+=1
        elif action==0 and is_attack==0: reward=+0.5
        elif action==0 and is_attack==1: reward=-1.0; fn+=1
        else:                            reward=-0.5; fp+=1

        total += 1
        blocked += 1 if action==1 else 0
        allowed += 1 if action==0 else 0
        if reward > 0: correct += 1
        accuracy  = round((correct/total)*100, 2)
        precision = round(tp/(tp+fp)*100, 2) if (tp+fp) > 0 else 0.0
        recall    = round(tp/(tp+fn)*100, 2) if (tp+fn) > 0 else 0.0
        f1        = round(2*precision*recall/(precision+recall), 2) if (precision+recall) > 0 else 0.0
        cpu = round(psutil.cpu_percent(), 1); mem = round(psutil.virtual_memory().percent, 1)

        # Flow inspector payload: real (inverse-scaled) feature values for
        # this specific packet, Wireshark-style field:value pairs. Only
        # built when the scaler/feature-order artifacts were available at
        # engine init -- otherwise this is just an empty dict and the
        # frontend inspector panel shows its "unavailable" state.
        features = {}
        if feature_names is not None and X_test_raw is not None:
            features = {name: round(float(val), 4) for name, val in zip(feature_names, X_test_raw[i])}

        analysis = analyze_flow(features, baseline_mean, baseline_std, feature_names)
        decision_str = 'BLOCK' if action==1 else 'ALLOW'
        narrative = build_narrative(label_name, decision_str, analysis['top_drift'])

        # Tiered confidence routing: below REVIEW_THRESHOLD, the decision
        # is held for human review instead of auto-enforced -- the queued
        # default matches what the model would have done, but a
        # reviewer's approve/override is what actually finalizes it.
        routing_tier = 'auto'
        if confidence < REVIEW_THRESHOLD:
            routing_tier = 'review'
            db.add_review_item(total, sid, label_name, confidence, decision_str)

        row = {'packet_id':total,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),
            'label':label_name,'is_attack':is_attack,'action':action,
            'decision':decision_str,'reward':reward,
            'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,
            'confidence':confidence,'cvss_severity':cvss_sev,'cvss_score':cvss_score}
        st['logs'].append(row)
        db.log_decision(sid, username, row)  # FIX: persisted -- survives restarts
        # NOTE: `features` is intentionally NOT persisted to SQLite here --
        # it would meaningfully bloat the database for what is, for now, a
        # demo-inspection feature. It's only kept in-memory client-side
        # (see JS below), capped the same way the feed table is capped.

        if total % 10 == 0:
            socketio.emit('update', {
                'total':total,'blocked':blocked,'allowed':allowed,'fp':fp,'fn':fn,'tp':tp,
                'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,
                'last_action':decision_str,
                'last_label':label_name,'last_reward':reward,
                'confidence':confidence,'cvss_info':[cvss_sev,cvss_score,cvss_clr],
                'mitre_info':[mitre_id,mitre_name],
                'cpu':cpu,'mem':mem,'packet_id':total,'features':features,
                'q_allow':round(float(q_vals[0]),4),'q_block':round(float(q_vals[1]),4),
                'rule_decision':analysis['rule_decision'],'drift_score':analysis['drift_score'],
                'top_drift':analysis['top_drift'],'narrative':narrative}, room=sid)
        time.sleep(st['packet_delay'])

    if not st['is_stopped']:
        tb = sum(1 for y in y_test if y == 0)
        fpr = round((fp/tb)*100, 2) if tb > 0 else 0.0
        st['stats'] = {'total':total,'blocked':blocked,'allowed':allowed,'tp':tp,'fp':fp,'fn':fn,
            'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'fpr':fpr}
        socketio.emit('done', st['stats'], room=sid)
        print(f"[AdaptiveShield] Session {sid[:8]} done — Acc:{accuracy}% F1:{f1}% FPR:{fpr}% Logs:{len(st['logs']):,}")


@socketio.on('connect')
def on_connect():
    if 'username' not in session:
        return False  # reject unauthenticated socket connections
    sid = request.sid
    username = session.get('username')  # FIX: capture NOW, while still in a
                                         # valid request context -- session
                                         # is not accessible from inside a
                                         # manually-spawned background thread
    session['sid'] = sid  # lets plain HTTP routes (/stats, /download-logs) find this connection's state
    with sessions_lock:
        sessions[sid] = new_session_state()
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
        if sid in sessions:
            sessions[sid]['is_stopped'] = True  # cleanly stop this client's own playback thread
    db.log_access(session.get('username'), 'disconnect', sid)


@socketio.on('control')
def on_control(cmd):
    sid = request.sid
    # FIX (foundation phase): viewers are read-only -- only admin/analyst
    # roles may pause, stop, or restart the simulation.
    if session.get('role') not in ('admin', 'analyst'):
        return
    with sessions_lock:
        st = sessions.get(sid)
    if not st: return
    if   cmd == 'pause':   st['is_paused'] = True;  print(f"[AdaptiveShield] {sid[:8]} paused")
    elif cmd == 'resume':  st['is_paused'] = False; print(f"[AdaptiveShield] {sid[:8]} resumed")
    elif cmd == 'stop':    st['is_stopped'] = True; st['is_paused'] = False; print(f"[AdaptiveShield] {sid[:8]} stopped")
    elif cmd == 'restart':
        with sessions_lock:
            sessions[sid] = new_session_state()
        print(f"[AdaptiveShield] {sid[:8]} restarting...")
        threading.Thread(target=run_session, args=(sid, session.get('username')), daemon=True).start()
    db.log_access(session.get('username'), f'control:{cmd}', sid)


@socketio.on('set_speed')
def on_set_speed(val):
    sid = request.sid
    with sessions_lock:
        st = sessions.get(sid)
    if st:
        st['packet_delay'] = round(0.2 - (val-1)*(0.195)/9, 4)
        print(f"[AdaptiveShield] {sid[:8]} speed {val}× — delay {st['packet_delay']}s")

if __name__ == "__main__":
    db.init_db()
    socketio.run(app, debug=False, port=5000, allow_unsafe_werkzeug=True)