"""Central configuration. Everything is overridable with environment variables,
so the same code runs locally and in Docker without edits."""
import os


def _bool(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --- Core -------------------------------------------------------------------
SECRET_KEY = os.environ.get("ADAPTIVESHIELD_SECRET", "dev-secret-change-before-real-deployment")
HOST = os.environ.get("ADAPTIVESHIELD_HOST", "127.0.0.1")
PORT = _int("ADAPTIVESHIELD_PORT", 5000)
DB_PATH = os.environ.get("ADAPTIVESHIELD_DB", "data/adaptiveshield_app.db")
MODELS_DIR = os.environ.get("ADAPTIVESHIELD_MODELS_DIR", "models")
CACHE_DIR = os.environ.get("ADAPTIVESHIELD_CACHE_DIR", "data/cache")
DEFAULT_MODEL_FILE = os.environ.get("ADAPTIVESHIELD_DEFAULT_MODEL", "dqn_firewall")
SCALER_PATH = os.environ.get("ADAPTIVESHIELD_SCALER", "models/cicids2017_scaler.pkl")
FEATURE_ORDER_PATH = os.environ.get("ADAPTIVESHIELD_FEATURES", "models/cicids2017_feature_order.json")

# --- Sessions and login hardening ---------------------------------------------
COOKIE_SECURE = _bool("ADAPTIVESHIELD_COOKIE_SECURE", False)   # set true behind HTTPS
SESSION_LIFETIME_HOURS = _int("ADAPTIVESHIELD_SESSION_HOURS", 12)
IDLE_TIMEOUT_MIN = _int("ADAPTIVESHIELD_IDLE_MIN", 60)          # admin can change live
LOCKOUT_MAX_FAILS = _int("ADAPTIVESHIELD_LOCKOUT_FAILS", 5)
LOCKOUT_WINDOW_MIN = _int("ADAPTIVESHIELD_LOCKOUT_MIN", 15)

# Browsers allowed to open the live Socket.IO channel. Empty = same origin only.
SOCKET_ORIGINS = [o.strip() for o in os.environ.get("ADAPTIVESHIELD_ORIGINS", "").split(",") if o.strip()] or None

# --- First-run accounts ------------------------------------------------------
# Created only when the user table is empty. Every seeded account must change
# its password at first sign-in.
ADMIN_PASSWORD = os.environ.get("ADAPTIVESHIELD_ADMIN_PASSWORD", "adaptiveshield")
SEED_DEMO_USERS = _bool("ADAPTIVESHIELD_SEED_DEMO_USERS", True)

# --- Email (scheduled reports and alert emails) ----------------------------------
SMTP_HOST = os.environ.get("ADAPTIVESHIELD_SMTP_HOST", "")
SMTP_PORT = _int("ADAPTIVESHIELD_SMTP_PORT", 587)
SMTP_USER = os.environ.get("ADAPTIVESHIELD_SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("ADAPTIVESHIELD_SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("ADAPTIVESHIELD_SMTP_FROM", SMTP_USER or "adaptiveshield@localhost")
SMTP_TLS = _bool("ADAPTIVESHIELD_SMTP_TLS", True)

# --- Tunables with live overrides in the Admin page ----------------------------------
DEFAULT_SETTINGS = {
    "review_threshold": 40,      # confidence % below which a decision is held for review
    "drift_alarm_mean": 3.0,     # average drifted-feature count that raises a drift alarm
    "idle_timeout_min": IDLE_TIMEOUT_MIN,
    "default_model": DEFAULT_MODEL_FILE,
}
