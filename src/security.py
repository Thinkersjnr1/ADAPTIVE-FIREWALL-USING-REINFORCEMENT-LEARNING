"""Login hardening: lockout, password policy and TOTP two-factor (RFC 6238).
TOTP is implemented with the standard library so there is no extra dependency."""
import base64
import hashlib
import hmac
import os
import struct
import time

from src import store

COMMON = {"password", "password1", "password123", "123456789", "1234567890", "qwerty123",
          "adaptiveshield", "adaptiveshield1", "analyst123", "viewer123", "letmein123", "admin12345"}


def check_password(pw, username=""):
    """Return a plain-language problem with the password, or None if it is acceptable."""
    if len(pw or "") < 10:
        return "Use at least 10 characters."
    if not any(c.isalpha() for c in pw) or not any(c.isdigit() for c in pw):
        return "Include at least one letter and one number."
    if username and username.lower() in pw.lower():
        return "The password can't contain the username."
    if pw.lower() in COMMON:
        return "That password is too common."
    return None


# ---- lockout ---------------------------------------------------------------------------------
def record_attempt(username, ip, ok):
    store.execute("INSERT INTO login_attempts(username, ip, ts, ok) VALUES(?,?,?,?)",
                  ((username or "")[:64], (ip or "")[:64], time.time(), 1 if ok else 0))
    if ok:  # tidy up so old failures never count against a good login
        store.change("DELETE FROM login_attempts WHERE username=? AND ok=0", (username,))
    store.change("DELETE FROM login_attempts WHERE ts < ?", (time.time() - 86400,))


def lockout_seconds(username, ip, max_fails, window_min):
    """Seconds left on a lockout for this account or address, or 0 if allowed to try."""
    window = window_min * 60
    cutoff = time.time() - window
    worst = 0
    for col, val, limit in (("username", username or "", max_fails), ("ip", ip or "", max_fails * 4)):
        if not val:
            continue
        rows = store.query(f"SELECT ts FROM login_attempts WHERE {col}=? AND ok=0 AND ts>? ORDER BY ts DESC",
                           (val, cutoff))
        if len(rows) >= limit:
            worst = max(worst, int(rows[0]["ts"] + window - time.time()) + 1)
    return max(0, worst)


def clear_lockout(username):
    store.change("DELETE FROM login_attempts WHERE username=?", (username,))


# ---- TOTP ------------------------------------------------------------------------------------
def new_secret():
    return base64.b32encode(os.urandom(20)).decode().rstrip("=")


def _code(secret, counter, digits=6):
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = digest[-1] & 0x0F
    num = (struct.unpack(">I", digest[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(num).zfill(digits)


def totp_now(secret, at=None, step=30):
    return _code(secret, int((at or time.time()) // step))


def verify_totp(secret, code, last_counter=0, at=None, step=30, window=1):
    """Return the matched time counter, or None. A counter that was already used is
    rejected so one code can't be replayed."""
    code = (code or "").strip().replace(" ", "")
    if len(code) != 6 or not code.isdigit():
        return None
    base = int((at or time.time()) // step)
    for c in range(base - window, base + window + 1):
        if c > last_counter and hmac.compare_digest(_code(secret, c), code):
            return c
    return None


def provisioning_uri(secret, username, issuer="AdaptiveShield"):
    from urllib.parse import quote
    return f"otpauth://totp/{quote(issuer)}:{quote(username)}?secret={secret}&issuer={quote(issuer)}&digits=6&period=30"


def qr_svg(uri):
    """Inline SVG of the QR code if the optional `qrcode` package is installed."""
    try:
        import io
        import qrcode
        import qrcode.image.svg
        img = qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage, box_size=8)
        buf = io.BytesIO()
        img.save(buf)
        return buf.getvalue().decode()
    except Exception:
        return None
