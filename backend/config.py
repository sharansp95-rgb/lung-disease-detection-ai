"""
Central config loader. Reads backend/.env (python-dotenv) once at import
time. Every other module reads secrets/config from here, not from
os.environ directly, so there's one place that knows what's configured
and what's missing.
"""

import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(ENV_PATH)

logger = logging.getLogger("ct_xray_api.config")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

JWT_SECRET = os.environ.get("JWT_SECRET", "")
if not JWT_SECRET:
    # generate + persist so sessions survive restarts even if .env was
    # missing this key entirely
    JWT_SECRET = secrets.token_hex(32)
    with open(ENV_PATH, "a") as f:
        f.write(f"\nJWT_SECRET={JWT_SECRET}\n")
    logger.warning("JWT_SECRET was not set -- generated a new one and appended it to .env")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_OAUTH_CONFIGURED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
EMAIL_CONFIGURED = bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD)

# Public-facing base URLs. Every email link, post-auth redirect, and the
# Google OAuth redirect_uri is built from these two -- for the ngrok setup
# (docs/DEPLOYMENT_PUBLIC.md) they become https://<domain> and https://<domain>/api.
# Trailing slashes stripped so f"{BACKEND_URL}/auth/..." never doubles up.
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173").rstrip("/")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
GOOGLE_REDIRECT_URI = f"{BACKEND_URL}/auth/google/callback"

# Frontend origins the app may be used from at the same time (e.g. local
# dev/preview and the ngrok domain). Email links and Google sign-in follow
# the origin the user is actually on, but only if it's listed here -- any
# other origin falls back to FRONTEND_URL/BACKEND_URL (see public_origin.py).
_DEFAULT_PUBLIC_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
]
ALLOWED_PUBLIC_ORIGINS = list(dict.fromkeys(
    o.strip().rstrip("/")
    for o in (os.environ.get("ALLOWED_PUBLIC_ORIGINS") or ",".join(_DEFAULT_PUBLIC_ORIGINS)).split(",")
    if o.strip()
))

# Path the frontend's Vite server proxies to this backend (vite.config.js).
# Behind the proxy, the backend's public base for origin O is O + this.
PROXY_API_PREFIX = "/api"

# Comma-separated browser origins allowed by CORS. Only matters for plain
# local dev (Vite on :5173 calling :8000 directly); behind the /api proxy
# the browser sees one origin and CORS never comes into play.
CORS_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]

DB_PATH = Path(os.environ.get("DB_PATH", BACKEND_DIR / "app.db"))

# Minimum top-class softmax probability required by the modality gate.
# Calls below this threshold return HTTP 422 "Modality uncertain".
MODALITY_GATE_CONFIDENCE_THRESHOLD = float(os.environ.get("MODALITY_GATE_CONFIDENCE_THRESHOLD", "0.90"))
