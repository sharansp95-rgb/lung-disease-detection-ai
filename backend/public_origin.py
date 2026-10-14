"""
Works out which public origin a request belongs to, so email links and
Google sign-in send the user back to the site they're actually on -- e.g.
http://localhost:4173 and the ngrok domain can be used side by side
without editing backend/.env.

The frontend origin is taken from, in order: the Origin header, the
Referer header, then X-Forwarded-Proto + X-Forwarded-Host (only when the
direct peer is the local Vite proxy, see rate_limit.py). The first
candidate that is in config.ALLOWED_PUBLIC_ORIGINS wins. The Host header is
never used, and nothing outside the allowlist is ever returned, so a
crafted header can at most pick another allowlisted origin -- no open
redirect. With no allowlisted candidate, FRONTEND_URL/BACKEND_URL are used
exactly as before.

Google OAuth carries the chosen origins through the flow in a signed,
short-lived `state` JWT, bound to the browser by a nonce cookie (CSRF).
"""

import datetime
import secrets
from dataclasses import dataclass
from urllib.parse import urlsplit

import jwt
from starlette.requests import Request

import config
from auth_utils import JWT_ALGORITHM
from rate_limit import TRUSTED_PROXY_PEERS

OAUTH_STATE_TYPE = "google_oauth_state"
OAUTH_STATE_EXPIRY_MINUTES = 10
OAUTH_NONCE_COOKIE = "google_oauth_nonce"


@dataclass(frozen=True)
class PublicOrigin:
    frontend: str  # e.g. https://<domain>
    backend: str   # e.g. https://<domain>/api

    @property
    def google_redirect_uri(self) -> str:
        return f"{self.backend}/auth/google/callback"


def default_origin() -> PublicOrigin:
    return PublicOrigin(config.FRONTEND_URL, config.BACKEND_URL)


def normalize_origin(value: str | None) -> str | None:
    """scheme://host[:port] from an origin or full URL, or None if unusable."""
    if not value or value == "null":
        return None
    try:
        parts = urlsplit(value.strip())
        _ = parts.port  # raises on a malformed port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname or "@" in parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc.lower()}"


def is_allowed(origin: str | None) -> bool:
    return origin is not None and (origin == config.FRONTEND_URL or origin in config.ALLOWED_PUBLIC_ORIGINS)


def _last_header_value(request: Request, name: str) -> str:
    # Proxies append; the right-most value is the one closest to us.
    return request.headers.get(name, "").split(",")[-1].strip()


def _forwarded_origin(request: Request) -> str | None:
    peer = request.client.host if request.client else None
    if peer not in TRUSTED_PROXY_PEERS:
        return None
    host = _last_header_value(request, "x-forwarded-host")
    proto = _last_header_value(request, "x-forwarded-proto") or "http"
    if not host:
        return None
    return normalize_origin(f"{proto}://{host}")


def _direct_backend_base(request: Request) -> str:
    # The socket this request arrived on (uvicorn's scope["server"]), not
    # the client-controlled Host header.
    server = request.scope.get("server")
    if not server:
        return config.BACKEND_URL
    host, port = server
    scheme = request.url.scheme
    default_port = 443 if scheme == "https" else 80
    return f"{scheme}://{host}" if port in (None, default_port) else f"{scheme}://{host}:{port}"


def resolve(request: Request) -> PublicOrigin:
    forwarded = _forwarded_origin(request)
    candidates = (
        normalize_origin(request.headers.get("origin")),
        normalize_origin(request.headers.get("referer")),
        forwarded,
    )
    frontend = next((c for c in candidates if is_allowed(c)), None)
    if frontend is None or frontend == config.FRONTEND_URL:
        return default_origin()

    if forwarded is not None:
        # Came through the Vite /api proxy, so that's the backend's public base.
        if not is_allowed(forwarded):
            return default_origin()
        return PublicOrigin(frontend, f"{forwarded}{config.PROXY_API_PREFIX}")
    # Frontend calls the backend directly (plain `npm run dev`).
    return PublicOrigin(frontend, _direct_backend_base(request))


# ------------------------------------------------------------ oauth state --

def create_oauth_state(origin: PublicOrigin) -> tuple[str, str]:
    """Returns (state, nonce). The nonce goes in a cookie and must come back."""
    nonce = secrets.token_urlsafe(24)
    payload = {
        "typ": OAUTH_STATE_TYPE,
        "fe": origin.frontend,
        "be": origin.backend,
        "nonce": nonce,
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(minutes=OAUTH_STATE_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=JWT_ALGORITHM), nonce


def decode_oauth_state(state: str | None) -> tuple[PublicOrigin, str] | None:
    """(origin, nonce) for a valid state we issued, else None."""
    if not state:
        return None
    try:
        payload = jwt.decode(state, config.JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != OAUTH_STATE_TYPE or not payload.get("nonce"):
        return None
    frontend, backend = payload.get("fe"), payload.get("be")
    if not isinstance(backend, str) or not is_allowed(frontend):
        return None
    return PublicOrigin(frontend, backend), payload["nonce"]
