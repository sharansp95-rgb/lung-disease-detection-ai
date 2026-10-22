"""
Auth endpoints: signup, login, email verification, password reset, Google
OAuth, and the get_current_user dependency other routers use to protect
themselves.
"""

import datetime
import logging
import re
import secrets
from urllib.parse import quote

import jwt
import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, field_validator

import config
import public_origin
from auth_utils import (
    create_session_token,
    decode_session_token,
    generate_random_token,
    hash_password,
    verify_password,
    RESET_TOKEN_EXPIRY_HOURS,
    VERIFY_TOKEN_EXPIRY_HOURS,
)
from db import get_db
from email_utils import send_password_reset_email, send_verification_email
from rate_limit import limiter

logger = logging.getLogger("ct_xray_api.auth")
router = APIRouter(prefix="/auth", tags=["auth"])

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------- schemas --

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None

    @field_validator("email")
    @classmethod
    def valid_email(cls, v):
        if not EMAIL_RE.match(v):
            raise ValueError("Invalid email address")
        return v.lower()

    @field_validator("password")
    @classmethod
    def valid_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    new_password: str

    @field_validator("new_password")
    @classmethod
    def valid_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def valid_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserOut(BaseModel):
    id: int
    email: str
    email_verified: bool
    display_name: str | None
    created_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ------------------------------------------------------------- dependency --

def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_session_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired, please log in again.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid session token.")

    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (int(payload["sub"]),)).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return dict(row)


def get_current_user_optional(authorization: str = Header(None)) -> dict | None:
    """For routes usable by both logged-in and anonymous users."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return get_current_user(authorization)
    except HTTPException:
        return None


def require_verified_if_logged_in(authorization: str = Header(None)) -> dict | None:
    """Like get_current_user_optional (anonymous callers pass through
    unchanged), but a logged-in user whose email isn't verified yet is
    blocked from scan predictions and chat until they click the link."""
    user = get_current_user_optional(authorization)
    if user is not None and not user["email_verified"]:
        raise HTTPException(status_code=403, detail="Please verify your email to use this feature.")
    return user


def _user_out(row: dict) -> UserOut:
    return UserOut(
        id=row["id"], email=row["email"], email_verified=bool(row["email_verified"]),
        display_name=row["display_name"], created_at=row["created_at"],
    )


# ------------------------------------------------------------------ signup --

@router.post("/signup", response_model=AuthResponse)
@limiter.limit("30/hour")
def signup(request: Request, req: SignupRequest):
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (req.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")

        user_id = conn.execute(
            "INSERT INTO users (email, password_hash, email_verified, display_name) VALUES (?, ?, 0, ?) RETURNING id",
            (req.email, hash_password(req.password), req.name.strip() if req.name and req.name.strip() else None),
        ).fetchone()["id"]

        verify_token = generate_random_token()
        expires = (datetime.datetime.now(datetime.timezone.utc)
                   + datetime.timedelta(hours=VERIFY_TOKEN_EXPIRY_HOURS)).isoformat()
        conn.execute(
            "INSERT INTO tokens (user_id, token, type, expires_at) VALUES (?, ?, 'verify', ?)",
            (user_id, verify_token, expires),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    origin = public_origin.resolve(request)
    link = f"{origin.backend}/auth/verify/{verify_token}"
    if origin.frontend != config.FRONTEND_URL:
        # The click comes from a mail client with no Origin/Referer, so say
        # where to land afterwards. Re-checked against the allowlist there.
        link += f"?origin={quote(origin.frontend, safe='')}"
    send_verification_email(req.email, link)

    access_token = create_session_token(user_id, req.email)
    return AuthResponse(access_token=access_token, user=_user_out(dict(row)))


# ------------------------------------------------------------------- login --

@router.post("/login", response_model=AuthResponse)
@limiter.limit("20/minute")
def login(request: Request, req: LoginRequest):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (req.email.lower(),)).fetchone()

    if row is None or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    access_token = create_session_token(row["id"], row["email"])
    return AuthResponse(access_token=access_token, user=_user_out(dict(row)))


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    return _user_out(user)


@router.post("/change-password")
def change_password(req: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    if not user["password_hash"]:
        raise HTTPException(
            status_code=400,
            detail="This account signed up via Google and has no password set, so there's nothing to change here.",
        )
    if not verify_password(req.current_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    with get_db() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(req.new_password), user["id"]),
        )
    return {"message": "Password updated."}


# ------------------------------------------------------ email verification --

@router.get("/verify/{token}")
def verify_email(token: str, origin: str | None = None):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM tokens WHERE token = ? AND type = 'verify'", (token,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=400, detail="Invalid or unknown verification link.")
        if row["used"]:
            raise HTTPException(status_code=400, detail="This verification link has already been used.")
        if datetime.datetime.fromisoformat(row["expires_at"]) < datetime.datetime.now(datetime.timezone.utc):
            raise HTTPException(status_code=400, detail="This verification link has expired.")

        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (row["user_id"],))
        conn.execute("UPDATE tokens SET used = 1 WHERE id = ?", (row["id"],))

    frontend = public_origin.normalize_origin(origin)
    if not public_origin.is_allowed(frontend):
        frontend = config.FRONTEND_URL
    return RedirectResponse(url=f"{frontend}/email-verified")


# ----------------------------------------------------------- password reset --

@router.post("/forgot-password")
@limiter.limit("5/hour")
def forgot_password(request: Request, req: ForgotPasswordRequest):
    # Always return the same generic response whether or not the email
    # exists -- standard practice, don't leak account existence.
    generic_response = {"message": "If an account exists for that email, a reset link has been sent."}

    with get_db() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ?", (req.email.lower(),)).fetchone()
        if row is None:
            return generic_response

        reset_token = generate_random_token()
        expires = (datetime.datetime.now(datetime.timezone.utc)
                   + datetime.timedelta(hours=RESET_TOKEN_EXPIRY_HOURS)).isoformat()
        conn.execute(
            "INSERT INTO tokens (user_id, token, type, expires_at) VALUES (?, ?, 'reset', ?)",
            (row["id"], reset_token, expires),
        )

    link = f"{public_origin.resolve(request).frontend}/reset-password/{reset_token}"
    send_password_reset_email(req.email.lower(), link)
    return generic_response


@router.post("/reset-password/{token}")
def reset_password(token: str, req: ResetPasswordRequest):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM tokens WHERE token = ? AND type = 'reset'", (token,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=400, detail="Invalid or unknown reset link.")
        if row["used"]:
            raise HTTPException(status_code=400, detail="This reset link has already been used.")
        if datetime.datetime.fromisoformat(row["expires_at"]) < datetime.datetime.now(datetime.timezone.utc):
            raise HTTPException(status_code=400, detail="This reset link has expired.")

        user = conn.execute("SELECT password_hash FROM users WHERE id = ?", (row["user_id"],)).fetchone()
        if user["password_hash"] and verify_password(req.new_password, user["password_hash"]):
            raise HTTPException(status_code=400, detail="New password must be different from your current password.")

        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(req.new_password), row["user_id"]),
        )
        conn.execute("UPDATE tokens SET used = 1 WHERE id = ?", (row["id"],))

    return {"message": "Password has been reset. You can now log in with your new password."}


# -------------------------------------------------------------- google oauth --

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@router.get("/google/login")
def google_login(request: Request):
    if not config.GOOGLE_OAUTH_CONFIGURED:
        raise HTTPException(
            status_code=501,
            detail=(
                "Google sign-in is not fully configured yet (missing GOOGLE_CLIENT_SECRET). "
                "See backend/README_AUTH.md for setup steps."
            ),
        )
    origin = public_origin.resolve(request)
    state, nonce = public_origin.create_oauth_state(origin)
    params = {
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": origin.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account",
        "state": state,
    }
    query = "&".join(f"{k}={requests.utils.quote(v)}" for k, v in params.items())
    resp = RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{query}")
    # Lax is sent on Google's top-level redirect back to the callback.
    resp.set_cookie(
        public_origin.OAUTH_NONCE_COOKIE, nonce,
        max_age=public_origin.OAUTH_STATE_EXPIRY_MINUTES * 60, path="/",
        httponly=True, samesite="lax", secure=origin.backend.startswith("https://"),
    )
    return resp


@router.get("/google/callback")
def google_callback(request: Request, code: str = None, error: str = None, state: str = None):
    decoded = public_origin.decode_oauth_state(state)
    cookie_nonce = request.cookies.get(public_origin.OAUTH_NONCE_COOKIE, "")
    state_valid = decoded is not None and secrets.compare_digest(cookie_nonce, decoded[1])
    origin = decoded[0] if state_valid else public_origin.default_origin()

    def to_frontend(path: str) -> RedirectResponse:
        resp = RedirectResponse(url=f"{origin.frontend}{path}")
        resp.delete_cookie(public_origin.OAUTH_NONCE_COOKIE, path="/")
        return resp

    if error:
        return to_frontend("/login?error=google_denied")
    if not config.GOOGLE_OAUTH_CONFIGURED:
        raise HTTPException(status_code=501, detail="Google sign-in is not configured.")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")
    if not state_valid:
        logger.warning("Google callback rejected: missing, expired or mismatched state.")
        return to_frontend("/login?error=google_state_invalid")

    token_resp = requests.post(GOOGLE_TOKEN_URL, data={
        "client_id": config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": origin.google_redirect_uri,
    }, timeout=15)
    if not token_resp.ok:
        logger.warning(f"Google token exchange failed: {token_resp.status_code} {token_resp.text}")
        return to_frontend("/login?error=google_token_exchange_failed")

    access_token = token_resp.json().get("access_token")
    userinfo_resp = requests.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
    if not userinfo_resp.ok:
        return to_frontend("/login?error=google_userinfo_failed")

    info = userinfo_resp.json()
    google_id = info["id"]
    email = info["email"].lower()
    name = info.get("name", email)

    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE google_id = ? OR email = ?", (google_id, email)).fetchone()
        if row is None:
            user_id = conn.execute(
                "INSERT INTO users (email, google_id, display_name, email_verified) VALUES (?, ?, ?, 1) RETURNING id",
                (email, google_id, name),
            ).fetchone()["id"]
        else:
            user_id = row["id"]
            if row["google_id"] is None:
                conn.execute("UPDATE users SET google_id = ?, email_verified = 1 WHERE id = ?", (google_id, user_id))

    session_token = create_session_token(user_id, email)
    return to_frontend(f"/oauth-callback?token={session_token}")
