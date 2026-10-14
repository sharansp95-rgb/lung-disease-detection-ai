"""
Password hashing (bcrypt) and JWT session tokens (PyJWT, HS256).
"""

import datetime
import secrets

import bcrypt
import jwt

from config import JWT_SECRET

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24 * 7  # 7 days

VERIFY_TOKEN_EXPIRY_HOURS = 24
RESET_TOKEN_EXPIRY_HOURS = 1


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_session_token(user_id: int, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.datetime.now(datetime.timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_session_token(token: str) -> dict:
    """Raises jwt.PyJWTError subclasses on invalid/expired token -- caller handles it."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def generate_random_token() -> str:
    return secrets.token_urlsafe(32)
