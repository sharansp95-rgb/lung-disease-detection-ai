"""
Test setup. Points the app at a throwaway SQLite file (never the real
app.db) *before* anything imports config.py, since DB_PATH is read once at
import time. Model checkpoints and data/ are the real ones -- there's no
mock for those, so the full startup reproduction check (see
backend/models.py) genuinely runs before the test session's first request.
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

_TEST_DB_PATH = BACKEND_DIR / "tests" / "_test.db"
if _TEST_DB_PATH.exists():
    _TEST_DB_PATH.unlink()
os.environ["DB_PATH"] = str(_TEST_DB_PATH)
# Never the DATABASE_URL from backend/.env (that may be the hosted DB):
# Postgres runs only when TEST_DATABASE_URL names a throwaway database.
# Set even when empty so load_dotenv() can't fill it in.
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", "")

import pytest
from fastapi.testclient import TestClient

from main import app

XRAY_SAMPLE = BACKEND_DIR.parent / "data" / "xray" / "test" / "Normal"
CT_SAMPLE = BACKEND_DIR.parent / "data" / "ct_scan" / "test" / "Normal"


@pytest.fixture(scope="session")
def client():
    # `with` triggers the real startup event (DB init only). Models lazy-load
    # with their reproduction checks on the first prediction request.
    # Session-scoped so that (slow) load happens once for the whole test run.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def xray_sample_path():
    return next(XRAY_SAMPLE.glob("*.png"))


@pytest.fixture(scope="session")
def ct_sample_path():
    return next(CT_SAMPLE.glob("*.jpg"))


@pytest.fixture(autouse=True)
def _no_real_email(monkeypatch):
    """Force the console-fallback path in email_utils regardless of what's
    configured in backend/.env -- tests must never make real SMTP calls."""
    import email_utils

    monkeypatch.setattr(email_utils, "EMAIL_CONFIGURED", False)


@pytest.fixture(autouse=True)
def _no_real_groq(monkeypatch):
    """Tests must never call the real Groq API, even if GROQ_API_KEY is set
    in backend/.env. report.py and chat.py import call_groq by name, so the
    mock is applied in both modules. Default behaviour raises, which drives
    the template-fallback path; a test can re-patch to simulate success."""
    import chat
    import report

    def _fail(*_args, **_kwargs):
        raise RuntimeError("Groq is mocked out in tests")

    monkeypatch.setattr(report, "call_groq", _fail)
    monkeypatch.setattr(chat, "call_groq", _fail)


@pytest.fixture
def signed_up_user(client):
    """Creates a fresh account per test (unique email) and returns
    (access_token, email)."""
    import uuid

    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    password = "correct-horse-battery-staple"
    resp = client.post("/auth/signup", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return token, email, password


@pytest.fixture
def verified_user(signed_up_user):
    """Same as signed_up_user, but with the email marked verified in the
    test DB -- the state a user reaches by clicking the verification link.
    Unverified users are blocked from /predict and /chat (403), so any test
    that needs a signed-in user to get a prediction must use this fixture."""
    from db import get_db

    token, email, password = signed_up_user
    with get_db() as conn:
        conn.execute("UPDATE users SET email_verified = 1 WHERE email = ?", (email,))
    return token, email, password
