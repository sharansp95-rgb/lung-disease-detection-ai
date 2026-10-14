"""
Multi-origin support (public_origin.py): the app used from localhost and
the ngrok domain at the same time, with email links and Google sign-in
following the origin the user is on -- and never an origin outside
ALLOWED_PUBLIC_ORIGINS.
"""

from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.requests import Request

import auth_routes
import config
import public_origin
from rate_limit import limiter

NGROK = "https://example-app.ngrok-free.app"
LOCAL_PREVIEW = "http://localhost:4173"
LOCAL_DEV = "http://localhost:5173"
EVIL = "https://evil.example.com"


@pytest.fixture(autouse=True)
def ngrok_config(monkeypatch):
    """Mirrors the real backend/.env: defaults point at ngrok, localhost allowlisted."""
    monkeypatch.setattr(config, "FRONTEND_URL", NGROK)
    monkeypatch.setattr(config, "BACKEND_URL", f"{NGROK}/api")
    monkeypatch.setattr(config, "ALLOWED_PUBLIC_ORIGINS", [NGROK, LOCAL_DEV, LOCAL_PREVIEW])
    limiter.reset()
    yield
    limiter.reset()


def _backend_path(public_path: str) -> str:
    """What the Vite proxy would forward: /api/auth/... -> /auth/..."""
    return public_path.removeprefix(config.PROXY_API_PREFIX)


def _request(peer="127.0.0.1", server=("127.0.0.1", 8000), **headers) -> Request:
    raw = [(k.replace("_", "-").lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({
        "type": "http", "method": "GET", "path": "/", "scheme": "http", "query_string": b"",
        "headers": raw, "client": (peer, 12345), "server": server,
    })


# ---------------------------------------------------------------- resolve --

def test_resolve_localhost_preview_via_proxy():
    req = _request(origin=LOCAL_PREVIEW, x_forwarded_host="localhost:4173", x_forwarded_proto="http")
    assert public_origin.resolve(req) == public_origin.PublicOrigin(LOCAL_PREVIEW, f"{LOCAL_PREVIEW}/api")


def test_resolve_localhost_from_forwarded_headers_alone():
    req = _request(x_forwarded_host="localhost:4173", x_forwarded_proto="http")
    assert public_origin.resolve(req).backend == f"{LOCAL_PREVIEW}/api"


def test_resolve_localhost_dev_calling_backend_directly():
    req = _request(referer=f"{LOCAL_DEV}/login")
    assert public_origin.resolve(req) == public_origin.PublicOrigin(LOCAL_DEV, "http://127.0.0.1:8000")


def test_resolve_ngrok_origin():
    req = _request(referer=f"{NGROK}/login", x_forwarded_host="example-app.ngrok-free.app",
                   x_forwarded_proto="https")
    assert public_origin.resolve(req) == public_origin.PublicOrigin(NGROK, f"{NGROK}/api")


def test_resolve_non_allowlisted_origin_falls_back_to_default():
    req = _request(origin=EVIL, referer=f"{EVIL}/x", x_forwarded_host="evil.example.com",
                   x_forwarded_proto="https", host="evil.example.com")
    assert public_origin.resolve(req) == public_origin.default_origin()


def test_resolve_ignores_forwarded_headers_from_non_local_peer():
    req = _request(peer="203.0.113.7", x_forwarded_host="localhost:4173", x_forwarded_proto="http")
    assert public_origin.resolve(req) == public_origin.default_origin()


def test_single_origin_config_keeps_old_behaviour(monkeypatch):
    monkeypatch.setattr(config, "FRONTEND_URL", LOCAL_DEV)
    monkeypatch.setattr(config, "BACKEND_URL", "http://127.0.0.1:8000")
    monkeypatch.setattr(config, "ALLOWED_PUBLIC_ORIGINS", [LOCAL_DEV])
    for req in (_request(), _request(origin=LOCAL_DEV), _request(origin=EVIL)):
        assert public_origin.resolve(req) == public_origin.PublicOrigin(LOCAL_DEV, "http://127.0.0.1:8000")


# ------------------------------------------------------ forgot password --

@pytest.fixture
def sent_links(monkeypatch):
    links = {}
    monkeypatch.setattr(auth_routes, "send_password_reset_email", lambda to, link: links.__setitem__("reset", link))
    monkeypatch.setattr(auth_routes, "send_verification_email", lambda to, link: links.__setitem__("verify", link))
    return links


@pytest.mark.parametrize("sent_origin, expected", [
    (LOCAL_PREVIEW, LOCAL_PREVIEW),
    (NGROK, NGROK),
    (EVIL, NGROK),
])
def test_forgot_password_link_follows_allowlisted_origin(client, signed_up_user, sent_links, sent_origin, expected):
    _, email, _ = signed_up_user
    resp = client.post("/auth/forgot-password", json={"email": email}, headers={"Origin": sent_origin})
    assert resp.status_code == 200
    assert sent_links["reset"].startswith(f"{expected}/reset-password/")


# ------------------------------------------------------ email verification --

@pytest.mark.parametrize("sent_origin, expected_link_prefix, expected_landing", [
    (LOCAL_DEV, "http://testserver/auth/verify/", LOCAL_DEV),  # direct call, TestClient's socket
    (NGROK, f"{NGROK}/api/auth/verify/", NGROK),
    (EVIL, f"{NGROK}/api/auth/verify/", NGROK),
])
def test_verification_link_and_landing_follow_origin(
    client, sent_links, sent_origin, expected_link_prefix, expected_landing
):
    import uuid

    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    resp = client.post("/auth/signup", json={"email": email, "password": "long-enough-pw"},
                       headers={"Origin": sent_origin})
    assert resp.status_code == 200, resp.text
    link = sent_links["verify"]
    assert link.startswith(expected_link_prefix)
    assert ("?origin=" in link) == (expected_landing != NGROK)

    parts = urlsplit(link)
    resp = client.get(f"{_backend_path(parts.path)}?{parts.query}", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == f"{expected_landing}/email-verified"


def test_verify_rejects_non_allowlisted_origin_param(client, sent_links):
    import uuid

    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    client.post("/auth/signup", json={"email": email, "password": "long-enough-pw"})
    path = _backend_path(urlsplit(sent_links["verify"]).path)
    resp = client.get(path, params={"origin": EVIL}, follow_redirects=False)
    assert resp.headers["location"] == f"{NGROK}/email-verified"


# ------------------------------------------------------------ google oauth --

@pytest.fixture
def google_configured(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_OAUTH_CONFIGURED", True)
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "test-client-secret")


@pytest.fixture
def fake_google(monkeypatch):
    calls = {}

    class _Resp:
        def __init__(self, data):
            self.ok, self.status_code, self.text, self._data = True, 200, "", data

        def json(self):
            return self._data

    def fake_post(url, data, timeout):
        calls["token_redirect_uri"] = data["redirect_uri"]
        return _Resp({"access_token": "fake-access"})

    def fake_get(url, headers, timeout):
        import uuid
        return _Resp({"id": uuid.uuid4().hex, "email": f"g-{uuid.uuid4().hex[:10]}@example.com", "name": "G"})

    monkeypatch.setattr(auth_routes.requests, "post", fake_post)
    monkeypatch.setattr(auth_routes.requests, "get", fake_get)
    return calls


def _google_login(client, referer):
    resp = client.get("/auth/google/login", headers={"Referer": referer}, follow_redirects=False)
    assert resp.status_code in (302, 307), resp.text
    params = parse_qs(urlsplit(resp.headers["location"]).query)
    # Passed back by hand: for an https backend the cookie is Secure, which
    # the http TestClient's jar would (correctly) refuse to send.
    nonce = resp.cookies.get(public_origin.OAUTH_NONCE_COOKIE)
    return params["redirect_uri"][0], params["state"][0], nonce


def _callback(client, state, nonce=None):
    headers = {"Cookie": f"{public_origin.OAUTH_NONCE_COOKIE}={nonce}"} if nonce else {}
    client.cookies.clear()
    return client.get("/auth/google/callback", params={"code": "abc", "state": state},
                      headers=headers, follow_redirects=False)


@pytest.mark.parametrize("referer, expected_redirect_uri, expected_frontend", [
    (f"{LOCAL_DEV}/login", "http://testserver/auth/google/callback", LOCAL_DEV),
    (f"{NGROK}/login", f"{NGROK}/api/auth/google/callback", NGROK),
    (f"{EVIL}/login", f"{NGROK}/api/auth/google/callback", NGROK),
])
def test_google_flow_round_trips_origin(
    client, google_configured, fake_google, referer, expected_redirect_uri, expected_frontend
):
    redirect_uri, state, nonce = _google_login(client, referer)
    assert redirect_uri == expected_redirect_uri

    resp = _callback(client, state, nonce)
    assert fake_google["token_redirect_uri"] == expected_redirect_uri
    assert resp.headers["location"].startswith(f"{expected_frontend}/oauth-callback?token=")


def test_google_callback_rejects_forged_state(client, google_configured, fake_google):
    # A validly signed state from someone else's login doesn't match this browser's nonce.
    _, _, nonce = _google_login(client, f"{LOCAL_DEV}/login")
    other_state = public_origin.create_oauth_state(public_origin.PublicOrigin(LOCAL_DEV, "http://x"))[0]
    resp = _callback(client, other_state, nonce)
    assert resp.headers["location"] == f"{NGROK}/login?error=google_state_invalid"
    assert "token_redirect_uri" not in fake_google


def test_google_callback_rejects_missing_state(client, google_configured, fake_google):
    client.cookies.clear()
    resp = client.get("/auth/google/callback", params={"code": "abc"}, follow_redirects=False)
    assert resp.headers["location"] == f"{NGROK}/login?error=google_state_invalid"


def test_google_state_tampered_signature_is_rejected():
    state, _ = public_origin.create_oauth_state(public_origin.default_origin())
    assert public_origin.decode_oauth_state(state[:-2] + ("AA" if state[-2:] != "AA" else "BB")) is None


def test_google_state_with_non_allowlisted_origin_is_rejected():
    state, _ = public_origin.create_oauth_state(public_origin.PublicOrigin(EVIL, EVIL))
    assert public_origin.decode_oauth_state(state) is None
