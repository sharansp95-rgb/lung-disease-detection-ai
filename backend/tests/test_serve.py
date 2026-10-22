"""
Single-origin app (serve.py): frontend build at "/", API under "/api", SPA
fallback for client routes, and what public_origin/rate_limit see through it.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

import config
import rate_limit
import serve
from rate_limit import limiter

SPACE = "https://user-space.hf.space"


@pytest.fixture(scope="module")
def site(tmp_path_factory, client):  # `client` first, so models are already loaded
    dist = tmp_path_factory.mktemp("dist")
    (dist / "index.html").write_text("<!doctype html><title>spa</title>")
    (dist / "assets").mkdir()
    (dist / "assets" / "app.js").write_text("console.log(1)")
    with TestClient(serve.create_app(dist)) as c:
        yield c


@pytest.fixture
def space_config(monkeypatch):
    monkeypatch.setattr(config, "FRONTEND_URL", SPACE)
    monkeypatch.setattr(config, "BACKEND_URL", f"{SPACE}/api")
    monkeypatch.setattr(config, "ALLOWED_PUBLIC_ORIGINS", [SPACE])
    limiter.reset()
    yield
    limiter.reset()


def test_root_serves_index(site):
    resp = site.get("/")
    assert resp.status_code == 200
    assert "<title>spa</title>" in resp.text


@pytest.mark.parametrize("path", ["/login", "/results", "/reset-password/abc123", "/oauth-callback?token=x"])
def test_client_routes_fall_back_to_index(site, path):
    resp = site.get(path)
    assert resp.status_code == 200
    assert "<title>spa</title>" in resp.text


def test_assets_served_and_missing_assets_404(site):
    assert site.get("/assets/app.js").text == "console.log(1)"
    assert site.get("/assets/missing.js").status_code == 404


def test_api_under_prefix(site):
    resp = site.get("/api/health")
    assert resp.status_code == 200
    assert set(resp.json()["models_loaded"]) == {"xray", "ct", "gate", "xray_sanity", "ct_sanity"}
    assert site.get("/api/auth/me").status_code == 401


def test_verification_link_points_at_api(site, space_config, monkeypatch):
    import auth_routes

    sent = {}
    monkeypatch.setattr(auth_routes, "send_verification_email", lambda to, link: sent.update(link=link))
    resp = site.post("/api/auth/signup", headers={"Origin": SPACE},
                     json={"email": f"serve-{uuid.uuid4().hex[:12]}@example.com",
                           "password": "correct-horse-battery-staple"})
    assert resp.status_code == 200, resp.text
    assert sent["link"].startswith(f"{SPACE}/api/auth/verify/")

    token = sent["link"].rsplit("/", 1)[-1]
    verify = site.get(f"/api/auth/verify/{token}", follow_redirects=False)
    assert verify.headers["location"] == f"{SPACE}/email-verified"


def test_reset_link_points_at_frontend(site, space_config, monkeypatch):
    import auth_routes

    email = f"serve-{uuid.uuid4().hex[:12]}@example.com"
    site.post("/api/auth/signup", json={"email": email, "password": "correct-horse-battery-staple"})
    sent = {}
    monkeypatch.setattr(auth_routes, "send_password_reset_email", lambda to, link: sent.update(link=link))
    resp = site.post("/api/auth/forgot-password", headers={"Referer": f"{SPACE}/forgot-password"},
                     json={"email": email})
    assert resp.status_code == 200
    assert sent["link"].startswith(f"{SPACE}/reset-password/")


def test_google_redirect_uri_and_nonce_cookie(site, space_config, monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    monkeypatch.setattr(config, "GOOGLE_OAUTH_CONFIGURED", True)
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "client-id")
    resp = site.get("/api/auth/google/login", headers={"Referer": f"{SPACE}/login"}, follow_redirects=False)
    assert resp.status_code in (302, 307)
    query = parse_qs(urlsplit(resp.headers["location"]).query)
    assert query["redirect_uri"] == [f"{SPACE}/api/auth/google/callback"]
    cookie = resp.headers["set-cookie"]
    assert "google_oauth_nonce=" in cookie and "Path=/" in cookie and "Secure" in cookie


def test_direct_backend_base_includes_mount_prefix():
    from starlette.requests import Request

    import public_origin

    req = Request({"type": "http", "method": "GET", "path": "/api/x", "root_path": "/api", "scheme": "http",
                   "query_string": b"", "headers": [], "client": ("203.0.113.9", 1), "server": ("127.0.0.1", 7860)})
    assert public_origin._direct_backend_base(req) == "http://127.0.0.1:7860/api"


# --------------------------------------------- trusted proxy networks (HF) --

def _request(peer, forwarded_for):
    from starlette.requests import Request

    return Request({"type": "http", "method": "GET", "path": "/", "client": (peer, 1),
                    "headers": [(b"x-forwarded-for", forwarded_for.encode())]})


@pytest.fixture
def private_proxies(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUSTED_PROXY_NETWORKS",
                        rate_limit._parse_networks(["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]))


def test_hf_chain_picks_rightmost_public_hop(private_proxies):
    assert rate_limit.client_ip(_request("10.1.2.3", "203.0.113.7, 10.4.5.6")) == "203.0.113.7"


def test_hf_chain_ignores_spoofed_leftmost(private_proxies):
    assert rate_limit.client_ip(_request("10.1.2.3", "1.1.1.1, 203.0.113.7, 10.4.5.6")) == "203.0.113.7"


def test_untrusted_peer_header_ignored(private_proxies):
    assert rate_limit.client_ip(_request("198.51.100.20", "203.0.113.7")) == "198.51.100.20"


def test_all_private_chain_falls_back_to_peer(private_proxies):
    assert rate_limit.client_ip(_request("10.1.2.3", "10.9.9.9")) == "10.1.2.3"
