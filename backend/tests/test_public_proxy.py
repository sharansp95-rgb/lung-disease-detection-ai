"""
Covers the changes for running behind the local /api proxy + ngrok
(docs/DEPLOYMENT_PUBLIC.md): real-client-IP rate-limit keying, the /chat rate
limit, and /chat payload caps.
"""

import pytest
from starlette.requests import Request

from rate_limit import client_ip, limiter
from schemas import MAX_CHAT_HISTORY_ITEMS, MAX_CHAT_MESSAGE_CHARS


def _request(peer: str, forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers, "client": (peer, 12345)})


# ------------------------------------------------------------ client_ip --

def test_client_ip_uses_peer_when_no_forwarded_header():
    assert client_ip(_request("127.0.0.1")) == "127.0.0.1"


def test_client_ip_trusts_forwarded_for_from_localhost_proxy():
    # ngrok sets the visitor's IP, Vite's proxy appends its own loopback peer
    assert client_ip(_request("127.0.0.1", "203.0.113.7, 127.0.0.1")) == "203.0.113.7"
    assert client_ip(_request("127.0.0.1", "203.0.113.7, ::1")) == "203.0.113.7"


def test_client_ip_ignores_client_spoofed_leftmost_entries():
    assert client_ip(_request("127.0.0.1", "1.1.1.1, 203.0.113.7, 127.0.0.1")) == "203.0.113.7"


def test_client_ip_ignores_forwarded_for_from_non_local_peer():
    assert client_ip(_request("198.51.100.20", "203.0.113.7")) == "198.51.100.20"


def test_client_ip_stops_at_malformed_hop():
    assert client_ip(_request("127.0.0.1", "203.0.113.7, not-an-ip, 127.0.0.1")) == "127.0.0.1"


def test_client_ip_only_loopback_chain_falls_back_to_peer():
    assert client_ip(_request("127.0.0.1", "127.0.0.1")) == "127.0.0.1"


# ----------------------------------------------------------------- /chat --

@pytest.fixture
def fresh_limits():
    limiter.reset()
    yield
    limiter.reset()


def test_chat_falls_back_when_groq_unavailable(client, fresh_limits):
    resp = client.post("/chat", json={"message": "What does the heatmap show?", "history": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"] == "template_fallback"


def test_chat_uses_mocked_groq_reply(client, fresh_limits, monkeypatch):
    import chat

    monkeypatch.setattr(chat, "call_groq", lambda *a, **k: "mocked reply")
    resp = client.post("/chat", json={"message": "hello", "history": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json() == {"reply": "mocked reply", "source": "groq"}


def test_chat_empty_message_still_400(client, fresh_limits):
    resp = client.post("/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_chat_rejects_overlong_message(client, fresh_limits):
    resp = client.post("/chat", json={"message": "x" * (MAX_CHAT_MESSAGE_CHARS + 1)})
    assert resp.status_code == 422


def test_chat_rejects_oversized_history(client, fresh_limits):
    history = [{"role": "user", "content": "hi"}] * (MAX_CHAT_HISTORY_ITEMS + 1)
    resp = client.post("/chat", json={"message": "hello", "history": history})
    assert resp.status_code == 422


def test_chat_rate_limited_per_ip(client, fresh_limits):
    statuses = [client.post("/chat", json={"message": "hello"}).status_code for _ in range(21)]
    assert statuses[:20] == [200] * 20
    assert statuses[20] == 429
