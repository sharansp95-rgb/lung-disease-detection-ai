"""
groq_client.call_groq's handling of Groq responses (HTTP mocked -- tests
never call the real API), and the report fallback when a reply is cut off.
"""

import pytest

import groq_client
import report


class _Resp:
    def __init__(self, finish_reason, content="A plain-language explanation."):
        self._body = {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def test_complete_reply_returned(groq_key, monkeypatch):
    monkeypatch.setattr(groq_client.requests, "post", lambda *a, **k: _Resp("stop", "  done  "))
    assert groq_client.call_groq([{"role": "user", "content": "hi"}]) == "done"


def test_truncated_reply_raises(groq_key, monkeypatch):
    monkeypatch.setattr(groq_client.requests, "post", lambda *a, **k: _Resp("length", "cut off mid-sen"))
    with pytest.raises(groq_client.GroqTruncated):
        groq_client.call_groq([{"role": "user", "content": "hi"}])


def test_report_falls_back_to_template_when_truncated(groq_key, monkeypatch):
    # conftest mocks report.call_groq; route it to the real one over a fake HTTP reply
    monkeypatch.setattr(groq_client.requests, "post", lambda *a, **k: _Resp("length", "cut off mid-sen"))
    monkeypatch.setattr(report, "call_groq", groq_client.call_groq)
    text, source = report.generate_report("xray", "Normal", 0.91, {"Normal": 0.91, "COVID-19": 0.09})
    assert source == "template_fallback"
    assert "cut off" not in text


def test_report_budget_leaves_room_for_reasoning(groq_key, monkeypatch):
    seen = {}

    def fake_post(*_a, json, **_k):
        seen.update(json)
        return _Resp("stop")

    monkeypatch.setattr(groq_client.requests, "post", fake_post)
    monkeypatch.setattr(report, "call_groq", groq_client.call_groq)
    _, source = report.generate_report("ct", "Benign", 0.8, {"Benign": 0.8})
    assert source == "groq"
    assert seen["max_tokens"] >= 1024
