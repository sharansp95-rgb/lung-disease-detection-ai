"""
Low-level Groq chat-completion call, shared by report.py (single-turn
report generation) and chat.py (multi-turn chatbot) -- centralized so the
HTTP call, timeout, and model identifier live in exactly one place.

Model note: originally spec'd as Llama 3.3 70B (llama-3.3-70b-versatile).
That model no longer exists on Groq's API as of this writing -- confirmed
via GET /openai/v1/models on this account, which returns zero Llama models
at all (Groq's catalog has shifted to OpenAI's open-weight gpt-oss models
and Groq's own "compound" systems). Using openai/gpt-oss-120b instead, the
largest general-purpose instruct model currently available. If Llama
access returns later, swap GROQ_MODEL back.
"""

import os

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"  # was llama-3.3-70b-versatile, see module docstring
GROQ_TIMEOUT_SECONDS = 15


def call_groq(messages: list[dict], max_tokens: int = 200, temperature: float = 0.3) -> str:
    """messages: list of {"role": ..., "content": ...} dicts (system/user/assistant).
    Raises on any failure (missing key, network error, non-2xx, bad response shape)
    -- callers are expected to catch and fall back, same discipline used
    throughout this backend."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=GROQ_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
