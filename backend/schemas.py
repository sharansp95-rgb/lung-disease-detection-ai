"""Pydantic response models for the prediction API."""

from typing import Optional

from pydantic import BaseModel, Field

# Caps on /chat payloads now that the endpoint is publicly reachable. The
# frontend already sends only the last 10 turns (the same window chat.py
# uses) and limits the textarea to MAX_CHAT_MESSAGE_CHARS, so normal use
# never hits these.
MAX_CHAT_MESSAGE_CHARS = 2000
MAX_CHAT_HISTORY_CONTENT_CHARS = 4000
MAX_CHAT_HISTORY_ITEMS = 50


class PredictionResponse(BaseModel):
    modality: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    heatmap_png_base64: str  # data URI, ready to drop into an <img src=...>
    report: str
    report_source: str  # "groq" or "template_fallback"
    warnings: list[str] = []


class ErrorResponse(BaseModel):
    detail: str


class ChatMessage(BaseModel):
    role: str = Field(max_length=20)  # "user" or "assistant"
    content: str = Field(max_length=MAX_CHAT_HISTORY_CONTENT_CHARS)


class PredictionContext(BaseModel):
    modality: Optional[str] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None


class ChatRequest(BaseModel):
    message: str = Field(max_length=MAX_CHAT_MESSAGE_CHARS)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_CHAT_HISTORY_ITEMS)
    prediction_context: Optional[PredictionContext] = None


class ChatResponse(BaseModel):
    reply: str
    source: str  # "groq" or "template_fallback"
