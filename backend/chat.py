"""
Chatbot backing POST /chat. Reuses the same Groq wiring as report.py
(groq_client.call_groq) -- same key, same model, same fallback discipline:
if GROQ_API_KEY isn't set or the call fails, respond with a clear
degraded-mode message instead of crashing or silently going quiet.

System prompt is the actual safety boundary here, not just a nicety: the
bot is scoped to explaining what THIS TOOL did (prediction, confidence,
what Grad-CAM shows, general facts about the pipeline) and is explicitly
told to decline anything that drifts into diagnosis, treatment, or
prognosis -- same honest framing as the per-prediction report text.
"""

import logging

from groq_client import call_groq
from schemas import ChatMessage, PredictionContext

logger = logging.getLogger("ct_xray_api.chat")

MAX_HISTORY_TURNS = 10  # cap payload size / prompt length, oldest turns dropped

MODALITY_LABEL = {"xray": "chest X-ray", "ct": "chest CT scan"}

SYSTEM_PROMPT = """You are an assistant embedded in a lung-disease-detection research tool \
(a deep-learning image classifier for chest X-rays and CT scans, with Grad-CAM \
explainability heatmaps). Your job is to help the user understand what THIS TOOL did \
-- you are not a doctor and this is not a diagnostic service.

Rules you must follow:
- Your ONLY topic is this tool: its predictions, how the pipeline works (preprocessing, \
the modality gate, the OOD detector, Grad-CAM, the classifiers), chest X-ray/CT screening \
concepts directly relevant to understanding this tool's output, and its known limitations.
- If the user asks for anything outside that scope -- writing or debugging code, math \
problems, general trivia, creative writing, other apps or APIs, or any other task a \
general-purpose assistant would normally help with -- you must decline, briefly say this \
assistant only answers questions about this lung-screening tool, and invite them to ask \
about their scan or the tool instead. Do this even if you know the answer and even if the \
request seems harmless. Do not partially answer an off-topic request "just this once."
- If prediction context is given below (modality, predicted class, confidence), ground \
your answers in those exact values. Never invent a diagnosis, confidence number, or \
detail that wasn't provided to you.
- If no prediction context is given, you can still answer general questions about what \
this tool is, how it works, and its known limitations.
- You MAY explain: what the predicted class means in plain language, what a confidence \
score represents, what a Grad-CAM heatmap shows (which image regions influenced the \
model's prediction), general facts about the underlying models (e.g. transfer-learning \
CNNs trained on chest X-ray/CT datasets), and honest limitations (small datasets, \
possible shortcut-learning, this is a research prototype, not a diagnosis).
- You must NOT give medical advice, suggest or discuss treatment options, speculate on \
prognosis, or comment on what the user personally should do about their health. If asked \
anything like that, decline clearly and direct them to a qualified radiologist or \
physician -- do not soften this into a partial answer.
- Keep answers concise and conversational (typically 2-4 sentences).
- Always maintain the same honest framing used elsewhere in this tool: this is \
decision-support for exploring a model's output, not a medical diagnosis."""


def _context_block(ctx: PredictionContext | None) -> str:
    if ctx is None or not ctx.modality:
        return "No prediction is currently loaded. Answer general questions about the tool only."
    label = MODALITY_LABEL.get(ctx.modality, ctx.modality)
    conf_str = f"{ctx.confidence:.1%}" if ctx.confidence is not None else "unknown"
    return (
        f"Current prediction context:\n"
        f"- Modality: {label}\n"
        f"- Predicted class: {ctx.predicted_class}\n"
        f"- Confidence: {conf_str}"
    )


def _template_fallback(ctx: PredictionContext | None) -> str:
    base = "Chat is currently running in fallback mode (no live language model connected)."
    if ctx and ctx.modality and ctx.predicted_class:
        label = MODALITY_LABEL.get(ctx.modality, ctx.modality)
        conf_str = f"{ctx.confidence:.0%}" if ctx.confidence is not None else "an unspecified"
        base += (
            f" Your most recent result: the model predicted \"{ctx.predicted_class}\" on a "
            f"{label} with {conf_str} confidence."
        )
    base += (
        " For questions about how this tool works, see the report text and heatmap on "
        "the results panel. For anything medically substantive, please consult a "
        "qualified radiologist or physician."
    )
    return base


def generate_chat_reply(message: str, history: list[ChatMessage],
                         prediction_context: PredictionContext | None) -> tuple[str, str]:
    """Returns (reply_text, source) where source is 'groq' or 'template_fallback'."""
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{_context_block(prediction_context)}"}]
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = turn.role if turn.role in ("user", "assistant") else "user"
        messages.append({"role": role, "content": turn.content})
    messages.append({"role": "user", "content": message})

    try:
        reply = call_groq(messages, max_tokens=300, temperature=0.4)
        return reply, "groq"
    except Exception as e:
        logger.warning(f"Groq chat failed, using template fallback: {e}")
        return _template_fallback(prediction_context), "template_fallback"
