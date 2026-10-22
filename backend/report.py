"""
Plain-language report generation via Groq. Falls back to a template message
if GROQ_API_KEY isn't set or the API call fails for any reason -- the
prediction endpoint must keep working without a key during development,
per Phase 4 spec.

Deliberately conservative prompt: explain the model's output in plain
language only, never phrase anything as a diagnosis or treatment
recommendation. Both the LLM path and the template fallback carry the same
disclaimer, since the disclaimer matters regardless of which one produced
the text.
"""

import logging

from groq_client import call_groq

logger = logging.getLogger("ct_xray_api.report")

DISCLAIMER = (
    "This is an automated model output, not a medical diagnosis. "
    "Results should be reviewed by a qualified radiologist or physician."
)

MODALITY_LABEL = {"xray": "chest X-ray", "ct": "chest CT scan"}


def _template_report(modality: str, predicted_class: str, confidence: float) -> str:
    label = MODALITY_LABEL.get(modality, modality)
    return (
        f"The model analyzed this {label} and predicted \"{predicted_class}\" "
        f"with {confidence:.0%} confidence. The highlighted (Grad-CAM) regions "
        f"in the accompanying heatmap show which parts of the image most "
        f"influenced this prediction. {DISCLAIMER}"
    )


def _groq_report(modality: str, predicted_class: str, confidence: float,
                  class_probabilities: dict[str, float]) -> str:
    label = MODALITY_LABEL.get(modality, modality)
    probs_str = ", ".join(f"{k}: {v:.1%}" for k, v in class_probabilities.items())

    prompt = (
        f"A machine learning model analyzed a {label} image and produced this "
        f"classification result:\n"
        f"Predicted class: {predicted_class}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Full class probabilities: {probs_str}\n\n"
        f"Write a short (2-3 sentence), plain-language explanation of this "
        f"result for a non-medical reader. Do NOT phrase this as a diagnosis, "
        f"do NOT suggest treatment, do NOT speculate beyond what the numbers "
        f"say. Just explain what the model output means in accessible terms."
    )

    text = call_groq([{"role": "user", "content": prompt}], max_tokens=1024, temperature=0.3)
    return f"{text}\n\n{DISCLAIMER}"


def generate_report(modality: str, predicted_class: str, confidence: float,
                     class_probabilities: dict[str, float]) -> tuple[str, str]:
    """Returns (report_text, source) where source is 'groq' or 'template_fallback'."""
    try:
        text = _groq_report(modality, predicted_class, confidence, class_probabilities)
        return text, "groq"
    except Exception as e:
        logger.warning(f"Groq report generation failed, using template fallback: {e}")
        return _template_report(modality, predicted_class, confidence), "template_fallback"
