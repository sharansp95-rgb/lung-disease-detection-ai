"""
Scan history: saving a prediction for a logged-in user, and retrieving
past results. Anonymous users can still predict (see main.py's
get_current_user_optional usage) -- they just don't get history saved.
"""

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth_routes import get_current_user
from db import get_db

router = APIRouter(prefix="/scans", tags=["scans"])


class ScanHistoryItem(BaseModel):
    id: int
    modality: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    heatmap_png_base64: str
    report: str | None
    report_source: str | None
    created_at: str


def save_scan(user_id: int, modality: str, predicted_class: str, confidence: float,
              class_probabilities: dict, heatmap_png_base64: str, report: str, report_source: str):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO scans
               (user_id, modality, predicted_class, confidence, class_probabilities,
                heatmap_png_base64, report, report_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, modality, predicted_class, confidence, json.dumps(class_probabilities),
             heatmap_png_base64, report, report_source),
        )


@router.get("/history", response_model=list[ScanHistoryItem])
def get_history(user: dict = Depends(get_current_user)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM scans WHERE user_id = ? ORDER BY created_at DESC", (user["id"],)
        ).fetchall()

    return [
        ScanHistoryItem(
            id=r["id"], modality=r["modality"], predicted_class=r["predicted_class"],
            confidence=r["confidence"], class_probabilities=json.loads(r["class_probabilities"]),
            heatmap_png_base64=r["heatmap_png_base64"], report=r["report"],
            report_source=r["report_source"], created_at=r["created_at"],
        )
        for r in rows
    ]
