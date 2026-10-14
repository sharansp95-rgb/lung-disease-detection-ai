"""
FastAPI backend serving the two verified disease classifiers
(xray_txrv_densenet121_best.pt, ct_best_model_phase35_v2.pt), plus the
sanity gates, modality gate and lung-segmentation model they rely on (see
CHECKPOINTS / SEG_CHECKPOINT below). All models lazy-load, with their
reproduction checks, on the first prediction request.

--- Route design: two separate routes, not one /predict + modality field ---
Chose POST /predict/xray and POST /predict/ct over a single /predict with a
"modality" form field. Reasons:
  1. Each route has its own fixed, self-documenting response schema (exact
     class list per modality shows up correctly in OpenAPI docs) instead of
     one endpoint whose valid class_probabilities keys depend on a field
     value.
  2. A single endpoint invites a whole class of bugs where the modality
     field and the actual image content disagree (client-side typo, wrong
     default, etc.) and the server silently runs the wrong model on it.
     Two routes make the model selection a URL-routing decision, not a
     runtime branch on user-supplied data -- one less thing to validate.
  3. It maps cleanly onto "reject mismatched modality/image combinations":
     the route already commits to a modality, so mismatch detection is
     just "does this image look like a plausible input for the model this
     route already calls" -- not "does the field match the file too."

--- Mismatched-modality / OOD detection: feature-space distance, not confidence ---
Originally this used a softmax-confidence floor (reject if the model's top
prediction was below ~random-chance-plus-margin). That was tested against a
genuine out-of-distribution image (a screenshot) and failed: the CT model
scored it 93% confidence as "Malignant" with zero warning. Discriminative
classifiers have no "none of the above" option baked into softmax, so
confidence alone doesn't detect inputs unlike anything in training.

Replaced with feature-space OOD detection (ood_detector.py): each modality's
training set was embedded once (scripts/compute_ood_thresholds.py) via the
model's own penultimate layer, a centroid computed, and a rejection
threshold set at the 99th-percentile training-image distance from that
centroid. At inference, an incoming image's embedding distance from the
centroid is checked BEFORE the classifier ever runs -- if it's farther out
than 99% of real training images ever were, it's rejected outright. The
softmax-confidence check still runs, but only as a soft "uncertain result"
warning on in-distribution images, not as the mismatch detector.

Run locally:
    uvicorn main:app --reload --port 8000
"""

import io
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

import config
import groq_client
from auth_routes import require_verified_if_logged_in
from auth_routes import router as auth_router
from chat import generate_chat_reply
from db import init_db
from gradcam_utils import predict_with_heatmap, preprocess_image, predict_with_heatmap_xrv, preprocess_image_xrv
from models import ModelBundle, ReproductionCheckFailed, is_xrv_backbone, load_model_bundle
from ood_detector import OODDetector
from rate_limit import limiter
from report import generate_report
from scans_routes import router as scans_router
from scans_routes import save_scan
from schemas import ChatRequest, ChatResponse, PredictionResponse
from seg_utils import SegBundle, SegReproductionCheckFailed, compute_lung_crop, load_seg_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ct_xray_api")

ROOT = Path(__file__).resolve().parent.parent

GATE_LOG_FILE = ROOT / "logs" / "modality_gate_debug.log"
GATE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
gate_file_handler = logging.FileHandler(GATE_LOG_FILE, encoding="utf-8")
gate_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
gate_logger = logging.getLogger("modality_gate_debug")
gate_logger.setLevel(logging.INFO)
gate_logger.addHandler(gate_file_handler)
CHECKPOINTS = {
    "xray": ROOT / "checkpoints" / "xray_txrv_densenet121_best.pt",
    "ct": ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt",
    "gate": ROOT / "checkpoints" / "modality_gate_best_model_v2.pt",
    "xray_sanity": ROOT / "checkpoints" / "xray_sanity_best.pt",
    "ct_sanity": ROOT / "checkpoints" / "ct_sanity_best.pt",
}
DATA_DIRS = {
    "xray": ROOT / "data" / "xray",
    "ct": ROOT / "data" / "ct_scan",
    "gate": ROOT / "data" / "modality_gate",
    "xray_sanity": ROOT / "data" / "xray_sanity",
    "ct_sanity": ROOT / "data" / "ct_sanity",
}
DISEASE_MODALITIES = ("xray", "ct")
MODEL_NAMES = ("xray", "ct", "gate", "xray_sanity", "ct_sanity")

SEG_CHECKPOINT = ROOT / "checkpoints" / "xray_seg_model.pt"
SEG_DATA_DIR = ROOT / "seg_data"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# A real chest X-ray/CT slice, even a low-contrast one, has grayscale pixel
# std-dev in the low double digits at minimum (measured across this
# project's own test data: 13.7-87.6 depending on class/modality). A blank
# or solid-colour upload measures 0, and even a blank scan with heavy
# compression noise stays under ~5. The OOD detector (feature-space
# distance from a training centroid) does NOT reliably catch this case --
# it was calibrated against realistic wrong-content images (screenshots,
# wrong modality), and a spatially-uniform image can land unexpectedly
# close to the training centroid after global average pooling. This is a
# separate, cheap, content-blind guard for that specific gap.
BLANK_IMAGE_STD_THRESHOLD = 10.0

# Real chest X-ray/CT files are grayscale at the source, so even after
# being re-encoded as RGB their R/G/B channels stay identical per pixel
# (measured 0.0 mean channel spread across sampled train images of both
# modalities, including after a JPEG re-encode round trip). A genuinely
# colorful upload -- a photo, an infographic, a "COVID-19" illustration --
# does not. This matters because preprocess_image() converts every input
# to grayscale before it reaches the sanity gate, the modality gate, or
# the classifier, so none of those models ever see the color that would
# otherwise be the cheapest signal that an upload isn't a real scan. This
# check runs on the original RGB image, before that conversion happens,
# so it doesn't share that blind spot. Threshold picked well above the
# real-scan floor of 0.0 and the mild off-white tint synthetic "scanned
# document" negatives can have (up to ~15), so it only fires on genuine
# color content.
COLOR_CONTENT_THRESHOLD = 20.0

# Confidence floor per modality, used as the "mismatched modality/image"
# detector: this is the only signal actually available without ground
# truth -- if the model isn't meaningfully better than chance on an input,
# treat that as evidence the image may not belong to this modality (or is
# some other out-of-distribution input). Not a perfect detector, but a
# real, honestly-grounded one. Baselines: 1/4=0.25 for X-ray, 1/3=0.33 for CT.
MIN_CONFIDENCE = {"xray": 0.35, "ct": 0.40}

app = FastAPI(title="Lung Disease Detection API", version="0.1.0")

# Per-client-IP rate limiting (slowapi/limits, in-memory -- fine for a
# single-process dev/demo deployment; a multi-worker production deploy
# would need a shared backend like Redis instead). Guards the two route
# groups actually worth abuse-protecting: prediction (expensive, runs 2+
# model forward passes per call) and auth (brute-force/enumeration).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: allows the Vite dev server (localhost:5173 by default) to call this
# API directly from the browser. Origins come from CORS_ORIGINS in
# backend/.env; behind the public /api proxy requests are same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(scans_router)

_bundles: dict[str, ModelBundle] = {}
_ood_detectors: dict[str, OODDetector] = {}
_seg_bundle: Optional[SegBundle] = None
_models_load_lock = threading.Lock()
_models_loaded = False


def _ensure_models_loaded():
    """Lazily load all model bundles, OOD detectors, and the segmentation
    bundle on the first prediction request.  Thread-safe: concurrent
    requests that arrive before loading finishes will block on the lock
    and then see the already-populated dicts."""
    global _models_loaded, _seg_bundle

    if _models_loaded:
        return

    with _models_load_lock:
        # Double-check after acquiring the lock -- another thread may have
        # loaded everything while we were waiting.
        if _models_loaded:
            return

        logger.info("Lazy-loading models on first prediction request...")

        for name in MODEL_NAMES:
            try:
                _bundles[name] = load_model_bundle(name, CHECKPOINTS[name], DATA_DIRS[name])
            except (ReproductionCheckFailed, FileNotFoundError) as exc:
                logger.error(f"Failed to load model '{name}': {exc}")
                raise

        for modality in DISEASE_MODALITIES:
            _ood_detectors[modality] = OODDetector(modality, _bundles[modality].device)

        try:
            _seg_bundle = load_seg_bundle(SEG_CHECKPOINT, SEG_DATA_DIR)
        except (SegReproductionCheckFailed, FileNotFoundError) as exc:
            logger.error(f"Failed to load segmentation model: {exc}")
            raise

        _models_loaded = True
        logger.info("All models loaded, verified, and OOD detectors ready.")


@app.on_event("startup")
def startup_load_models():

    logger.info("Startup: initializing database...")
    init_db()

    logger.info("Startup: preloading models...")
    _ensure_models_loaded()

    if config.GROQ_API_KEY:
        key = config.GROQ_API_KEY
        masked = f"{key[:7]}...{key[-4:]}" if len(key) > 12 else "***"
        logger.info(f"GROQ_API_KEY is set ({masked}) -- report/chat generation will use Groq ({groq_client.GROQ_MODEL}).")
    else:
        logger.warning("GROQ_API_KEY is not set -- report generation will use the template fallback.")

    if config.GOOGLE_OAUTH_CONFIGURED:
        logger.info("Google OAuth is configured (client ID + secret present).")
    else:
        logger.warning(
            "Google OAuth is NOT fully configured (missing GOOGLE_CLIENT_SECRET) -- "
            "'Continue with Google' will return a clear 501 until it's set. See backend/README_AUTH.md."
        )

    if config.EMAIL_CONFIGURED:
        logger.info(f"Gmail SMTP is configured ({config.GMAIL_ADDRESS}) -- verification/reset emails will send for real.")
    else:
        logger.warning(
            "Gmail SMTP is NOT configured -- verification/reset links will be printed to this "
            "console instead of emailed. See backend/README_AUTH.md."
        )

    logger.info(f"FRONTEND_URL = {config.FRONTEND_URL}")
    logger.info(f"BACKEND_URL = {config.BACKEND_URL}")
    logger.info(f"Google OAuth redirect_uri = {config.GOOGLE_REDIRECT_URI} (default)")
    logger.info(f"Allowed public origins = {', '.join(config.ALLOWED_PUBLIC_ORIGINS)}")
    logger.info(f"CORS origins = {', '.join(config.CORS_ORIGINS)}")

    logger.info("Startup complete -- models will lazy-load on first prediction request.")


def _sanitized_500(_request, exc: Exception):
    logger.exception("Unhandled exception while processing request")
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred while processing the request."})


app.add_exception_handler(Exception, _sanitized_500)


def _read_and_validate_image(file: UploadFile) -> Image.Image:
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_BYTES // (1024*1024)}MB.")
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # verify() invalidates the image object for further use -- reopen
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    gray_std = float(np.array(img.convert("L"), dtype=np.float32).std())
    if gray_std < BLANK_IMAGE_STD_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image appears to be blank or has no visible content. Please upload an actual scan image.",
        )

    rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
    color_dev = float((rgb_arr.max(axis=2) - rgb_arr.min(axis=2)).mean())
    if color_dev > COLOR_CONTENT_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail="This image has color content that a real X-ray or CT scan wouldn't have. Please upload an actual scan image.",
        )

    return img


MODALITY_LABEL = {"xray": "chest X-ray", "ct": "chest CT scan"}


def _sanity_predict(modality: str, input_tensor) -> str:
    """Runs the per-modality sanity gate (resnet18, binary scan-vs-not_scan,
    trained on real scans as positives against a pool of photos/scanned
    documents/screenshots as negatives). Returns 'scan' or 'not_scan'."""
    bundle = _bundles[f"{modality}_sanity"]
    with bundle.lock:
        bundle.model.eval()
        with torch.no_grad():
            logits = bundle.model(input_tensor)
            pred_idx = int(logits.argmax(dim=1).item())
    return bundle.class_names[pred_idx]


def _gate_predict(input_tensor) -> tuple[str, float, dict[str, float], dict[str, float]]:
    """Runs the modality-gate model (resnet18, xray-vs-ct, 100% test acc)
    on an already-preprocessed tensor.
    Returns (pred_label, top_confidence, class_probabilities, raw_logits)."""
    gate_bundle = _bundles["gate"]
    with gate_bundle.lock:
        gate_bundle.model.eval()
        with torch.no_grad():
            logits = gate_bundle.model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(logits.argmax(dim=1).item())
    pred_label = gate_bundle.class_names[pred_idx]
    top_conf = float(probs[pred_idx].item())
    class_probs = {cls_name: float(probs[i].item()) for i, cls_name in enumerate(gate_bundle.class_names)}
    raw_logits = {cls_name: float(logits[0, i].item()) for i, cls_name in enumerate(gate_bundle.class_names)}
    return pred_label, top_conf, class_probs, raw_logits


def _run_prediction(modality: str, file: UploadFile, user: Optional[dict]) -> PredictionResponse:
    _ensure_models_loaded()

    bundle = _bundles.get(modality)
    ood_detector = _ood_detectors.get(modality)
    if bundle is None or ood_detector is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    image = _read_and_validate_image(file)
    input_tensor = preprocess_image(image, bundle.device)

    # Sanity gate runs FIRST, before the modality gate: the modality gate is
    # a closed-set xray-vs-ct classifier with no "neither" option, so on a
    # genuinely non-medical upload (a photo, a scanned document, a
    # screenshot) it -- and the disease classifier behind it -- will
    # confidently pick one of their known classes anyway (verified: a
    # synthetic random photo scored 92%+ "xray" from the gate and the
    # feature-distance OOD check still passed it through, since OOD
    # distance and softmax confidence for that class of input land well
    # inside the range real training images occupy). This gate is a
    # separately trained binary classifier per modality (real scans vs. a
    # pool of photos/documents/screenshots) specifically for that gap.
    sanity_prediction = _sanity_predict(modality, input_tensor)
    if sanity_prediction != "scan":
        label = MODALITY_LABEL.get(modality, modality)
        raise HTTPException(
            status_code=422,
            detail=f"This does not look like a {label}. Please upload an actual scan image.",
        )

    # Modality gate runs next, before OOD and before the disease classifier:
    # a lightweight resnet18 trained specifically to distinguish X-ray vs CT
    # (100% val/test accuracy -- the two domains are visually very distinct).
    # This exists because the OOD detector, while catching some
    # out-of-distribution input, was tested and confirmed to NOT catch a
    # real CT scan submitted to the X-ray route (and vice versa) -- both
    # got confidently misclassified as a real disease with no warning. The
    # gate model is a direct, specific fix for exactly that failure mode.
    t_shape = list(input_tensor.shape)
    t_dtype = str(input_tensor.dtype)
    t_min = float(input_tensor.min().item())
    t_max = float(input_tensor.max().item())
    t_mean = float(input_tensor.mean().item())
    t_std = float(input_tensor.std().item())

    gate_prediction, gate_conf, gate_probs, gate_logits = _gate_predict(input_tensor)

    gate_logger.info(
        f"Request endpoint=/predict/{modality} filename='{getattr(file, 'filename', 'unknown')}' | "
        f"Input tensor shape={t_shape} dtype={t_dtype} min={t_min:.4f} max={t_max:.4f} mean={t_mean:.4f} std={t_std:.4f} | "
        f"Gate pred={gate_prediction} conf={gate_conf:.4f} | "
        f"Softmax probs={gate_probs} | Logits={gate_logits}"
    )

    # Guardrail 1: Low-confidence rejection (threshold from config.MODALITY_GATE_CONFIDENCE_THRESHOLD)
    if gate_conf < config.MODALITY_GATE_CONFIDENCE_THRESHOLD:
        label = MODALITY_LABEL.get(modality, modality)
        gate_logger.warning(
            f"Modality gate UNCERTAIN for /predict/{modality}: top conf {gate_conf:.4f} < "
            f"threshold {config.MODALITY_GATE_CONFIDENCE_THRESHOLD:.4f}. Returning 422."
        )
        raise HTTPException(
            status_code=422,
            detail=f"Modality uncertain — please confirm this is a {label} (confidence: {gate_conf:.1%}).",
        )

    # Guardrail 2: Hard mismatch rejection for high-confidence wrong modality
    if gate_prediction != modality:
        gate_logger.warning(
            f"Modality gate MISMATCH for /predict/{modality}: predicted {gate_prediction} "
            f"with conf {gate_conf:.4f} >= threshold {config.MODALITY_GATE_CONFIDENCE_THRESHOLD:.4f}. Returning 422."
        )
        expected_label = MODALITY_LABEL.get(modality, modality)
        actual_label = MODALITY_LABEL.get(gate_prediction, gate_prediction)
        raise HTTPException(
            status_code=422,
            detail=(
                f"This looks like a {actual_label}, not a {expected_label} -- "
                f"please use the {'CT' if gate_prediction == 'ct' else 'X-ray'} tab instead."
            ),
        )

    # Both gates confirmed the upload matches the requested modality.
    # Apply modality-specific lung crop only now, on confirmed X-rays:
    if modality == "xray":
        image, _ = compute_lung_crop(image, _seg_bundle)
        input_tensor = (
            preprocess_image_xrv(image, bundle.device) if is_xrv_backbone(bundle.backbone)
            else preprocess_image(image, bundle.device)
        )

    # OOD check + classify/Grad-CAM run inside ONE continuous hold of
    # bundle.lock: both touch the same shared model object (OOD via
    # forward_features(), Grad-CAM via forward hooks), and holding the lock
    # across the whole sequence -- not just each call individually --
    # guarantees no other request's Grad-CAM hooks can be registered on
    # this model in between. See ModelBundle.lock docstring in models.py
    # for the concurrency bug this fixes.
    with bundle.lock:
        # OOD check: reject inputs whose penultimate-layer embedding sits
        # farther from the training-set centroid than the 99th-percentile
        # training distance. This replaced a softmax-confidence heuristic
        # that was tested and failed (a screenshot scored 93% confidence as
        # "Malignant") -- confidence alone doesn't detect out-of-distribution
        # input for a discriminative classifier.
        is_ood, distance = ood_detector.check(bundle.model, input_tensor)
        if is_ood:
            label = MODALITY_LABEL.get(modality, modality)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"This does not look like a {label}. "
                    f"(feature-space distance {distance:.1f} exceeds threshold {ood_detector.threshold:.1f})"
                ),
            )

        pred_idx, probs, heatmap_uri = (
            predict_with_heatmap_xrv(bundle.model, input_tensor) if is_xrv_backbone(bundle.backbone)
            else predict_with_heatmap(bundle.model, input_tensor)
        )

    predicted_class = bundle.class_names[pred_idx]
    confidence = float(probs[pred_idx])
    class_probabilities = {name: float(p) for name, p in zip(bundle.class_names, probs)}

    warnings = []
    borderline_ceiling = MIN_CONFIDENCE[modality] + 0.15
    if confidence < borderline_ceiling:
        warnings.append(
            f"Confidence ({confidence:.1%}) is low -- treat this prediction as uncertain."
        )

    report_text, report_source = generate_report(modality, predicted_class, confidence, class_probabilities)

    if user is not None:
        save_scan(user["id"], modality, predicted_class, confidence, class_probabilities,
                  heatmap_uri, report_text, report_source)

    return PredictionResponse(
        modality=modality,
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
        heatmap_png_base64=heatmap_uri,
        report=report_text,
        report_source=report_source,
        warnings=warnings,
    )


@app.post("/predict/xray", response_model=PredictionResponse)
@limiter.limit("20/minute")
def predict_xray(request: Request, file: UploadFile = File(...), user: Optional[dict] = Depends(require_verified_if_logged_in)):
    return _run_prediction("xray", file, user)


@app.post("/predict/ct", response_model=PredictionResponse)
@limiter.limit("20/minute")
def predict_ct(request: Request, file: UploadFile = File(...), user: Optional[dict] = Depends(require_verified_if_logged_in)):
    return _run_prediction("ct", file, user)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
def chat(request: Request, req: ChatRequest, user: Optional[dict] = Depends(require_verified_if_logged_in)):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    reply, source = generate_chat_reply(req.message.strip(), req.history, req.prediction_context)
    return ChatResponse(reply=reply, source=source)


@app.get("/")
def root():
    return {
        "service": "Lung Disease Detection API",
        "status": "ok" if len(_bundles) == len(MODEL_NAMES) else "degraded",
        "docs": "/docs",
        "health": "/health",
        "note": "This is the backend API server, not the web app. "
                "The frontend runs separately -- see FRONTEND_URL in backend/.env "
                "(default http://localhost:5173).",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(_bundles.keys()),
        "google_oauth_configured": config.GOOGLE_OAUTH_CONFIGURED,
        "email_configured": config.EMAIL_CONFIGURED,
    }
