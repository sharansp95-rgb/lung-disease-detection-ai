# Project Understanding

> Read-only orientation pass, written 2026-09-10. Synthesized from direct code inspection plus the repo's own audit docs (`PROJECT_CONTEXT.md`, `PROJECT_STATUS.md`, `PROJECT_HANDOFF.md`), cross-checked against `backend/main.py`, `backend/db.py`, `backend/schemas.py`, `frontend/package.json`, and a live pytest run. Where this document and those docs might drift apart in the future, trust the code and the newest-dated doc (`PROJECT_CONTEXT.md`) over this one.

## 1. Project Overview

A **7th-semester student design project (SDP)**: a full-stack dual-modality lung-disease image classifier.

- Accepts one chest **X-ray** or **CT** image, runs it through a multi-stage guardrail + inference pipeline, and returns a predicted disease class, confidence, full class-probability breakdown, a Grad-CAM heatmap, and a plain-language report.
- X-ray classes: `COVID-19`, `Normal`, `Pneumonia`, `Tuberculosis`. CT classes: `Benign`, `Malignant`, `Normal`.
- Full user accounts (email/password + Google OAuth), persistent scan history, and an LLM-backed chat widget scoped to explaining the tool's own output.
- Explicitly framed as a **research/decision-support prototype, not a diagnostic tool** — this disclaimer is baked into the UI, generated reports, and the chatbot's system prompt. Intended for a course deliverable/demo audience, not clinical use, and runs entirely locally (no deployment target — see §8).

**Tech stack:**
- **Backend:** Python 3.12, FastAPI, Pydantic, PyTorch/Torchvision (+ `torchxrayvision` for the active X-ray backbone), stdlib `sqlite3` (no ORM), `pytest`+`httpx` for testing, `bcrypt`+`PyJWT` for auth, `slowapi` for rate limiting.
- **Frontend:** React 19 + React Router 7, Vite 8, hand-written CSS (no framework), `oxlint`. `@playwright/test` is installed but no test files exist yet.
- **ML tooling (offline, not served live):** `ml/` (architectures, training, eval) and `scripts/` (data prep, leakage/shortcut audits, checkpoint probes, figure generation) using numpy, PIL, OpenCV, scikit-image/learn, scipy, matplotlib, ImageHash.
- **External APIs:** Groq (LLM for reports/chat), Google OAuth 2.0, Gmail SMTP — all three are optional and degrade gracefully (template fallback, 501, or console-logged link, respectively) if unconfigured.
- **No git repository currently exists** — `.git/` was intentionally deleted 2026-09-10 during a storage cleanup (~12 GB of large historical blobs). A fresh `.gitignore` is in place for a future `git init`, but there is no commit history, branches, or diff right now. The working tree on disk is the sole source of truth.

## 2. Architecture

Layered modular monolith, not microservices:

```
Presentation   → React pages/components/context providers (frontend/src/)
API/App        → FastAPI routes + orchestration (backend/main.py, plus focused routers)
Domain/Infer   → model bundles, preprocessing, OOD/sanity/modality gates, segmentation, Grad-CAM, reports
Data           → direct SQLite repository-style functions (backend/db.py); no ORM, no migrations
Offline ML     → ml/ + scripts/ produce/inspect checkpoints, OOD stats, eval artifacts (not served live)
```

**Entry points:**
- Backend: `backend/main.py` exposes `app`; run with `uvicorn main:app --reload --port 8000` from `backend/`.
- Frontend: `frontend/src/main.jsx` renders `App.jsx`; `npm run dev` (Vite) from `frontend/`.
- Offline ML: `ml/train.py`, `ml/train_xray_txrv.py`, `ml/train_ct_monai.py`, `ml/evaluate.py` are standalone CLI scripts, not server entry points.

**Request flow (the only live serving path — `/predict/xray` and `/predict/ct`):**

```
Browser UploadArea → MainApp (nav state) → AnalyzingPage → api.predict() → POST /predict/{modality}
        ↓
backend/main.py: _run_prediction()
  1. Read + validate uploaded image (size cap, decodable, blank/color-content check)
  2. Per-modality sanity gate  (rejects non-scan images — photos, screenshots, documents)
  3. Modality gate             (rejects wrong-modality image, e.g. CT sent to /predict/xray)
  4. X-ray only: segmentation-model lung crop (falls back to uniform border-crop)
  5. Resize/normalize per model's own preprocessing
  6. OOD check                 (feature-space distance from training centroid)
  7. Classifier + Grad-CAM     (each model guarded by its own threading.Lock)
  8. Report generation         (Groq, falls back to local template)
  9. If authenticated: save_scan() to SQLite
  10. Return PredictionResponse (class, confidence, all probabilities, heatmap data-URI, report)
        ↓
ResultsPage / ResultsDisplay renders it
```

An experimental full-DICOM "CT volume" feature (`/predict/ct/volume`, a MONAI nodule detector, `CtVolumePage.jsx`) existed previously but has since been **removed entirely** — routes, pipeline, and frontend page are gone. Don't treat it as current.

## 3. Key Components

### Backend (`backend/`)
| File | Responsibility |
|---|---|
| `main.py` | App composition, six-model startup + reproduction checks, `_run_prediction` pipeline, `/predict/*`, `/chat`, `/health`, input guardrails |
| `auth_routes.py` | `/auth/*` endpoints, `get_current_user[_optional]` dependency |
| `scans_routes.py` | `/scans/history` + `save_scan()` |
| `models.py` | `ModelBundle` dataclass, `load_model_bundle()` + per-checkpoint reproduction check |
| `seg_utils.py` | `SegBundle`, `load_seg_bundle()`, `compute_lung_crop()` (U-Net-based X-ray cropping) |
| `ood_detector.py` | `OODDetector` — feature-space distance rejection against a training centroid |
| `gradcam_utils.py` | Wraps `ml/gradcam.py`; returns base64 heatmap overlay; has both ImageNet-style and torchxrayvision-style preprocessing/rendering paths |
| `report.py`, `chat.py`, `groq_client.py` | LLM-backed report/chat generation with template fallback |
| `auth_utils.py`, `email_utils.py` | Password hashing/JWT helpers, verification/reset token emailing (SMTP or console fallback) |
| `config.py` | `.env` loading, thresholds, feature flags |
| `db.py`, `schemas.py` | SQLite schema/access; Pydantic API response models |
| `tests/` | 26 integration-style pytest cases against real local checkpoints/data, isolated SQLite test DB |

Startup loads **five checkpoint-backed model bundles** (`xray`, `ct`, `gate`, `xray_sanity`, `ct_sanity`) plus one segmentation model, and each undergoes a **reproduction check**: its checkpoint's claimed validation metric is independently recomputed against live validation data on every boot — a mismatch beyond tolerance raises and blocks startup. This is the project's strongest engineering property: it can't silently serve a wrong/stale model.

### Frontend (`frontend/src/`)
| Path | Responsibility |
|---|---|
| `pages/` | `LandingPage`, `LoginPage`/`SignupPage`/`ForgotPasswordPage`/`ResetPasswordPage`/`OAuthCallbackPage`, `MainApp` (upload), `AnalyzingPage`, `ResultsPage`, `ScanHistoryPage`, `ProfilePage`, `EmailVerifiedPage` |
| `components/` | `AppNavbar` (nav + theme toggle + avatar), `ChatWidget` (floating chat, FAQ chips), `ModalitySelector`, `UploadArea`, `ResultsDisplay`, `ThemeToggle` |
| `AuthContext.jsx` | Token/user state, localStorage-persisted, calls `/auth/me` |
| `ThemeContext.jsx` | Dark/light theme, localStorage-persisted |
| `api.js` | Central fetch wrapper / API base URL, prediction/chat/history calls |

### Routes (backend)
`/`, `/health`, `/predict/xray`, `/predict/ct`, `/chat`, `/auth/signup`, `/auth/login`, `/auth/me`, `/auth/change-password` (no frontend UI calls it — intentional, `forgot-password` covers the need), `/auth/verify/{token}`, `/auth/forgot-password`, `/auth/reset-password/{token}`, `/auth/google/login`, `/auth/google/callback`, `/scans/history`.

### Routes (frontend)
`/` (landing or upload depending on auth), `/login`, `/signup`, `/forgot-password`, `/reset-password/:token`, `/oauth-callback` (public); `/analyzing`, `/results`, `/history`, `/profile` (protected via `RequireAuth`).

## 4. Data Model

SQLite (`backend/app.db`), plain `sqlite3` (no ORM), schema inline in `backend/db.py`:

- **`users`** — id, unique email, nullable `password_hash` (null for Google-only accounts), `email_verified` flag, unique `google_id`, `display_name`, `created_at`.
- **`tokens`** — one-time verify/reset tokens: `user_id` (FK), `token`, `type` (`verify`|`reset`), `expires_at`, `used`, `created_at`.
- **`scans`** — one saved prediction per authenticated request: `user_id` (FK), `modality`, `predicted_class`, `confidence`, `class_probabilities` (JSON text), `heatmap_png_base64` (full data URI), `report`, `report_source`, `created_at`.

Relationships: `users 1:N tokens`, `users 1:N scans`. Foreign keys enabled per-connection (`PRAGMA foreign_keys = ON`) but no cascade deletes and no migration framework — schema changes are additive edits to the inline `CREATE TABLE IF NOT EXISTS` script. Anonymous predictions are never persisted.

**Model artifacts** (not a DB, but structurally similar): checkpoints in `checkpoints/*.pt` bundle model state + class names + backbone identifier + the metric used for the startup reproduction check. `backend/ood_stats/*.pt` store per-modality feature centroid + distance threshold for the OOD detector.

## 5. External Dependencies & Integrations

All three are **optional** and degrade gracefully rather than hard-failing:

- **Groq** (`api.groq.com/openai/v1/chat/completions`) — used in `backend/groq_client.py` for report generation (`report.py`) and chat (`chat.py`). Model is `openai/gpt-oss-120b` (the originally spec'd `llama-3.3-70b-versatile` no longer exists on Groq's catalog). Falls back to a local template if `GROQ_API_KEY` is unset or the call fails.
- **Google OAuth 2.0** — "Continue with Google" login (`auth_routes.py`). Returns a clear 501 if `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` aren't set. Confirmed working end-to-end with a real Google account in prior testing.
- **Gmail SMTP** (`smtp.gmail.com:587`) — verification/reset emails (`email_utils.py`). Falls back to printing the link to the backend console/log if `GMAIL_ADDRESS`/`GMAIL_APP_PASSWORD` aren't set.
- **torchxrayvision** — pretrained DenseNet-121 backbone (`densenet121-res224-all`) used as the active X-ray classifier's transfer-learning base. **Not listed in `requirements.txt`** despite being a live runtime dependency — a clean `pip install -r requirements.txt` will not reproduce the active X-ray path without also installing it manually.
- **pydicom, monai, nibabel** — leftover from the now-removed CT-volume feature; no longer imported by the live backend. `monai` is still used by the offline research script `ml/train_ct_monai.py` (misleadingly named — it actually uses a RadImageNet DenseNet implementation, not a MONAI pretrained classifier).

No hosted database, CI, Docker, or deployment target exists — this is a deliberate scope decision, not a gap.

## 6. Conventions & Patterns

- **Reproduce-before-serve:** every model checkpoint's claimed metric is re-verified against live validation data at startup; a mismatch blocks boot. Applied consistently across all six model bundles.
- **Layered guardrail pipeline ahead of classification:** blank/color check → sanity gate (real scan vs. not) → modality gate (right scanner type) → OOD detector (alien feature space) → classifier. Each layer was added because a specific failure mode was found and tested (documented in `PROJECT_STATUS.md` §3), not speculative hardening.
- **Everything degrades, nothing hard-fails on optional integrations:** Groq, Google OAuth, and Gmail SMTP all have an explicit fallback path and are exercised by tests in both configured and unconfigured states.
- **No ORM, no framework CSS, minimal abstraction:** SQLite accessed via raw `sqlite3` + a `get_db()` context manager; frontend styling is hand-written CSS with custom properties for theming rather than Tailwind/MUI/etc. Consistent with the project's small, stable scope.
- **Concurrency via per-model locks:** each `ModelBundle` holds its own `threading.Lock`, held across preprocess+forward+Grad-CAM — fixes a real found bug where overlapping requests corrupted Grad-CAM forward hooks.
- **Explicit non-diagnostic framing** repeated at every layer: UI copy, generated reports, and the chatbot's system prompt all carry the same disclaimer.
- **Audit-driven documentation:** this project keeps unusually detailed, dated, self-correcting status docs (`PROJECT_STATUS.md`, `PROJECT_HANDOFF.md`, `PROJECT_CONTEXT.md`, `VERIFICATION_REPORT*.md`, `GENERALIZATION_DIAGNOSIS.md`) that disclose negative results (e.g. a rejected RadImageNet CT backbone swap, a TB accuracy regression) rather than hiding them. These docs can drift out of sync with each other — prefer the newest-dated one and the code itself.

## 7. Current State

**Tests:** ran `python -m pytest tests -v` from `backend/` — **26/26 passed in 70.48s** (auth flows, health, and real-checkpoint prediction/guardrail cases; no mocked inference). Three deprecation warnings only (`@app.on_event("startup")` → should migrate to FastAPI lifespan handlers; `httpx`/`starlette.testclient` deprecation notice) — non-blocking.

**Known open issues (per the repo's own audit docs — not independently re-verified in this pass):**
- 🔴 **Unresolved:** `VERIFICATION_REPORT_2.md` reports the live `/predict/xray` path over-predicting COVID-19 in practice, estimating ~78–84% served accuracy vs. the documented offline 96.57% test accuracy — suspected crop/serving-path discrepancy, not reproduced/root-caused as of the last audit. The passing pytest suite does not have per-class live-path regression coverage that would catch this. **Don't treat current X-ray predictions as trustworthy until this is reconciled.**
- `GENERALIZATION_DIAGNOSIS.md` documents sharp accuracy degradation under preprocessing perturbations (X-ray inversion, CT histogram equalization) and X-ray Grad-CAM heatmaps concentrated on image borders/corners in some failure cases — a shortcut-learning sensitivity, not a crash bug.
- OOD/sanity/modality gates catch most alien or wrong-modality input but not an in-scope-but-untrained pathology (e.g. a genuine X-ray showing Lung Opacity, which isn't one of the 4 trained classes).
- `requirements.txt` is missing `torchxrayvision` (see §5).
- JWTs remain valid after a password change/reset — stateless tokens, no revocation/session-versioning.
- `frontend/README.md` is still the stock Vite template, not project documentation.
- No git history currently exists (see §1) — treat the working tree as the only source of truth until/unless `git init` happens.

**Deliberately out of scope (confirmed project decisions — not gaps):** no deployment/CI/Docker, in-memory (not Redis) rate limiting, no change-password frontend UI (backend endpoint works, `forgot-password` covers the need), small CT dataset (1,036 images) not being grown further.

## 8. Setup

```powershell
# From repository root — Python deps (CPU PyTorch by default; see requirements.txt
# comment for CUDA wheel install instructions if a GPU is available)
python -m pip install -r requirements.txt
python -m pip install torchxrayvision   # not in requirements.txt but required — see §5

# Terminal 1 — backend (needs local checkpoints/ + data/ + seg_data/ for startup
# reproduction checks; these are present on disk but gitignored)
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev

# Backend test suite (from backend/)
python -m pytest tests -v

# Production frontend build (from frontend/)
npm run build
```

**Config:** `backend/config.py` loads `backend/.env` (gitignored, not present in a fresh checkout — must be created). Relevant variables: `GROQ_API_KEY`, `JWT_SECRET` (auto-generated and appended to `.env` if absent), `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`, `GMAIL_ADDRESS`/`GMAIL_APP_PASSWORD`, `FRONTEND_URL` (default `http://localhost:5173`), `BACKEND_URL` (default `http://127.0.0.1:8000`), `DB_PATH` (default `backend/app.db`), `MODALITY_GATE_CONFIDENCE_THRESHOLD` (default `0.90`). Frontend: `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`).

CORS is hardcoded to `localhost:5173`/`127.0.0.1:5173` only — changing the frontend host requires a code change, not just config.

**Missing for a truly clean checkout:** `torchxrayvision` isn't declared as a dependency; large runtime assets (`checkpoints/*.pt`, `data/`, `seg_data/`) are gitignored/excluded and must be present on disk already (they are, on this machine) — there's no documented acquisition/download step for them elsewhere in the repo.
