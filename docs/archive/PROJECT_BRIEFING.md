# Project Briefing: Lung Disease Detection (Chest X-ray / CT)

Snapshot: **2026-09-15, evening**. Repo root: `D:\7th sem\sdp` (Windows 11, Python 3.12.10, Node 24.19.0). All paths are relative to that root.

**How this was produced.** I read every backend module and test, all of `frontend/src`, `frontend/vite.config.js`, the core `ml/` files, `start_public.ps1`, `DEPLOYMENT_PUBLIC.md`, `backend/README_AUTH.md`, the demo scripts and the `test/` harness README. I skimmed the headers of the other `ml/` and `scripts/` files. I loaded checkpoint metadata (not weights), counted images on disk, ran the backend test suite (**59 passed, 57.6 s**), and ran `oxlint` (4 warnings). Where a claim comes from a doc in the repo rather than from code, I say so. Secrets are redacted. I list env variable *names* only.

---

## 1. Project Overview

- **What it is:** a 7th-semester student design project (SDP). It is a full-stack web app where a user uploads **one chest image**, either a chest X-ray or a single 2D CT slice, and gets back:
  - a predicted class and per-class probabilities;
  - a Grad-CAM heatmap showing which image regions drove the prediction;
  - a short plain-language report, written by an LLM (Groq) with a template fallback;
  - an optional chat assistant that is restricted to explaining the tool's own output.
- **Classes:**
  - X-ray: `COVID-19`, `Normal`, `Pneumonia`, `Tuberculosis`.
  - CT: `Benign`, `Malignant`, `Normal`.
- **Target users:** evaluators and demo audiences for the SDP, plus an accompanying IEEE-style paper (`documents/`). The app says clearly, in the UI, the reports and the chat system prompt, that it is a **research / decision-support prototype, not a diagnostic tool**.
- **Problem it addresses:** automated screening support with explainability. The engineering emphasis is on **not giving confident wrong answers**. The pipeline puts several guardrails in front of the classifier (blank-image check, color-content check, scan-vs-not-scan "sanity" gate, X-ray-vs-CT modality gate, feature-space OOD detector). At startup, every model is re-checked against its claimed validation accuracy, and the app refuses to boot if any check fails.
- **Deployment model (deliberate):** everything runs on the author's laptop. Since 2026-09-15 it can be made publicly reachable through an **ngrok** free dev domain (`start_public.ps1`). There is no cloud hosting, Docker or CI by design.

---

## 2. Tech Stack

### Backend (`backend/`, pinned in `requirements.txt`)
| Area | Library / version |
|---|---|
| Web framework | FastAPI 0.141.1, uvicorn 0.52.4, python-multipart 0.0.32, Pydantic 2.13.4 |
| Auth | bcrypt 5.0.0 (passwords), PyJWT 2.13.0 (HS256 sessions) |
| Rate limiting | slowapi 0.1.10 (in-memory, per client IP) |
| Config | python-dotenv 1.2.3 |
| HTTP client | requests 2.34.2 (Groq, Google OAuth) |
| ML | torch 2.11.0 (+cu128 installed locally), torchvision 0.26.0, **torchxrayvision 1.5.4 (installed, but NOT in `requirements.txt`, see §13)**, numpy 2.5.2, pillow 12.3.0, opencv-python-headless 5.0.0.93, scikit-image 0.26.0, scikit-learn 1.9.0, scipy 1.18.1, matplotlib 3.11.1, ImageHash 4.3.2 |
| Research-only | monai 1.6.0 (installed, not in requirements; used only by `ml/train_ct_monai.py`) |
| Tests | pytest 9.1.1, httpx 0.28.1 (FastAPI TestClient) |
| Database | SQLite through stdlib `sqlite3` (no ORM, no migrations) |

### Frontend (`frontend/package.json`)
- React ^19.2.8, react-dom ^19.2.8, react-router-dom ^7.18.2.
- Vite ^8.2.2 with @vitejs/plugin-react ^6.1.0.
- oxlint ^1.79.0 (the linter; `.oxlintrc.json`).
- @playwright/test ^1.62.1 (dev dependency, used only by stale ad-hoc `.cjs` scripts).
- Hand-written CSS (`src/App.css`, ~31 KB), with a dark/light theme.

### External services
- **Groq** chat completions API. Model `openai/gpt-oss-120b` (`backend/groq_client.py`). Used for reports and chat.
- **Google OAuth 2.0** ("Continue with Google").
- **Gmail SMTP** (smtp.gmail.com:587 with an app password) for verification and reset emails. If it isn't configured, the email content is printed to the console instead.
- **ngrok** agent (external binary) for public access.

### Tooling
- PowerShell launcher `start_public.ps1`.
- Model training happened on Colab or locally (RTX 3050 4 GB GPU, per `ml/train_xray_txrv.py` docstring).
- Jupyter notebook `scripts/train_sanity_gate.ipynb`.

---

## 3. Directory Structure

Only the important parts are shown. Generated or bulky folders are marked.

```
sdp/
├── backend/                       FastAPI app (~2,300 lines of Python)
│   ├── main.py                    App, startup model loading, /predict/*, /chat, /, /health
│   ├── config.py                  Loads backend/.env; all config + URL/origin allowlists
│   ├── public_origin.py           NEW (today): per-request origin resolution + signed OAuth state
│   ├── auth_routes.py             /auth/* (signup, login, verify, reset, change-pw, Google OAuth)
│   ├── auth_utils.py              bcrypt hashing, JWT create/decode, random tokens
│   ├── scans_routes.py            /scans/history + save_scan()
│   ├── db.py                      SQLite schema (users, tokens, scans) + get_db() context manager
│   ├── models.py                  ModelBundle, checkpoint loading + startup reproduction check
│   ├── seg_utils.py               Lung U-Net loading + compute_lung_crop() for X-rays
│   ├── ood_detector.py            Centroid-distance OOD detector
│   ├── gradcam_utils.py           Preprocessing + Grad-CAM heatmap (ImageNet path and XRV path)
│   ├── report.py / chat.py        Groq report + chatbot, both with template fallback
│   ├── groq_client.py             Single Groq HTTP call
│   ├── email_utils.py             Gmail SMTP sender + console fallback
│   ├── rate_limit.py              slowapi Limiter; trusts X-Forwarded-For only from 127.0.0.1
│   ├── schemas.py                 Pydantic request/response models + chat payload caps
│   ├── clear_db.py                One-off script: wipes all rows in app.db
│   ├── ood_stats/{xray,ct}_ood.pt Precomputed OOD centroids + thresholds
│   ├── tests/                     pytest suite (6 files, 59 tests)
│   ├── README_AUTH.md             Google OAuth / Gmail setup guide
│   ├── .env                       Secrets + URLs (gitignored; contents not reproduced)
│   ├── app.db                     Live SQLite DB (2 users, 2 tokens, 0 scans at snapshot)
│   └── dev_backend*.log, *_fp_test_list.txt, groq_report_text.txt   (leftover clutter)
├── frontend/
│   ├── src/
│   │   ├── main.jsx, App.jsx      Router + providers
│   │   ├── api.js                 fetch wrappers, API base URL, class/label constants
│   │   ├── AuthContext.jsx        Token in localStorage, /auth/me, login/signup/reset helpers
│   │   ├── ThemeContext.jsx       Dark/light theme in localStorage
│   │   ├── pages/                 12 route pages (see §7)
│   │   └── components/            AppNavbar, ChatWidget, ModalitySelector, ResultsDisplay,
│   │                              ThemeToggle, UploadArea
│   ├── vite.config.js             /api proxy + allowedHosts for the ngrok setup
│   ├── .env.public.example        Template for public-mode env (.env.public.local is gitignored)
│   ├── test_*.cjs, capture_report_screenshots.cjs   Ad-hoc Playwright scripts (stale)
│   ├── *.png                      ~20 old screenshots (clutter)
│   ├── README.md                  Unmodified Vite template README
│   └── dist/, node_modules/       (generated)
├── ml/                            Training / eval code; also imported by the backend at runtime
│   ├── model.py                   ImageClassifier (torchvision densenet121 / resnet18) + forward_features
│   ├── dataset.py                 ImageFolder datasets, ImageNet transforms, class weights
│   ├── gradcam.py                 GradCAM class, overlay helpers
│   ├── train.py                   Warmup-then-finetune training loop (ImageNet backbones)
│   ├── train_xray_txrv.py         torchxrayvision DenseNet121 training + XRVClassifier + XRV transforms
│   ├── train_ct_monai.py          RadImageNet DenseNet121 CT training (research; not served)
│   ├── seg_model.py               U-Net architecture (reverse-engineered from the checkpoint)
│   ├── evaluate.py                Test-split metrics + Grad-CAM grids (ImageNet backbones only)
│   └── smoke_test.py              Synthetic-data pipeline smoke test
├── scripts/                       ~40 one-off data prep / audit / figure scripts (unorganized)
│   ├── prepare_xray.py            Builds data/xray (also imported by backend/seg_utils.py!)
│   ├── compute_ood_thresholds.py  Builds backend/ood_stats/*.pt
│   ├── predict_cli.py             Terminal demo of the pipeline (STALE, see §13)
│   └── ct_leakage_audit.py, tb_artifact_detect.py, ...   audits used in the demo
├── checkpoints/                   13 .pt files (~430 MB); 6 are served, the rest are experiments
├── data/                          Model-ready splits (gitignored)
├── seg_data/                      Lung-segmentation images + masks (train/val/test)
├── datasets/, raw/                Raw Kaggle sources (gitignored, large)
├── eval_run/                      Investigation phase notes + scripts (CT shortcut work, crop fix, QA runs)
├── eval_outputs/, ml_logs/, audit/, diagnosis_outputs/   Metrics JSON, curves, audit grids
├── test/                          External-generalization diagnostic harness (corruption / provenance / external)
├── demo/terminal{1,2,3}_commands.txt   Scripted live-demo commands (partly stale)
├── documents/                     Review-1 PPT, IEEE paper PDF/zip
├── logs/                          modality_gate_debug.log (1.2 MB, grows), public_* run logs
├── start_public.ps1               One-command public (ngrok) launcher
├── DEPLOYMENT_PUBLIC.md           Public-access guide (updated today)
├── requirements.txt, .gitignore
└── 10 overlapping top-level .md reports: PROJECT_STATUS, PROJECT_BRIEF, PROJECT_CONTEXT,
    PROJECT_HANDOFF, PROJECT_UNDERSTANDING, VERIFICATION_REPORT(_2), GENERALIZATION_DIAGNOSIS,
    CLASSIFIER_EVALUATION_REPORT, and this file. Each is a snapshot from a different date.
```

---

## 4. Architecture

**Pattern:** a client-server modular monolith.
- A React SPA talks to a single FastAPI process over JSON and multipart HTTP.
- All ML inference runs **in-process** in the API, on CUDA if available, otherwise CPU.
- SQLite holds persistence.
- There are no background workers or queues. Every request is synchronous. FastAPI runs the sync handlers in its threadpool, and each model is guarded by a `threading.Lock`.
- The backend imports code from `ml/` and `scripts/` by pushing those folders onto `sys.path` (`backend/models.py:24`, `backend/seg_utils.py:34-35`). So `ml/` and `scripts/prepare_xray.py` are **runtime dependencies**, not just offline tooling.

### Two run modes

**Local dev:** `npm run dev` on :5173 calls `http://127.0.0.1:8000` directly, which is cross-origin, so CORS applies.

**Public (ngrok):**
```
Browser (phone, anywhere)
   │ https://<ngrok-domain>
   ▼
ngrok cloud ──tunnel──► ngrok agent (laptop)
                           │ http://127.0.0.1:4173
                           ▼
                  vite preview (built dist/)
                    ├─ /        → static SPA
                    └─ /api/*   → strip /api, add X-Forwarded-* → http://127.0.0.1:8000
                                                         ▼
                                      FastAPI (bound to 127.0.0.1 only)
```
In public mode the browser sees one origin (`VITE_API_BASE_URL=/api`). The backend picks the real client IP out of `X-Forwarded-For` for rate limiting (`backend/rate_limit.py`). It also works out the public origin for each request (`backend/public_origin.py`), so email links and Google sign-in go back to the site the user is actually on. Localhost and the ngrok domain can be used at the same time.

### Prediction data flow (`backend/main.py::_run_prediction`, lines 307–452)
```
upload ─► _read_and_validate_image
            size ≤10MB (413) · empty (400) · PIL verify (400)
            grayscale std <10 → 400 "blank"
            mean RGB channel spread >20 → 400 "color content"
      ─► preprocess_image (grayscale→3ch, 224², ImageNet norm)
      ─► sanity gate  (resnet18 scan / not_scan, per modality) → 422 if not_scan
      ─► modality gate (resnet18 xray / ct)
            top conf < 0.90 → 422 "Modality uncertain"
            wrong modality  → 422 "use the X-ray/CT tab"
      ─► [X-ray only] compute_lung_crop (U-Net mask bbox → squash to TARGET_SIZE)
                      → preprocess_image_xrv (1ch, [-1024,1024])
      ─► with bundle.lock:
            OOD: ‖forward_features − centroid‖ > p99 threshold → 422
            classify + Grad-CAM → heatmap PNG data URI
      ─► warning if confidence < MIN_CONFIDENCE+0.15 (X-ray 0.50, CT 0.55)
      ─► generate_report (Groq → template fallback)
      ─► if logged in: save_scan → SQLite
      ─► PredictionResponse JSON
```

### Served models (verified from checkpoint metadata + startup log)
| Bundle | Checkpoint | Architecture | Classes | Val acc (claimed = reproduced) |
|---|---|---|---|---|
| `xray` | `xray_txrv_densenet121_best.pt` | torchxrayvision DenseNet121 (`densenet121-res224-all`), 1-ch input | 4 | 0.9646 |
| `ct` | `ct_best_model_phase35_v2.pt` | torchvision DenseNet121 (ImageNet) | 3 | 0.9904 |
| `gate` | `modality_gate_best_model_v2.pt` | resnet18 | `xray`, `ct` | 1.0000 |
| `xray_sanity` | `xray_sanity_best.pt` | resnet18 | `not_scan`, `scan` | 0.9996 |
| `ct_sanity` | `ct_sanity_best.pt` | resnet18 | `not_scan`, `scan` | 1.0000 |
| seg | `xray_seg_model.pt` | U-Net (base 32) | lung mask | dice 0.989 claimed / 0.983 reproduced (tolerance 0.01) |

OOD thresholds (`backend/ood_stats/`): X-ray **8.72** (p99 over 15,373 train images, recomputed 2026-09-06 for the txrv model), CT **25.62** (over 829 images).

Test-set accuracy from `CLASSIFIER_EVALUATION_REPORT.md` (2026-09-10; this is a doc, I did not re-run it):
- **X-ray:** 96.57% offline, **91.05% through the live API**. The live figure counts 13 rejections as errors, so the serving path loses about 5.5 points.
- **CT:** 98.06% both offline and live, on 103 images.
- No external-data evaluation exists: `test/external_probe/` contains only placeholders.

Unused checkpoints kept on disk:
- `xray_best_model_phase35_v2.pt` (old ImageNet X-ray model) and `modality_gate_best_model.pt` (v1 gate).
- `ct_best_model_phase36*`, `ct_*phase37*` (CT retraining experiments).
- `ct_lidc_nodulecrop_v4_stack3.pt` (from the removed CT-volume experiment).

---

## 5. Data Model

### SQLite (`backend/db.py`, schema created by `init_db()` at startup; no migrations)

**users**
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK AUTOINCREMENT | |
| email | TEXT UNIQUE NOT NULL | lower-cased on signup |
| password_hash | TEXT NULL | bcrypt; NULL for Google-only accounts |
| email_verified | INTEGER (0/1) | informational only, gates nothing |
| google_id | TEXT UNIQUE NULL | |
| display_name | TEXT NULL | |
| created_at | TEXT | `datetime('now')`, UTC without a `Z` (the frontend appends `Z`) |

**tokens** (email verification + password reset)
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| user_id | FK → users.id | |
| token | TEXT UNIQUE | `secrets.token_urlsafe(32)` |
| type | TEXT CHECK in (`verify`, `reset`) | |
| expires_at | TEXT ISO-8601 (tz-aware) | verify 24 h, reset 1 h |
| used | INTEGER 0/1 | |
| created_at | TEXT | |

**scans** (history, saved only for logged-in users)
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| user_id | FK → users.id | no index on this column |
| modality | TEXT | `xray` / `ct` |
| predicted_class | TEXT | |
| confidence | REAL | |
| class_probabilities | TEXT | JSON string |
| heatmap_png_base64 | TEXT NOT NULL | full data-URI PNG stored inline |
| report, report_source | TEXT | |
| created_at | TEXT | |

**Relationships:** users 1—N tokens, users 1—N scans. The original uploaded image is **not** stored, only the heatmap overlay. Old tokens are never cleaned up.

### Image datasets (file counts on disk, train / val / test)
| Folder | Train | Val | Test | Used by |
|---|---|---|---|---|
| `data/xray` (pre-cropped lung fields, 4 classes) | 15,373 | 1,922 | 1,922 | xray model + startup check |
| `data/ct_scan` (IQ-OTH/NCCD, 3 classes, grouped resplit) | 829 | 104 | 103 | ct model + startup check |
| `data/modality_gate` | 16,202 | 2,026 | 2,025 | gate |
| `data/xray_sanity` | 22,833 | 2,596 | 2,604 | xray_sanity |
| `data/ct_sanity` | 3,829 | 778 | 785 | ct_sanity |
| `seg_data/{train,val,test}/{images,masks}` | (count not verified) | | | seg model + startup check |

Raw sources: COVID-19 Radiography DB, TB Chest Radiography DB, IQ-OTH/NCCD lung cancer CT (`datasets/`, `data/README.md`). Data prep history is documented in `data/README.md`: TB artifact cleaning, CT near-duplicate leakage audit and grouped resplit, scanner-table shortcut exclusion.

---

## 6. Features & Modules

| Feature | Where | Status |
|---|---|---|
| X-ray classification + Grad-CAM | `main.py`, `gradcam_utils.py` (XRV path), `models.py`, `ml/train_xray_txrv.py` | **Complete** |
| CT slice classification + Grad-CAM | same files, ImageNet path | **Complete** (known shortcut-learning concern, see §13) |
| Blank / color-content upload checks | `main.py::_read_and_validate_image` | **Complete** |
| Sanity gates (scan vs not-scan) | `main.py::_sanity_predict`, checkpoints `*_sanity_best.pt` | **Complete** |
| Modality gate (X-ray vs CT) with confidence floor | `main.py::_gate_predict`, `config.MODALITY_GATE_CONFIDENCE_THRESHOLD` (0.90) | **Complete** |
| Lung-segmentation crop for X-rays | `seg_utils.py::compute_lung_crop` (reuses `scripts/prepare_xray.py`) | **Complete** (crop fix from 2026-09-08 in place) |
| OOD detection | `ood_detector.py`, `scripts/compute_ood_thresholds.py` | **Complete** |
| Startup reproduction checks (refuse to boot on mismatch) | `models.py::load_model_bundle`, `seg_utils.py::load_seg_bundle` | **Complete** (adds ~1–2 min to startup) |
| LLM report with template fallback | `report.py`, `groq_client.py` | **Complete** |
| Scoped chatbot | `chat.py`, `frontend/src/components/ChatWidget.jsx` | **Complete** (length caps and 20/min limit added today) |
| Email/password auth (JWT, 7 days) | `auth_routes.py`, `auth_utils.py`, `AuthContext.jsx` | **Complete** |
| Email verification | `auth_routes.py::verify_email`, `EmailVerifiedPage.jsx` | **Complete but informational only**: the flag doesn't gate any feature (deliberate) |
| Forgot / reset password | `auth_routes.py`, `ForgotPasswordPage.jsx`, `ResetPasswordPage.jsx` | **Complete** |
| Change password | `POST /auth/change-password` | **Backend only**. There is deliberately no UI, because forgot-password covers the need |
| Google OAuth, multi-origin, signed state + nonce cookie | `auth_routes.py::google_login/google_callback`, `public_origin.py` | **Complete**. The rewrite landed today, and the logs show successful callbacks from both the ngrok and localhost:4173 origins |
| Scan history | `scans_routes.py`, shown inside `ProfilePage.jsx` | **Complete** (no pagination) |
| Standalone `/history` page | `ScanHistoryPage.jsx` | **Orphaned**: routed but not linked from anywhere, and duplicates ProfilePage |
| Dark/light theme | `ThemeContext.jsx`, `ThemeToggle.jsx` | **Complete** |
| Public access through ngrok | `start_public.ps1`, `vite.config.js`, `rate_limit.py`, `public_origin.py` | **Complete** (new today) |
| CLI demo pipeline | `scripts/predict_cli.py` | **Broken/stale** (§13) |
| External generalization harness | `test/scripts/*.py` | **Partial**: corruption test run, external probe never filled |
| CT-volume (DICOM series) pipeline | formerly `/predict/ct/volume` | **Removed** after 28.6% external accuracy (`eval_run/phaseJ_external_validation.md`). No code remains, only notes and a checkpoint |

---

## 7. API / Routes / Screens

### Backend endpoints
| Method | Path | Purpose | Auth | Rate limit |
|---|---|---|---|---|
| POST | `/predict/xray` | multipart `file` → `PredictionResponse` | optional Bearer (saves history if present) | 20/min |
| POST | `/predict/ct` | same, CT | optional Bearer | 20/min |
| POST | `/chat` | `{message ≤2000, history ≤50 items (each ≤4000 chars), prediction_context?}` → `{reply, source}` | none | 20/min |
| GET | `/` | service info / status | none | – |
| GET | `/health` | `{status, models_loaded, google_oauth_configured, email_configured}` | none | – |
| POST | `/auth/signup` | `{email, password ≥8, name?}` → `{access_token, user}` + sends verification email | none | 30/hour |
| POST | `/auth/login` | `{email, password}` → `{access_token, user}` | none | 20/min |
| GET | `/auth/me` | current `UserOut` | Bearer required | – |
| POST | `/auth/change-password` | `{current_password, new_password}` | Bearer required | – |
| GET | `/auth/verify/{token}?origin=` | marks verified, 307 → `{frontend}/email-verified` | token | – |
| POST | `/auth/forgot-password` | always returns a generic message; emails a reset link if the account exists | none | 5/hour |
| POST | `/auth/reset-password/{token}` | `{new_password}`; rejects reuse of the current password | token | – |
| GET | `/auth/google/login` | 307 → Google consent; sets `google_oauth_nonce` cookie | none | – |
| GET | `/auth/google/callback` | exchanges the code, creates or links the user, 307 → `{frontend}/oauth-callback?token=<jwt>` or `/login?error=...` | state + nonce | – |
| GET | `/scans/history` | all of the user's scans, newest first, heatmaps inline | Bearer required | – |
| GET | `/docs`, `/openapi.json` | FastAPI defaults (also reachable publicly via `/api/docs`) | none | – |

Error format: FastAPI `{"detail": "..."}`. Pydantic validation errors return 422 with a `detail` array. Unhandled exceptions return a sanitized 500 (`main.py::_sanitized_500`).

### Frontend routes (`frontend/src/App.jsx`)
| Path | Page | Guard | Notes |
|---|---|---|---|
| `/` | `LandingPage` (logged out) / `MainApp` (logged in) | – | MainApp: modality selector, upload, "Analyze" button, chat widget |
| `/login` | `LoginPage` | – | email/password + "Continue with Google" link |
| `/signup` | `SignupPage` | – | afterwards shows "Check your email" and auto-advances when another tab verifies |
| `/forgot-password` | `ForgotPasswordPage` | – | |
| `/reset-password/:token` | `ResetPasswordPage` | – | |
| `/oauth-callback` | `OAuthCallbackPage` | – | reads `?token=`, stores it, navigates to `/` |
| `/email-verified` | `EmailVerifiedPage` | – | writes a localStorage key to notify other tabs |
| `/analyzing` | `AnalyzingPage` | RequireAuth | receives `{file, modality, previewUrl}` via router state; calls `/predict`; simulated step animation; AbortController guards against StrictMode double-submit |
| `/results` | `ResultsPage` | RequireAuth | `ResultsDisplay` (class, probability bars, heatmap/original toggle, report) + chat with prediction context |
| `/profile` | `ProfilePage` | RequireAuth | account card, logout, scan-history grid + modal |
| `/history` | `ScanHistoryPage` | RequireAuth | orphaned duplicate |

Navigation: `AppNavbar` (theme toggle + avatar → `/profile`) appears only when logged in. **In the UI, prediction requires login.** The backend allows anonymous prediction, but the SPA only shows the upload screen to logged-in users.

---

## 8. Key Workflows

### 8.1 Upload → prediction (logged-in user)
1. `MainApp.jsx`: the user picks a modality in `ModalitySelector` and a file in `UploadArea`. The client rejects non-image MIME types and files over 10 MB. `handleSubmit` navigates to `/analyzing` with `{file, modality, previewUrl}` in router state.
2. `AnalyzingPage.jsx`: the effect calls `api.js::predict(modality, file, token, signal)`, which does `POST {API_BASE}/predict/{modality}` as multipart with the Bearer token.
3. `main.py::predict_xray`/`predict_ct`: the slowapi limit applies, `get_current_user_optional` decodes the JWT (an invalid token just means anonymous), then `_run_prediction`:
   - validates the image (§4 diagram);
   - runs the sanity gate, then the modality gate (logging to `logs/modality_gate_debug.log`);
   - for X-rays, applies the lung crop and XRV preprocessing;
   - under the model lock, runs the OOD check, then `predict_with_heatmap(_xrv)`;
   - adds low-confidence warnings, calls `report.generate_report`, calls `scans_routes.save_scan` if logged in, and returns.
4. On success the frontend `navigate("/results", {replace:true, state:{result, previewUrl}})`, and `ResultsPage` renders `ResultsDisplay` + `ChatWidget`. On error, `AnalyzingPage` shows the backend's `detail` string with Retry / Back buttons.

### 8.2 Sign up + email verification
1. `SignupPage.jsx` → `AuthContext.signup` → `POST /auth/signup`.
2. `auth_routes.signup`:
   - checks for a duplicate email (409);
   - inserts the user with a bcrypt hash;
   - inserts a `verify` token (24 h);
   - `public_origin.resolve(request)` picks the frontend and backend bases from the Origin / Referer / X-Forwarded headers, checked against `ALLOWED_PUBLIC_ORIGINS`;
   - builds the link `{backend}/auth/verify/{token}` (plus `?origin=` when the origin isn't the default);
   - `email_utils.send_verification_email` sends it (or prints it to the console);
   - returns a JWT immediately, since the user is logged in right away.
3. The user clicks the link → `verify_email` marks the user verified and the token used, re-checks `origin` against the allowlist, and redirects to `/email-verified`.
4. `EmailVerifiedPage` writes `email_verified_broadcast` to localStorage. The original tab's `AuthContext` storage listener re-fetches `/auth/me`, and `SignupPage` auto-navigates to `/`.

### 8.3 Google OAuth (multi-origin)
1. `LoginPage` has a plain `<a href={API_BASE}/auth/google/login>`.
2. `google_login` does the following:
   - returns 501 if OAuth isn't configured;
   - `resolve(request)` → `PublicOrigin(frontend, backend)`;
   - `create_oauth_state` signs a 10-minute JWT `{typ, fe, be, nonce, exp}` with `JWT_SECRET`;
   - sets an httponly SameSite=Lax nonce cookie (Secure on https);
   - redirects to Google with `redirect_uri = {backend}/auth/google/callback`.
3. `google_callback` does the following:
   - `decode_oauth_state` checks the signature, type, and that `fe` is allowlisted;
   - compares the cookie nonce in constant time (`secrets.compare_digest`);
   - if the state is invalid, redirects to `{default frontend}/login?error=google_state_invalid`;
   - exchanges the code (the token request reuses the same redirect_uri) and fetches userinfo;
   - finds the user **by google_id OR email**. It creates the user if none exists; otherwise it links `google_id` and sets `email_verified=1`;
   - issues a session JWT and redirects to `{fe}/oauth-callback?token=...`.
4. `OAuthCallbackPage` calls `loginWithToken(token)` then `navigate("/")`. `AuthContext` fetches `/auth/me`.

### 8.4 Forgot / reset password
1. `ForgotPasswordPage` → `POST /auth/forgot-password`. The response is always generic. If the account exists, the backend inserts a 1-hour `reset` token and emails `{resolved frontend}/reset-password/{token}`.
2. `ResetPasswordPage` → `POST /auth/reset-password/{token}`. The backend checks the token exists, is unused and unexpired, and that the new password differs from the current one. It then updates the hash and marks the token used. Existing JWT sessions stay valid (there is no revocation).

### 8.5 Public launch (`start_public.ps1`)
1. **Preflight:** checks that `python`, `node` and `ngrok` are on PATH and that `frontend/.env.public.local` exists with `VITE_API_BASE_URL=/api`. The domain comes from `FRONTEND_ALLOWED_HOSTS` or `$env:NGROK_DOMAIN`. It warns if `backend/.env` `FRONTEND_URL`/`BACKEND_URL` don't match, and fails if ports 8000 or 4173 are busy.
2. `vite build --mode public`.
3. Starts `uvicorn main:app --host 127.0.0.1 --port 8000 --no-proxy-headers` and polls `/health` for up to 600 s. The reproduction checks took ~105 s in today's log.
4. Starts `vite preview --mode public` on 127.0.0.1:4173 and polls it.
5. Runs `ngrok http 127.0.0.1:4173 --url https://<domain>` in the foreground. Ctrl+C runs a `finally` block that kills preview and backend. `-Stop` cleans up leftovers.

---

## 9. Authentication & Security

**Mechanism**
- Stateless JWT, HS256, 7-day expiry (`auth_utils.py`), with payload `{sub, email, exp, iat}`.
- The frontend stores the token in **localStorage** (`auth_token`) and sends it as `Authorization: Bearer`.
- Logout only clears the client. There is no server-side revocation.
- bcrypt password hashing. Minimum length is 8; there are no other rules.
- There are **no roles or permissions**: every account is equal, and the only authorization rule is "scan history belongs to `user_id`".
- `JWT_SECRET` is read from `.env`. If it's missing, `config.py:24-30` generates one and **appends it to `.env`** at import time.

**Protections present**
- Rate limits per real client IP (§7). `X-Forwarded-For` is trusted only from peer 127.0.0.1 and read right to left (`rate_limit.py`).
- Origin allowlist for every generated link and redirect, which prevents open redirects. The Host header is never trusted (`public_origin.py`).
- OAuth state is a signed JWT bound to a browser nonce cookie (CSRF protection).
- Forgot-password gives a generic response, so account existence isn't revealed.
- Sanitized 500s. Upload size cap. Chat payload caps.
- Tests force-disable real SMTP and Groq (`tests/conftest.py`).
- Backend `torch.load(..., weights_only=True)`. Scripts (`ml/evaluate.py` etc.) still use `weights_only=False`.

**Visible concerns (my assessment; ordered roughly by impact now that the app is public)**
1. **Account pre-hijacking via Google linking** (`auth_routes.py:379-390`). An attacker can sign up with a victim's email and a password they know; the account stays `email_verified=0`. When the victim later uses "Continue with Google", the callback matches by email, links `google_id`, marks it verified and logs the victim into *that* account. The attacker's password still works, so they can read the victim's scan history. It is also not checked whether Google reports the email as verified (`verified_email`).
2. **Signup can send email to arbitrary addresses** from the configured Gmail account, limited only to 30/hour **per IP**. On a public URL this is a spam and reputation vector for the Gmail account. Forgot-password is 5/hour per IP, with no per-email limit.
3. **Possible memory-exhaustion input.** `_read_and_validate_image` converts the full-resolution image to float32 arrays twice (`main.py:255, 262`). A ≤10 MB, highly compressible PNG with very large dimensions (PIL only *warns* below ~179 MP) could allocate gigabytes. Not tested; flagged as plausible.
4. **Session JWT in a URL query string** (`/oauth-callback?token=`). It can end up in browser history and the ngrok request inspector. `OAuthCallbackPage` navigates with a push, not `replace`.
5. No session invalidation after a password reset or change. The 7-day tokens can't be revoked.
6. `/auth/reset-password`, `/auth/verify` and `/auth/change-password` have no rate limit. The tokens are 256-bit so brute force isn't realistic, but change-password allows unthrottled bcrypt work.
7. `/docs` and `/openapi.json` are publicly reachable through the proxy.
8. Every prediction calls Groq, so public users spend the owner's Groq quota. This is already acknowledged in `DEPLOYMENT_PUBLIC.md` §7.
9. `logs/modality_gate_debug.log` logs every upload's filename and has no rotation (1.2 MB and growing).
10. Secrets hygiene: `backend/.env` holds real Groq, Google, Gmail and JWT secrets, and it is gitignored. `backend/README_AUTH.md` contains the full Google OAuth client ID. That is not a secret, but it is worth knowing about before any public push.

---

## 10. Configuration & Environment

### `backend/.env` (read once by `backend/config.py`)
| Variable | Default if unset | Purpose |
|---|---|---|
| `GROQ_API_KEY` | "" → template fallback | Groq reports/chat (note: `groq_client.py` reads `os.environ` directly, not `config`) |
| `JWT_SECRET` | auto-generated + appended to `.env` | JWT signing (sessions + OAuth state) |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | "" → Google login returns 501 | OAuth |
| `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` | "" → links printed to console | SMTP |
| `FRONTEND_URL` | `http://localhost:5173` | default frontend base for links/redirects |
| `BACKEND_URL` | `http://127.0.0.1:8000` | default backend base; default OAuth redirect_uri |
| `ALLOWED_PUBLIC_ORIGINS` | `FRONTEND_URL` + localhost/127.0.0.1 on :5173 and :4173 | origin allowlist |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | CORS (only relevant for direct dev calls) |
| `DB_PATH` | `backend/app.db` | SQLite file (tests override to `backend/tests/_test.db`) |
| `MODALITY_GATE_CONFIDENCE_THRESHOLD` | `0.90` | gate confidence floor |

**Current state of `.env`:** all secrets are set (per the startup log, which prints only masked or boolean values). `FRONTEND_URL`/`BACKEND_URL` currently point at the **ngrok domain**, with the localhost values commented out. `CORS_ORIGINS` includes localhost + ngrok. `ALLOWED_PUBLIC_ORIGINS` is not set, so the default applies. Plain local dev still works because localhost origins are allowlisted.

### Frontend
| Variable | File | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `.env.public.local` (`/api`); unset in dev → `http://127.0.0.1:8000` | every API call's base |
| `FRONTEND_ALLOWED_HOSTS` | `.env.public.local` | extra Host headers vite accepts (the ngrok domain) |
| `BACKEND_PROXY_TARGET` | optional | where `/api` proxies to |

`start_public.ps1` also reads `$env:NGROK_DOMAIN`. The ngrok authtoken lives in ngrok's own config outside the repo.

**Environment differences:**
- **Local dev:** `npm run dev` on :5173 calls :8000 directly, so CORS matters.
- **Public:** built bundle + `vite preview --mode public` on :4173 → `/api` proxy → :8000, same origin, real IP from `X-Forwarded-For`.
- **Tests:** throwaway DB, SMTP and Groq mocked, but the real checkpoints and data are required.

There are no separate staging or production configs.

---

## 11. Setup, Build, Run & Test

**Prerequisites:**
- Python 3.12 and Node 24.
- The large, gitignored assets: `checkpoints/*.pt` (6 served files), `data/{xray,ct_scan,modality_gate,xray_sanity,ct_sanity}`, `seg_data/` and `backend/ood_stats/*.pt`. **The backend will not start without them**, because the startup checks evaluate the val splits.

```powershell
# Backend deps
pip install -r requirements.txt
pip install torchxrayvision==1.5.4      # REQUIRED but missing from requirements.txt
# (for GPU: install the matching CUDA torch build first, see comment in requirements.txt)

# Frontend deps
cd frontend; npm install

# Run locally (two terminals)
cd backend;  uvicorn main:app --reload --port 8000     # ~1-2 min until "Startup complete"
cd frontend; npm run dev                               # http://localhost:5173

# Public (ngrok) - one-time setup in DEPLOYMENT_PUBLIC.md, then from repo root:
.\start_public.ps1          # Ctrl+C stops everything
.\start_public.ps1 -Stop    # cleanup leftovers

# Tests (backend only; loads real models, ~1 min on the RTX 3050)
cd backend; python -m pytest -q        # result today: 59 passed, 3 deprecation warnings

# Lint / build frontend
cd frontend; npm run lint              # oxlint: 4 warnings today
cd frontend; npm run build

# Offline ML (examples)
cd ml; python train_xray_txrv.py --data_dir ../data/xray --epochs 15
python scripts/compute_ood_thresholds.py --modality both   # must be re-run after any classifier change
```

Deployment means only the ngrok flow. There is no Docker, CI or cloud, by design.

---

## 12. Testing Status

### Backend (`backend/tests/`): 59 tests, all passing (run today)
| File | Covers |
|---|---|
| `conftest.py` | temp DB, session-scoped TestClient (runs the real startup incl. reproduction checks), autouse mocks for SMTP + Groq, `signed_up_user` fixture |
| `test_health.py` | `/`, `/health` report 5 models |
| `test_auth.py` | signup/me, duplicate email, short password, bad email, login ok/wrong/unknown, `/me` without/garbage token, change-password success/wrong/unauth, empty history |
| `test_predict.py` | real X-ray + CT predictions, history saved/not saved, wrong modality → 422, blank → 400, color → 400, non-image/empty → 400, gate-uncertain → 422 (mocked), lung crop not run on gate rejection |
| `test_public_proxy.py` | `client_ip` X-Forwarded-For logic, `/chat` fallback/mocked Groq/empty/over-long/oversized history/20-per-min limit |
| `test_public_origin.py` | origin resolution (proxy, direct, ngrok, non-allowlisted, untrusted peer), reset & verify links per origin, verify `?origin=` rejection, Google flow round-trip per origin with mocked Google, forged/missing/tampered state, non-allowlisted state |

**Gaps:**
- No tests for: reset-password (token expiry, reuse, same-password rule), verify-token expiry/reuse, and Google account **linking** to an existing email (the §9.1 issue).
- No tests for: an OOD rejection with a real image, a sanity-gate rejection with a real photo, a 413 oversize upload, the rate limits on predict/login/signup/forgot, or the `/scans/history` isolation between users.
- Rate-limit fragility: the signup limit (30/hour) is shared across the whole session. Only `test_public_origin.py`/`test_public_proxy.py` reset the limiter, so adding more signup tests could start producing 429s.
- Tests need the full data and checkpoints plus substantial compute, so they are machine-bound.

### Frontend
- **No unit or component tests.**
- `frontend/test_*.cjs` are ad-hoc Playwright scripts from Aug 24–26. They navigate to `http://localhost:5173/` and assume the upload UI appears there. Since the landing-page redesign, logged-out users see `LandingPage`, and the change-password UI they test no longer exists. **Treat them as stale.** I did not run them.

### ML / external
- The startup reproduction checks act as a regression gate for every served model.
- `test/scripts/test_corruption.py` results exist in `test/results/`.
- The external probe set was never populated, so there is **no measured cross-source accuracy**.

---

## 13. Current State of Development

### Version control
**There is no git repository.** `.git/` was intentionally deleted on 2026-09-10 during a storage cleanup (per `PROJECT_CONTEXT.md`), and a fresh `.gitignore` was recreated. `git log` and `git status` both fail with "not a git repository". The author keeps a full Drive backup and plans to push to GitHub once the project is finished.

### Recent activity (file timestamps)
- **2026-09-15 (today):**
  - Public/ngrok work: `start_public.ps1`, `DEPLOYMENT_PUBLIC.md`, `vite.config.js`, `.env.public.example`, `rate_limit.py`, `schemas.py` (chat caps), `api.js`, `tests/test_public_proxy.py`.
  - Then a multi-origin follow-up (17:47–17:55): new `public_origin.py`, changes to `config.py`, `auth_routes.py` and `main.py` startup logging, new `tests/test_public_origin.py`.
  - The backend was started at 18:05. Logs show Google callbacks succeeding for both ngrok and localhost:4173.
  - `scripts/export_graph_metrics.py` and `eval_outputs/graph_metrics.json` (paper figures).
- **2026-09-10:** `CLASSIFIER_EVALUATION_REPORT.md`, CT-volume experiment NO-GO and removal, git removal, docs refresh.
- **2026-09-08:** X-ray crop fix in `seg_utils.py` (live X-ray accuracy on original uploads 0.883 → 0.950 per its docstring).
- **2026-09-06:** X-ray model swapped to torchxrayvision, gate v2, OOD stats recomputed.

### TODO / FIXME / HACK
None in code. A grep across the repo (excluding data and node_modules) finds these words only inside earlier briefing docs. There is also no significant commented-out code in `backend/` or `frontend/src/`. The one leftover is an `eslint-disable` comment in two pages, which is dead since the linter is now oxlint.

### Unfinished, stale or broken parts (verified by reading code)
1. **`requirements.txt` is missing `torchxrayvision`.** `models.py` imports it (through `ml/train_xray_txrv.py`) to load the served X-ray model, so a fresh install fails at startup. `monai` is also missing, but it's research-only.
2. **`scripts/predict_cli.py` (used by `demo/terminal3`) doesn't match the live pipeline:**
   - it loads the old `xray_best_model_phase35_v2.pt` and the v1 gate;
   - it crops *before* the gates;
   - it uses ImageNet preprocessing for X-rays;
   - it compares against `xray_ood.pt`, which was recomputed for the txrv model (threshold 8.72, where the demo script expects 32.9).

   An earlier session verified that this combination rejects real test X-rays as OOD. **Demo-critical.**
3. **Demo scripts are stale.** `demo/terminal3_commands.txt` uses images from `sample images/`, a folder that **no longer exists**. `demo/terminal2_commands.txt` evaluates the old non-served X-ray checkpoint. `ml/evaluate.py` can't load the served txrv checkpoint anyway, because `build_model` rejects backbone `densenet121-res224-all`.
4. **`ml/train_xray_txrv.py:352`** saves `backbone: "txrv_<weights>"`, but `models.is_xrv_backbone()` looks for the prefix `densenet121-res224`. A retrain with this script would produce a checkpoint the backend can't load. The served checkpoint has the un-prefixed name, so it was evidently fixed up by hand.
5. **Frontend loose ends:**
   - `LoginPage` ignores `?error=google_*` from the OAuth callback, so failures show no message.
   - `AuthContext.jsx:71-78` deletes the stored token on *any* `/auth/me` failure, including backend-down or a network error, which logs users out.
   - `/history` is orphaned.
   - `AnalyzingPage.jsx` has an unused `useRef` import.
6. **CT model generalization is an open research problem.** A chain of investigations in `eval_run/` (index: `eval_run/RESUME_STATE.md`) found that the CT classifier relies on acquisition and framing cues rather than lesion features. The documented findings:
   - Retraining variants didn't fix it.
   - A LIDC-IDRI nodule-crop / volume pipeline scored 28.6% on external data and was removed.
   - The notes explicitly recommend not starting further phases before the author's exam.

   The served CT model is unchanged.
7. **Unmeasured parity gap:** the modality gate and `xray_sanity` were trained on pre-cropped `data/xray`, but at inference they see the *uncropped* upload. The disease classifier gets the crop. The 5.5-point offline-to-live X-ray drop (13 rejections) may be related. Not investigated.
8. **Stale comments and docs:**
   - `main.py` docstring (lines 2–3) names the old X-ray checkpoint and "two checkpoints".
   - The `MIN_CONFIDENCE` comment (`main.py:136-142`) calls it the mismatch detector, but it now only drives a warning.
   - `conftest.py` says "all 4 checkpoints".
   - `PROJECT_STATUS.md` says "four models".
   - The ten top-level report docs overlap and partly contradict each other.
   - A blank line in `DEPLOYMENT_PUBLIC.md`'s env table (lines 83–84) breaks the table rendering.
9. `@app.on_event("startup")` is deprecated in this FastAPI version (warning in the test run).

---

## 14. Code Quality Observations

**Strengths / conventions**
- Consistent Python style: module docstrings that explain *why* (often citing the failure that motivated a guard), type hints, small focused modules, and explicit error messages.
- A "fallback, never crash" pattern throughout: Groq falls back to a template, SMTP to the console, and OAuth returns a clear 501.
- Defensive ML engineering: startup reproduction checks, per-model locks (with a documented Grad-CAM hook race), and centralized preprocessing so the OOD check and the classifier see identical tensors.
- The frontend is simple and consistent: function components, a context for auth and theme, one `ApiError` type, and backend `detail` strings surfaced verbatim.
- New code (`public_origin.py`, `rate_limit.py`) is careful and well tested.

**Inconsistencies / duplication / tech debt**
- **Duplicated fetch wrappers:** `api.js` (`predict`, `sendChatMessage`, `getScanHistory`) and `AuthContext.jsx` (`apiPost`, `apiGet`) repeat the same try/parse/throw logic. Only `apiPost` flattens Pydantic `detail` arrays, so a 422 from `/predict` or `/chat` with an array detail would render `[object Object]`.
- **Duplicated UI:** `ScanHistoryPage.jsx` ≈ the history section of `ProfilePage.jsx`.
- **Duplicated constants:** `MODALITY_LABEL` is defined in `main.py`, `report.py`, `chat.py` and `predict_cli.py`. The pipeline thresholds are copied into `predict_cli.py` (and have already drifted). `train_xray_txrv.py` re-inlines `set_seed`, `run_one_epoch` and `plot_curves` from `train.py` (a documented workaround for torchxrayvision's `model` module-name collision).
- **Coupling through `sys.path` manipulation:** backend → `ml/` and `scripts/`. The `model` module-name collision needs the careful import-order hack in `ml/train_xray_txrv.py:90-95`.
- `config.py` docstring says every module reads config from there, but `groq_client.py` uses `os.environ`.
- `config.py` writes to `.env` as an import side effect.
- **Repo clutter:** ~20 PNGs in `frontend/`, logs, text lists and `app.db` inside `backend/`, `test.zip` and PNGs at the root, ~40 unorganized one-off scripts, zipped copies of data folders (`data/*.zip`), and a 111 KB `RESUME_STATE.md`.
- `frontend/README.md` is the untouched Vite template.
- UI copy says "Diagnosis" (`ResultsDisplay.jsx:25`) next to a "not a medical diagnosis" disclaimer. The landing page has a hard-coded decorative "98% confidence" badge.

**Likely bugs** (beyond §13)
- `MainApp.jsx::handleFileSelected` creates object URLs with `URL.createObjectURL` and never revokes them (minor leak).
- `/predict` and `/chat` 422 array details are not flattened (see above).
- `OAuthCallbackPage` leaves `?token=` in history (push navigation).
- `forgot_password` lower-cases the email for lookup but not `ForgotPasswordRequest.email` in general. Harmless.

**Performance**
- Startup takes ~105 s: every model re-evaluates its full val split, and the seg check alone takes 44 s.
- Each prediction runs up to 4 forward passes, a Grad-CAM backward pass and a **synchronous Groq call (up to a 15 s timeout)** inside a threadpool worker. Per-model locks serialize concurrent requests for the same modality.
- `/scans/history` returns every scan with its full base64 heatmap and has no pagination. `scans.user_id` has no index. The history grid uses the full heatmap images as thumbnails.
- `modality_gate_debug.log` grows without bound.

---

## 15. Suggested Next Steps (prioritized)

These respect the project's settled scope decisions: no Docker, CI or cloud; in-memory rate limiting; no change-password UI; the CT dataset stays as is.

**P0: small, and matters for the demo or the public URL**
1. Add `torchxrayvision==1.5.4` to `requirements.txt`, so a clean install can start the backend.
2. Fix the demo path. Either rewrite `scripts/predict_cli.py` to call the same functions `main.py` uses, or drop it from the demo:
   - served checkpoints, v2 gate;
   - gates before the crop;
   - XRV preprocessing;
   - `_read_and_validate_image` logic.

   Then update `demo/terminal2`/`terminal3` to existing image paths and the served model. The simplest robust fix is to factor `_run_prediction`'s core into a function both the route and the CLI import.
3. Close the Google account pre-hijack (§9.1). When linking to an existing account whose email isn't verified, either refuse and ask the user to log in with their password, or clear `password_hash` on link. Also require `verified_email` from Google. Add a test.
4. `LoginPage`: map the `?error=` codes to messages. `AuthContext`: clear the token only on a 401 from `/auth/me`, not on network errors.

**P1: robustness and hygiene**
5. Throttle signup emails per target address (or per address per day) to protect the Gmail account on the public URL.
6. Guard against huge-dimension images: check `img.size` / set `Image.MAX_IMAGE_PIXELS` before the float32 conversions.
7. Use `navigate("/", {replace:true})` in `OAuthCallbackPage`, or move the token to a URL fragment.
8. Fix `train_xray_txrv.py`'s saved `backbone` string. Make `ml/evaluate.py` load XRV checkpoints so the demo's offline evaluation matches the served model.
9. Frontend tidy-up:
   - delete or link `ScanHistoryPage`;
   - share one fetch helper that flattens array `detail`s;
   - revoke object URLs;
   - fix the oxlint warnings;
   - rename "Diagnosis" to "Predicted class".
10. Add a `LIMIT`/pagination to `/scans/history` and an index on `scans(user_id)`. Use a `RotatingFileHandler` for the gate debug log.

**P2: when time allows**
11. Add tests for reset/verify token edge cases, the Google linking rules, a real OOD rejection, 413, and cross-user history isolation. Reset the limiter in an autouse fixture.
12. Replace or delete the stale Playwright `.cjs` scripts.
13. Migrate `@app.on_event` to a lifespan handler. Refresh the stale docstrings and comments listed in §13.8.
14. Consolidate the top-level `.md` reports into one current status doc plus an archive folder. Move clutter (screenshots, logs, zips) out of source folders.
15. If the X-ray live gap matters for the paper or demo, measure whether feeding cropped vs. uncropped images to the gate and sanity models explains the 13 rejections (§13.7).

---

## 16. Open Questions

1. **Deadline:** notes from 2026-09-08 mention an exam roughly 10 days out. Is the exam or demo still around 2026-09-18, and should work be limited to P0 items until then?
2. **Demo format:** will the live demo still include the terminal CLI (`demo/terminal1-3`), or only the web app via ngrok? That decides whether fixing `predict_cli.py` is urgent.
3. **Public exposure:** who will get the ngrok URL (evaluators only, or anyone)? That determines how urgent §9.1–9.3 are.
4. **Should anonymous prediction stay allowed by the API?** The UI requires login, but `/predict/*` accepts anonymous calls, and anyone with the URL can call them directly.
5. **Seg model provenance:** `ml/seg_model.py` says the U-Net was trained externally with no training script in the repo. Is that acceptable for the paper and demo, or should the training code be recovered?
6. **Checkpoint cleanup:** which unused checkpoints (phase36/37 CT, LIDC, old X-ray, v1 gate) must be kept for the paper? About 250 MB could go.
7. **The served X-ray checkpoint's `backbone` field** differs from what the training script writes. Was it edited by hand, or produced by a different script version? I couldn't determine this.
8. **Which of the 10 top-level report docs is authoritative?** They were written on different dates and disagree in places (for example model counts). I could not reconcile them all and relied on the code instead.
9. **`seg_data/` image counts** and the total disk footprint of `data/`, `datasets/` and `raw/` weren't measured (the `du` run timed out).
