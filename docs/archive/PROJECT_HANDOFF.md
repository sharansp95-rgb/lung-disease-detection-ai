# Project Handoff — Lung Disease Detection (Dual-Modality X-ray/CT Classifier)

This document is written for an AI assistant (or engineer) with zero prior exposure to this
repository. It reflects the actual code as of 2026-09-05, not the README or filenames alone.
For a deeper, independently-audited narrative of the ML/data decisions, also read
`PROJECT_STATUS.md` (last reconciled 2026-08-26) and `data/README.md` — this file summarizes
and cross-checks both against the live code rather than replacing them.

**Update (2026-09-10): Git has been removed from this project entirely.** `.git/` (all
commit history) was intentionally deleted along with `.gitignore` during a storage cleanup,
and a fresh `.gitignore` has since been recreated for a possible future `git init`. There is
currently no commit history, branches, or `git status`/`git log`/`git diff` output at all —
every mention below of checking `git status`/`git diff`/`git log` is stale; the working tree
on disk is the only source of truth. See `PROJECT_CONTEXT.md` for the current snapshot.

---

## 1. WHAT IT IS

A **7th-semester student design project (SDP)**: a full-stack web app that classifies chest
**X-ray** images (COVID-19 / Normal / Pneumonia / Tuberculosis) and chest **CT** images
(Benign / Malignant / Normal) using transfer-learning CNNs, with Grad-CAM heatmaps explaining
each prediction. It has full user accounts (email+password and Google OAuth), persistent scan
history, an LLM-backed chat assistant scoped to explaining the tool's own output, and a
multi-layer defense pipeline that rejects non-scan images, wrong-modality images, and
out-of-distribution input before they ever reach the disease classifier.

It is explicitly framed as a **research/decision-support prototype, not a diagnostic tool** —
this disclaimer is baked into the UI, the generated reports, and the chatbot's system prompt.

**Who it's for:** a course deliverable/demo audience, not real patients or clinicians. It runs
entirely locally (no deployment target — see §12).

**Completion: ~90-95%.** "Done" here means: both classifiers trained and verified, backend
API fully wired with auth/history/chat, frontend fully wired to the API, a passing pytest
suite, and an honest, audited account of the ML pipeline's limitations. What's genuinely
missing is polish-level, not architecture-level (see §11). The project owner has explicitly
decided several things are *out of scope* and confirmed them multiple times — see §12's
"settled, do not re-litigate" list.

---

## 2. TECH STACK

**Backend** — Python, FastAPI (`requirements.txt`):
- `fastapi==0.141.1`, `uvicorn==0.52.4`, `pydantic==2.13.4`, `python-multipart==0.0.32`
- `torch==2.11.0` / `torchvision==0.26.0` (CUDA 12.8 wheels on the dev machine; falls back to
  CPU automatically if unavailable — see comment in `requirements.txt`)
- Auth: `PyJWT==2.13.0` (HS256 session tokens), `bcrypt==5.0.0` (password hashing)
- `slowapi==0.1.10` (in-memory per-IP rate limiting)
- `python-dotenv==1.2.3` (env loading), `requests==2.34.2` (Groq + Google OAuth HTTP calls)
- Testing: `pytest==9.1.1`, `httpx==0.28.1` (FastAPI `TestClient` dependency)
- Image/ML support: `numpy`, `pillow`, `opencv-python-headless`, `scikit-image`,
  `scikit-learn`, `scipy`, `matplotlib`, `ImageHash` (used by offline data-audit scripts, not
  the live API)

**Database:** SQLite (`backend/app.db`), accessed via stdlib `sqlite3` directly — **no ORM**.
Schema lives inline in `backend/db.py` as a `CREATE TABLE IF NOT EXISTS` script.

**Frontend** — React 19 + Vite (`frontend/package.json`):
- `react@^19.2.8`, `react-dom@^19.2.8`, `react-router-dom@^7.18.2`
- Build tool: `vite@^8.2.2` + `@vitejs/plugin-react@^6.1.0`
- Lint: `oxlint@^1.79.0`
- No CSS framework — hand-written CSS (`App.css`, `index.css`) with CSS custom properties for
  theming
- `@playwright/test` is a devDependency but **no test files were found** — it's installed, not
  used yet (see §11)

**External APIs:**
- **Groq** (`api.groq.com/openai/v1/chat/completions`) — LLM for report generation and chat.
  Model is `openai/gpt-oss-120b` (see `backend/groq_client.py` docstring: the originally
  spec'd `llama-3.3-70b-versatile` no longer exists on Groq's catalog). Optional — both
  report generation and chat degrade to a template/fallback message if `GROQ_API_KEY` is
  unset or the call fails.
- **Google OAuth 2.0** (accounts.google.com / oauth2.googleapis.com) — "Continue with Google"
  login. Optional — returns a clear 501 if not configured.
- **Gmail SMTP** (`smtp.gmail.com:587`) — verification/reset emails. Optional — falls back to
  printing the email (including the real link) to the backend console/log.

**Deployment target:** **none — intentionally local-only.** CORS is hardcoded to
`http://localhost:5173` / `http://127.0.0.1:5173`. No Docker, CI, or hosting config exists,
and none is expected (confirmed project-scope decision, see §12).

---

## 3. STRUCTURE

```
sdp/
├── PROJECT_STATUS.md         Detailed, independently-audited status doc (models, data, limitations)
├── PROJECT_HANDOFF.md        This file
├── requirements.txt          Python deps for backend/ + ml/ + scripts/
│
├── backend/                  FastAPI application (the live server)
│   ├── main.py                 Routes, prediction pipeline (_run_prediction), startup checks
│   ├── auth_routes.py           /auth/* endpoints, get_current_user[_optional] dependency
│   ├── scans_routes.py          /scans/history + save_scan()
│   ├── models.py                ModelBundle dataclass, load_model_bundle() + reproduction check
│   ├── seg_utils.py             SegBundle, load_seg_bundle(), compute_lung_crop()
│   ├── ood_detector.py          OODDetector (feature-space distance rejection)
│   ├── gradcam_utils.py          preprocess_image(), predict_with_heatmap() — wraps ml/gradcam.py
│   ├── report.py                 generate_report() — Groq + template fallback
│   ├── chat.py                   generate_chat_reply() — Groq + template fallback, system prompt
│   ├── groq_client.py            call_groq() — shared low-level HTTP call to Groq
│   ├── auth_utils.py              hash_password/verify_password, JWT create/decode
│   ├── email_utils.py             send_verification_email/send_password_reset_email (Gmail SMTP)
│   ├── db.py                      SQLite schema + get_db() context manager
│   ├── config.py                  Loads backend/.env; every module reads secrets from here
│   ├── rate_limit.py              Shared slowapi Limiter instance
│   ├── schemas.py                 Pydantic response models (PredictionResponse, ChatRequest, ...)
│   ├── clear_db.py                 Utility script to wipe app.db
│   ├── .env                       Real secrets — gitignored, not in repo history
│   ├── app.db                     SQLite database file (gitignored)
│   ├── ood_stats/                 Precomputed {xray,ct}_ood.pt (centroid + threshold per modality)
│   ├── README_AUTH.md             Setup steps for Google OAuth + Gmail SMTP
│   └── tests/
│       ├── conftest.py             Test DB isolation, session-scoped TestClient fixture
│       ├── test_health.py          / and /health checks
│       ├── test_auth.py            Signup/login/change-password/history-empty tests
│       └── test_predict.py         Full pipeline tests against real sample images
│
├── frontend/                  React + Vite SPA
│   ├── src/
│   │   ├── main.jsx                 Entry point
│   │   ├── App.jsx                   Router setup, RootRoute/RequireAuth guards
│   │   ├── AuthContext.jsx           Token/user state, localStorage-persisted, signup/login/etc.
│   │   ├── ThemeContext.jsx          Dark/light theme, localStorage-persisted
│   │   ├── api.js                    All backend fetch calls, ApiError class
│   │   ├── App.css / index.css       Hand-written styling, CSS variables for theming
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx         Public marketing/entry page (logged-out root)
│   │   │   ├── LoginPage.jsx / SignupPage.jsx   Auth forms + Google OAuth button
│   │   │   ├── ForgotPasswordPage.jsx / ResetPasswordPage.jsx
│   │   │   ├── OAuthCallbackPage.jsx    Consumes ?token= from Google redirect, calls loginWithToken
│   │   │   ├── MainApp.jsx              Logged-in root: modality select + upload
│   │   │   ├── AnalyzingPage.jsx        Fires the real /predict call, shows simulated progress
│   │   │   ├── ResultsPage.jsx          Shows ResultsDisplay + ChatWidget for the finished prediction
│   │   │   ├── ScanHistoryPage.jsx       Standalone history grid (route: /history, currently unlinked from nav)
│   │   │   └── ProfilePage.jsx           Account info + embedded scan history grid (route: /profile)
│   │   └── components/
│   │       ├── AppNavbar.jsx            Persistent top nav (theme toggle, profile avatar)
│   │       ├── ModalitySelector.jsx     X-ray/CT tab toggle
│   │       ├── UploadArea.jsx           Drag/drop + click-to-browse file input, client-side validation
│   │       ├── ResultsDisplay.jsx       Diagnosis card, probability bars, heatmap/original toggle
│   │       ├── ChatWidget.jsx           Floating chat FAB + panel, FAQ quick-reply chips
│   │       └── ThemeToggle.jsx
│   └── package.json
│
├── ml/                        Model architectures + training/eval code (offline, not imported by the live API except via sys.path hacks)
│   ├── model.py                 ImageClassifier (DenseNet-121/ResNet-18 wrapper), forward_features()
│   ├── seg_model.py              UNet (4-level, reverse-engineered from an externally-trained checkpoint)
│   ├── dataset.py                build_transforms/build_datasets/build_dataloaders/compute_class_weights
│   ├── train.py                  Training loop (used to produce the checkpoints in checkpoints/)
│   ├── evaluate.py                Standalone eval script
│   ├── gradcam.py                 GradCAM class, get_target_layer/denormalize/overlay_cam
│   └── smoke_test.py
│
├── scripts/                   One-off data-prep / audit scripts (prepare_*.py, *_audit.py, compute_ood_thresholds.py, etc.) — not part of the live app, but the provenance trail for data/, seg_data/, ood_stats/
│
├── checkpoints/                6 active .pt files loaded at backend startup (gitignored, tracked outside git)
├── data/                        xray/, ct_scan/, modality_gate/, xray_sanity/, ct_sanity/ — processed, model-ready (gitignored; data/README.md has full provenance)
├── seg_data/                    train/val/test images+masks for the segmentation model
├── raw/, datasets/               Original Kaggle downloads (source of truth data/ was built from)
├── audit/, ml_logs/, eval_outputs/   Data-quality audit outputs and training logs from earlier phases
├── report/, documents/, ieee paper/   Academic writeups (not part of the running application)
└── sample images/                Example images for manual demo/testing
```

---

## 4. DATA MODEL

SQLite, `backend/db.py`, 3 tables, no ORM, no migrations tooling (schema changes would be
manual `ALTER TABLE`s — none exist yet since the schema hasn't changed since creation).

**`users`**
| column | type | notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `email` | TEXT UNIQUE NOT NULL | lowercased before insert |
| `password_hash` | TEXT | **nullable** — null for Google-only accounts |
| `email_verified` | INTEGER (bool) | default 0; set to 1 by `/auth/verify/{token}` or automatically on Google signup |
| `google_id` | TEXT UNIQUE | nullable; set on first Google login, or backfilled if a matching email signs in via Google later |
| `display_name` | TEXT | nullable |
| `created_at` | TEXT | `datetime('now')` default |

**`tokens`** (verification + password-reset tokens, unified table)
| column | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `user_id` | INTEGER NOT NULL → `users(id)` | no `ON DELETE` clause (users are never deleted) |
| `token` | TEXT UNIQUE NOT NULL | `secrets.token_urlsafe(32)` |
| `type` | TEXT CHECK IN ('verify','reset') | |
| `expires_at` | TEXT | ISO datetime; verify=24h, reset=1h (`auth_utils.py`) |
| `used` | INTEGER (bool) | default 0; single-use enforcement |
| `created_at` | TEXT | |

**`scans`** (prediction history, one row per authenticated prediction)
| column | type | notes |
|---|---|---|
| `id` | INTEGER PK | |
| `user_id` | INTEGER NOT NULL → `users(id)` | anonymous predictions are never inserted here |
| `modality` | TEXT | `"xray"` or `"ct"` |
| `predicted_class` | TEXT | |
| `confidence` | REAL | top-class softmax probability |
| `class_probabilities` | TEXT | **JSON string**, `json.dumps`'d dict — not normalized into rows |
| `heatmap_png_base64` | TEXT | full data URI, stored inline (no separate blob/file storage) |
| `report` | TEXT | nullable |
| `report_source` | TEXT | `"groq"` or `"template_fallback"` |
| `created_at` | TEXT | |

Relationships: `users 1—N tokens`, `users 1—N scans`. No foreign-key `ON DELETE` behavior is
defined (`PRAGMA foreign_keys = ON` is set per-connection in `get_db()`, but no cascade rule
exists — deleting a user, if that code path were ever added, would currently fail on FK
constraint rather than cascade).

**Migrations:** none exist. `init_db()` runs `CREATE TABLE IF NOT EXISTS` on every startup —
fine for additive schema changes during development, but there's no mechanism to alter
existing columns on a live `app.db` without writing manual SQL.

---

## 5. API / INTERFACES

Base URL: `http://127.0.0.1:8000` (configurable via `BACKEND_URL`). All routes below are from
directly reading `backend/main.py`, `backend/auth_routes.py`, `backend/scans_routes.py`.

| Method | Path | Auth | Request | Response | Purpose |
|---|---|---|---|---|---|
| GET | `/` | none | — | `{service, status, docs, health, note}` | Friendly root, avoids bare 404 |
| GET | `/health` | none | — | `{status, models_loaded[], google_oauth_configured, email_configured}` | Health/readiness check |
| POST | `/predict/xray` | optional (JWT bearer) | multipart `file` | `PredictionResponse` (see below) | Run X-ray pipeline; saves history if authenticated |
| POST | `/predict/ct` | optional | multipart `file` | `PredictionResponse` | Run CT pipeline; saves history if authenticated |
| POST | `/chat` | none | `ChatRequest {message, history[], prediction_context?}` | `ChatResponse {reply, source}` | Chat about the tool/its output |
| POST | `/auth/signup` | none | `{email, password, name?}` | `AuthResponse {access_token, token_type, user}` | Create account, auto-login, sends verify email |
| POST | `/auth/login` | none | `{email, password}` | `AuthResponse` | Password login |
| GET | `/auth/me` | required | — | `UserOut` | Current user info |
| POST | `/auth/change-password` | required | `{current_password, new_password}` | `{message}` | **Backend-only — no frontend UI calls this** |
| GET | `/auth/verify/{token}` | none | — | 302 redirect to `{FRONTEND_URL}/?verified=1` | Consumes email verification link |
| POST | `/auth/forgot-password` | none, rate-limited 5/hr | `{email}` | `{message}` (generic, doesn't leak existence) | Sends reset link (or logs it) |
| POST | `/auth/reset-password/{token}` | none | `{new_password}` | `{message}` | Consumes reset token; rejects reusing current password |
| GET | `/auth/google/login` | none | — | 302 to Google consent screen | Starts OAuth flow; 501 if not configured |
| GET | `/auth/google/callback` | none | query `code`/`error` | 302 to `{FRONTEND_URL}/oauth-callback?token=...` | Completes OAuth, creates/logs in user |
| GET | `/scans/history` | required | — | `list[ScanHistoryItem]` | Full scan history for the current user, newest first |

**`PredictionResponse`** (`backend/schemas.py`):
```json
{
  "modality": "xray" | "ct",
  "predicted_class": "string",
  "confidence": 0.0-1.0,
  "class_probabilities": {"ClassName": 0.0-1.0, ...},
  "heatmap_png_base64": "data:image/png;base64,...",
  "report": "string",
  "report_source": "groq" | "template_fallback",
  "warnings": ["string", ...]
}
```

**Auth:** `Authorization: Bearer <jwt>` header. JWT is HS256, 7-day expiry, payload
`{sub: user_id, email, exp, iat}` (`auth_utils.py`). `get_current_user_optional`
(`auth_routes.py`) lets `/predict/*` work for both logged-in and anonymous callers — anonymous
calls simply skip the `save_scan()` step.

**Rate limits** (`slowapi`, per-client-IP, in-memory): `/predict/*` 20/min, `/auth/login`
20/min, `/auth/signup` 30/hour, `/auth/forgot-password` 5/hour.

**Error shape:** all errors are `{"detail": "..."}`. Unhandled exceptions are caught by a
global handler (`_sanitized_500` in `main.py`) that logs the full traceback server-side but
returns a generic message to the client — no stack traces leak to the frontend.

---

## 6. KEY FLOWS

### 6.1 Signup → auto-login → verification email (not gating)
`SignupPage.jsx` → `AuthContext.signup()` → `POST /auth/signup`
(`auth_routes.py:signup`) → inserts `users` row (email_verified=0) + a `verify` token row →
`send_verification_email()` (`email_utils.py`, real SMTP or console fallback) → returns a
**valid JWT immediately** → frontend calls `setSession()`, storing the token in
`localStorage` and navigating to the app. **Email verification never gates anything** — this
is a confirmed intentional decision (see §12), not a bug.

### 6.2 Login → protected route access
`LoginPage.jsx` → `AuthContext.login()` → `POST /auth/login` (bcrypt `verify_password`
against `users.password_hash`) → JWT returned → stored in `localStorage` under key
`auth_token` → `App.jsx`'s `RequireAuth` wrapper gates `/history`, `/profile`, `/analyzing`,
`/results` by checking `useAuth().user` (populated via `AuthContext`'s `useEffect` calling
`GET /auth/me` whenever `token` changes).

### 6.3 Google OAuth login
`LoginPage.jsx`'s "Continue with Google" link → browser navigates directly to
`GET /auth/google/login` (`auth_routes.py`) → 302 to Google's consent screen → user consents
→ Google redirects to `GET /auth/google/callback?code=...` → backend exchanges the code for
an access token, fetches userinfo, upserts a `users` row (creates new, or attaches
`google_id` to an existing email-matched row, setting `email_verified=1`) → issues a session
JWT → 302 to `{FRONTEND_URL}/oauth-callback?token=<jwt>` → `OAuthCallbackPage.jsx` reads the
`token` query param, calls `loginWithToken()`, navigates to `/`.

### 6.4 Upload → predict → results (the core feature)
`MainApp.jsx` (modality tab + `UploadArea.jsx` drag/drop, client-side type/size check) →
on submit, `navigate("/analyzing", {state: {file, modality, previewUrl}})` →
`AnalyzingPage.jsx` fires `predict(modality, file, token, signal)` (`api.js`) against
`POST /predict/{modality}` while showing a **simulated** step checklist (the backend is a
single request/response, so steps 1-3 advance on fixed timers; step 4 only resolves when the
real response arrives). Backend side, `_run_prediction()` in `main.py`:
1. `_read_and_validate_image()` — size cap (10MB), valid-image check, blank-image std-dev
   check, RGB color-content check (rejects colorful non-scan uploads before grayscale
   conversion would hide that signal)
2. X-ray only: `compute_lung_crop()` (`seg_utils.py`) — U-Net segmentation crop, falls back to
   `border_crop()` if the mask is degenerate
3. `preprocess_image()` (`gradcam_utils.py`) — resize/normalize to the training transform
4. `_sanity_predict()` — per-modality `scan`/`not_scan` gate; 422 if not a scan
5. `_gate_predict()` — xray-vs-ct gate; 422 if wrong modality for the called route
6. `OODDetector.check()` (`ood_detector.py`) — feature-space distance from training centroid;
   422 if out-of-distribution
7. `predict_with_heatmap()` (`gradcam_utils.py` → `ml/gradcam.py`'s `GradCAM`) — classify +
   Grad-CAM overlay, all inside one hold of `bundle.lock` (a `threading.Lock` — see §7 for why)
8. `generate_report()` (`report.py`) — Groq or template
9. If `user` is present (JWT decoded successfully), `save_scan()` (`scans_routes.py`) inserts
   the `scans` row
10. Returns `PredictionResponse` → `AnalyzingPage` navigates to `/results` with it in router
    state → `ResultsPage.jsx` renders `ResultsDisplay.jsx` (probability bars, heatmap/original
    toggle) + a `ChatWidget` pre-loaded with this prediction as context.

### 6.5 Chat about a result
`ChatWidget.jsx` (mounted on both `MainApp` with `predictionContext={null}` and `ResultsPage`
with the actual prediction) → `sendChatMessage()` (`api.js`) → `POST /chat` → `chat()` in
`main.py` → `generate_chat_reply()` (`chat.py`) builds a system prompt from `SYSTEM_PROMPT` +
`_context_block(prediction_context)`, truncates history to `MAX_HISTORY_TURNS=10`, calls
`call_groq()` (`groq_client.py`); on any failure, falls back to `_template_fallback()`. The
system prompt is the actual guardrail — it explicitly refuses off-topic requests and medical
advice, not just a style suggestion.

### 6.6 Forgot password → reset
`ForgotPasswordPage.jsx` → `POST /auth/forgot-password` (always returns the same generic
message regardless of whether the email exists — no account-enumeration leak) → if the email
exists, inserts a `reset` token (1hr expiry) and emails/logs the link →
`ResetPasswordPage.jsx` (route `/reset-password/:token`) → `POST /auth/reset-password/{token}`
→ validates token not used/expired, rejects if the new password equals the current one, then
updates `password_hash` and marks the token used.

---

## 7. CONVENTIONS

**Backend (Python/FastAPI):**
- Every module that touches secrets imports from `backend/config.py` — never reads
  `os.environ` directly elsewhere. `config.py` also has the side effect of **generating and
  persisting `JWT_SECRET` into `.env` if missing**, so sessions survive restarts.
- Fallback discipline is consistent everywhere something is optional (Groq, Gmail, Google
  OAuth): log a clear warning, degrade gracefully, never crash the request path. Both
  `report.py` and `chat.py` return a `(text, source)` tuple so callers/UI can show which path
  was used.
- **Reproduction checks at startup**: every model bundle (`models.py`, `seg_utils.py`)
  re-verifies its checkpoint's claimed accuracy against a fresh eval pass over local data
  before the app is allowed to serve traffic (`ReproductionCheckFailed` /
  `SegReproductionCheckFailed` abort startup). This is a deliberate, repeatedly-useful pattern
  in this codebase — it has caught real checkpoint/data mismatches before. If you add a new
  model bundle, follow this pattern.
- **Concurrency:** each `ModelBundle`/`SegBundle` carries its own `threading.Lock`, held
  across *all* uses of that shared `nn.Module` in a single request (gate predict, OOD check,
  classify+Grad-CAM) — not just individually per call. This exists because Grad-CAM's forward
  hooks are shared mutable state and FastAPI's sync routes run concurrently in a threadpool;
  see the docstring in `models.py` above `ModelBundle` for the exact bug this fixes.
- Heavily commented **decision rationale** (not restating what the code does) throughout
  `main.py`, `models.py`, `ood_detector.py`, `seg_utils.py` — comments explain *why* a
  particular defense/threshold/architecture choice was made, often citing a specific failure
  mode that was tested and observed. Preserve this style if you touch these files — it's load
  bearing for understanding the pipeline's design, not decoration.
- Route split by modality (`/predict/xray`, `/predict/ct`) rather than one endpoint with a
  `modality` field — deliberate, see the module docstring at the top of `main.py`.
- Tests hit the **real** pipeline (real checkpoints, real data, real reproduction checks) —
  no mocking of the ML stack. Only email sending is mocked (`conftest.py`'s
  `_no_real_email` autouse fixture) and the DB is redirected to a throwaway file. This makes
  `pytest` genuinely slow to start (loads 6 checkpoints + reproduction) but real.

**Frontend (React):**
- Functional components + hooks only, no class components.
- Two React Context providers (`AuthContext`, `ThemeContext`) wrap the whole app in
  `App.jsx`; no Redux/Zustand/other state library.
- Navigation carries payload via React Router **location state** (`navigate(path, {state})`)
  rather than URL params or global state — e.g. `MainApp` → `AnalyzingPage` → `ResultsPage`
  pass `file`/`result`/`previewUrl` this way. This means a hard page refresh on `/analyzing`
  or `/results` loses the data (both pages `navigate("/", {replace:true})` if state is
  missing — a graceful redirect, not a crash, but worth knowing).
- All API errors are normalized into a single `ApiError` class (`api.js`) carrying the
  backend's actual `detail` message — components check `e instanceof ApiError` and show
  `e.message` directly rather than a generic string, falling back to a generic message only
  for truly unexpected errors.
- No CSS framework or CSS-in-JS — plain CSS files with custom properties (`--chart-red`,
  `--accent`, etc.) switched by a `data-theme` attribute on `<html>` (`ThemeContext.jsx`).
- No prop-types/TypeScript — plain JS with JSDoc-style comments on some exported functions in
  `api.js`.

**Naming:** snake_case in Python (PEP 8), camelCase in JS/JSX, PascalCase for React
components and Python classes. Backend modules are named after their responsibility
(`auth_routes.py`, `scans_routes.py`, `email_utils.py`) — no `utils.py` grab-bag.

---

## 8. HOW TO RUN

**Backend:**
```
cd backend
pip install -r ../requirements.txt   # or install torch separately first for GPU, see requirements.txt comment
uvicorn main:app --reload --port 8000
```
Environment variables (set in `backend/.env`, auto-loaded by `config.py` — **names only**):
- `GROQ_API_KEY` — optional; report/chat use a template fallback if unset
- `JWT_SECRET` — optional; auto-generated and appended to `.env` if missing
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` — optional; Google login returns 501 if either is missing
- `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` — optional; emails print to console/log if unset
- `FRONTEND_URL` — default `http://localhost:5173`
- `BACKEND_URL` — default `http://127.0.0.1:8000` (also derives `GOOGLE_REDIRECT_URI`)
- `DB_PATH` — default `backend/app.db`

No seed/migration step exists — `init_db()` creates tables on first startup if missing.
`backend/clear_db.py` exists to wipe the DB during development.

**Frontend:**
```
cd frontend
npm install
npm run dev        # Vite dev server, default http://localhost:5173
npm run build       # production build (into frontend/dist/) — not currently deployed anywhere
npm run lint         # oxlint
```
Env var: `VITE_API_BASE_URL` (optional, defaults to `http://127.0.0.1:8000` — see `api.js`
line 1).

**Tests:**
```
cd backend
pytest
```
Runs against real model checkpoints and real data — expect the first test to be slow (model
loading + reproduction checks). Requires `checkpoints/*.pt`, `data/`, `seg_data/`,
`backend/ood_stats/*.pt` to actually be present on disk (they are gitignored but tracked
outside git per the project owner).

No frontend test runner is currently wired up despite `@playwright/test` being installed (see
§11).

---

## 9. CURRENT STATE

**Works and is tested (pytest, `backend/tests/`):**
- Full signup/login/JWT session flow, including duplicate-email/short-password/malformed-email
  rejection (`test_auth.py`)
- Change-password (backend logic, not exposed in UI) — full success/failure/re-login coverage
- Real end-to-end prediction pipeline against real sample images, both modalities
  (`test_predict.py`) — including gate rejection (wrong modality) and history save/no-save for
  logged-in vs anonymous callers
- Blank-image rejection, colorful-non-scan rejection, non-image-file rejection, empty-file
  rejection
- `/` and `/health` liveness (`test_health.py`)

**Works, confirmed live, but not covered by automated tests:**
- Google OAuth end-to-end (confirmed by the project owner with a real Google account,
  2026-08-26) — no test exercises the OAuth code path since it requires a real Google
  round-trip
- Real Gmail SMTP sending (tests force the console-fallback path via `conftest.py`'s
  `_no_real_email` fixture, so the actual `smtplib` send path is untested)
- The entire frontend — no frontend test suite exists yet

**Works but rough / has known trade-offs (see `PROJECT_STATUS.md` §3 for full detail):**
- Tuberculosis-class accuracy regressed slightly (93%→87% on a 60-image sample) after
  switching to segmentation-guided cropping, because TB source images were never trained on
  that crop style — a disclosed, accepted trade-off, not a bug to silently fix
- CT dataset is small (1,036 images) — the project owner has explicitly decided not to grow
  it (see §12)
- OOD detection does not catch "correct modality, out-of-scope class" (e.g. a genuine
  Lung-Opacity X-ray, a real category in the source data that isn't one of the 4 trained
  classes) — by design, documented, not silently wrong

**Stubbed / not enforced:**
- Email verification: the link and flag exist and work, but **nothing checks
  `email_verified`** anywhere — a new account gets a working session token in the signup
  response itself, before the link is ever clicked. This is a confirmed intentional decision,
  not an oversight (see §12).

**Not started:**
- Any deployment/hosting/CI setup (intentionally out of scope, §12)
- Frontend automated tests
- A change-password UI (backend endpoint fully works; intentionally has no frontend entry
  point since forgot-password covers the same need)

---

## 10. KNOWN ISSUES

- **No TODO/FIXME/HACK comments exist in the actual application code** (`backend/`, `ml/`,
  `frontend/src/`) — the only regex hits for those markers are inside a vendored
  `report/diagrams/mermaid.min.js`, unrelated to the app. This is a genuinely clean codebase
  by that metric, not an artifact of not looking.
- **`torch.load(..., weights_only=False)` does not currently appear** — as of this read, all
  checkpoint loads (`models.py`, `seg_utils.py`, `ood_detector.py`) use `weights_only=True`.
  (Prior project notes flagged `weights_only=False` as a pickle-deserialization risk; that no
  longer matches the current code — worth confirming this stays true if you touch checkpoint
  loading.)
- **`scans.class_probabilities` is stored as a JSON string**, not normalized — fine at this
  scale, but any future query that needs to filter/aggregate by per-class probability would
  need to `json.loads()` every row in Python; there's no SQL-level way to query into it.
- **No foreign-key cascade** on `tokens.user_id` / `scans.user_id` → `users.id`. There's no
  user-deletion endpoint today, so this is latent, not active — but if one is ever added, it
  will need to explicitly delete or reassign dependent rows first, or it will hit an FK
  constraint violation (`PRAGMA foreign_keys = ON` is set).
- **`/history` (`ScanHistoryPage.jsx`) and `/profile` (`ProfilePage.jsx`) render nearly
  identical scan-history UI** (same grid, same modal, same fields) — duplicated JSX, not
  shared via a component. Confirmed via `AppNavbar.jsx:20`: the only in-app navigation is the
  profile avatar button, which calls `navigate("/profile")`. **`/history` has no link anywhere
  in the UI** — it's only reachable by typing the URL directly. Likely dead code from an
  earlier design (`PROJECT_STATUS.md` calls `/history` "standalone" and `/profile` "current
  profile page design," consistent with `/history` being the superseded version).
- **CORS is hardcoded** to two localhost origins in `main.py` — fine for local-only scope, but
  if a demo ever needs to run frontend and backend on different hosts, this needs updating.
- **In-memory rate limiting** (`slowapi`) resets on every backend restart and doesn't share
  state across multiple worker processes — intentional per project scope (single-process
  local demo), not a bug, but would need a Redis-backed limiter for anything beyond that.
- **Segmentation model has no training script in this repo** (`PROJECT_STATUS.md` §2) — its
  architecture in `ml/seg_model.py` was reverse-engineered from the checkpoint's `state_dict`.
  If it ever needs retraining, that pipeline doesn't exist here and would need to be rebuilt
  from scratch.

---

## 11. WHAT'S LEFT

Ordered by priority for reaching a polished, demo-ready state:

1. **(Low effort) Fix any remaining stale prose in `PROJECT_STATUS.md`.** Per prior project
   notes, §1's overview paragraph may still say "four models" instead of six — a one-line
   text fix, `PROJECT_STATUS.md` only, no code impact. Worth a final pass before any
   demo/submission to make the documentation internally consistent.
2. **(Low-medium effort) Resolve the `/history` vs `/profile` duplication.** `/history`
   (`ScanHistoryPage.jsx`) has no in-app link anywhere (§10, §12) and duplicates
   `ProfilePage.jsx`'s embedded history grid almost verbatim. Either delete the route/page, or
   extract the shared grid+modal into one component (e.g. `components/ScanHistoryGrid.jsx`)
   and give `/history` a real purpose/entry point. Touches:
   `frontend/src/pages/ScanHistoryPage.jsx`, `frontend/src/pages/ProfilePage.jsx`,
   `frontend/src/App.jsx` (route removal, if deleting), possibly a new file under
   `frontend/src/components/`.
3. **(Medium effort) Add at least a handful of frontend tests.** `@playwright/test` is already
   a devDependency with no config or spec files — either wire up a minimal Playwright config
   and a couple of end-to-end specs (signup → predict → view result), or remove the unused
   dependency if frontend testing isn't going to happen before submission. Touches:
   `frontend/package.json`, new `frontend/tests/` or `frontend/e2e/` directory.
4. **(Low effort, optional) Decide and document a final stance on email verification
   enforcement.** Currently informational-only by deliberate choice (§12) — if a reviewer is
   likely to ask "why can you use the app before verifying your email," having a one-line
   answer ready (or flipping it to gate `/scans/history` behind `email_verified`) is worth
   deciding explicitly rather than leaving implicit. No code change required unless the
   decision changes.
5. **(As-needed) Re-run the full `pytest` suite and a manual click-through of every frontend
   route** before any demo. There is no Git history to diff against (see the update note at
   the top of this document) — the working tree on disk is the only source of truth, and
   nothing in this handoff has been re-verified against its current state.

Explicitly **not** on this list because the project owner has already decided against them:
deployment/CI/Docker, growing the CT dataset, a change-password UI, Redis-backed rate
limiting. See §12.

---

## 12. OPEN QUESTIONS

- **No Git repository.** At the time this handoff was originally written, `git status` showed
  many modified and untracked files not yet committed (backend prediction/model files,
  frontend pages, sample images, renamed/added documents), and the project owner had
  deliberately deferred pushing to GitHub until the project was "fully done." Git has since
  been removed from the project entirely (see the update note at the top of this document) —
  there is no `git log`/`git diff`/`git status` to consult at all anymore. Check the working
  tree directly.
- **Is `/history`/`ScanHistoryPage.jsx` meant to be removed, or re-linked somewhere?** It's
  fully functional but has zero in-app entry points (confirmed — see §10). Neither
  `PROJECT_STATUS.md` nor the code indicates an intended future use for it distinct from
  `/profile`. Decide: delete the route + page, or link to it from somewhere (e.g. as a
  dedicated "view all history" link off `/profile` if the embedded grid there is meant to be
  abbreviated) — this reads as accidental leftover, not a deliberate two-surface design.
- **Settled, do not re-litigate** (confirmed by the project owner on the dates noted — raising
  these again as "gaps" will re-tread already-closed ground):
  - No deployment/CI/Docker/hosting — explicitly out of scope for this SDP (confirmed
    2026-08-26).
  - In-memory rate limiting is intentional, not a shortcut (confirmed 2026-08-26).
  - No change-password frontend UI despite a working backend endpoint — forgot-password
    already covers the need (confirmed 2026-08-26).
  - CT dataset size (1,036 images) will not be grown — confirmed twice (2026-08-26 and
    2026-09-01) after being raised as a weakness both times.
  - Google OAuth was live-tested end-to-end with a real account (2026-08-26) — not an open
    risk.
  - Email verification is deliberately informational-only, not access-gating (confirmed
    2026-08-26) — see item 4 in §11 if this ever needs revisiting.
