# Project Brief — Lung Disease Detection (Dual-Modality Chest X-ray / CT Classifier)

Snapshot date: 2026-09-15. Written for a reader with no prior knowledge of the project. All paths are relative to the repo root (`D:\7th sem\sdp` on the author's Windows machine).

---

## 1. What this is

A full-stack web app that takes **one chest image** (a chest X-ray or a single CT slice), classifies it with a PyTorch CNN, and returns:

- the predicted class and per-class probabilities
- a Grad-CAM heatmap (a picture showing which parts of the image drove the prediction)
- a short plain-language report written by an LLM (Groq), or a fixed template if Groq is unavailable
- an optional chat assistant that answers questions about that result

Classes:
- **X-ray:** `COVID-19`, `Normal`, `Pneumonia`, `Tuberculosis`
- **CT:** `Benign`, `Malignant`, `Normal`

Users have accounts (email/password or Google), and logged-in users get their scan history saved.

**SDP** = "student design project". The repo's own docs (`PROJECT_HANDOFF.md:20`, `PROJECT_UNDERSTANDING.md:7`) call it a "7th-semester student design project (SDP)". This is **coursework**: it runs locally only and is presented as a research/decision-support demo, **not a clinical tool**. Every report carries a "not a medical diagnosis" disclaimer. The intended users (inferred) are the student team, their evaluators, and demo audiences. Deliverables live in `documents/` (a review PPTX and an IEEE-style paper PDF titled "pulmo").

What makes it unusual for a student project: a lot of effort went into **ML integrity**. There are audits for data leakage, shortcut learning and label-correlated artifacts. There is a layered input-rejection pipeline, and a startup check that refuses to boot if a checkpoint's accuracy doesn't reproduce on local data.

---

## 2. Stack and runtime

**Backend (Python 3.12.10, pip, no virtualenv).** Pinned in `requirements.txt`:
- fastapi 0.141.1, uvicorn 0.52.4, pydantic 2.13.4, python-multipart 0.0.32, python-dotenv 1.2.3, requests 2.34.2
- bcrypt 5.0.0, PyJWT 2.13.0, slowapi 0.1.10 (rate limiting)
- torch 2.11.0 (installed locally as `+cu128`), torchvision 0.26.0
- numpy 2.5.2, pillow 12.3.0, opencv-python-headless 5.0.0.93, scikit-image 0.26.0, scikit-learn 1.9.0, scipy 1.18.1, matplotlib 3.11.1, ImageHash 4.3.2
- pytest 9.1.1, httpx 0.28.1

**Installed but NOT in requirements.txt:**
- `torchxrayvision` 1.5.4 — **required by the live X-ray model**
- `monai` 1.6.0 — used only by an offline script

**Frontend (npm, `frontend/package-lock.json`).**
- Runtime: react / react-dom 19.2.8, react-router-dom 7.18.2
- Build: vite 8.2.2, @vitejs/plugin-react 6.1.0
- Lint: oxlint 1.79.0
- Browser automation: @playwright/test 1.62.1, used only by ad-hoc scripts
- Plain JavaScript/JSX with no TypeScript and hand-written CSS (`frontend/src/App.css`, ~1,700 lines)

**Database:** SQLite through the stdlib `sqlite3` module, with no ORM. The file is `backend/app.db`.

**External services** (all optional; each falls back gracefully):
- Groq chat completions API, model `openai/gpt-oss-120b`
- Google OAuth 2.0
- Gmail SMTP (`smtp.gmail.com:587`)

**Hardware:** the dev machine has an RTX 3050 Laptop GPU (4 GB). Code falls back to CPU. Training for the served checkpoints was done on Colab (per script docstrings).

**Models served** (paths in `backend/main.py:86-103`; metadata read from the checkpoint files):

| Key | File | Architecture | Classes | Stored val_acc |
|---|---|---|---|---|
| xray | `checkpoints/xray_txrv_densenet121_best.pt` | torchxrayvision DenseNet121 (`densenet121-res224-all`) | 4 | 0.9646 |
| ct | `checkpoints/ct_best_model_phase35_v2.pt` | ImageNet DenseNet121 | 3 | 0.9904 |
| gate | `checkpoints/modality_gate_best_model_v2.pt` | ResNet18 | xray / ct | 1.0 |
| xray_sanity | `checkpoints/xray_sanity_best.pt` | ResNet18 | not_scan / scan | 0.9996 |
| ct_sanity | `checkpoints/ct_sanity_best.pt` | ResNet18 | not_scan / scan | 1.0 |
| seg | `checkpoints/xray_seg_model.pt` | U-Net (1→1 channel) | lung mask | dice 0.989 |

Supporting files: OOD statistics in `backend/ood_stats/{xray,ct}_ood.pt`. The thresholds are 8.72 over 15,373 training images (X-ray) and 25.62 over 829 (CT).

---

## 3. Repo layout

```
.
├── backend/            Live FastAPI service, SQLite DB, OOD stats, pytest suite
│   ├── ood_stats/      Per-modality feature centroid + threshold (.pt)
│   └── tests/          26 integration tests (real models, throwaway DB)
├── frontend/           React/Vite SPA
│   ├── src/pages/      Route-level screens (landing, auth, analyze, results, history, profile)
│   ├── src/components/ Navbar, upload area, modality selector, results panel, chat widget
│   └── dist/           Build output (ignore)
├── ml/                 Reusable model/dataset/Grad-CAM code + offline train/evaluate CLIs
├── scripts/            ~40 one-off data-prep, audit, OOD-threshold, figure scripts
├── checkpoints/        .pt weights (served ones + experimental ones)
├── data/               Model-ready ImageFolder splits: xray, ct_scan, modality_gate, *_sanity
├── seg_data/           Images/masks for the lung-segmentation U-Net
├── raw/                Raw Kaggle downloads consumed by prep scripts
├── datasets/           Duplicate browsing copy of raw + processed data (not wired into code)
├── audit/              Leakage / TB-artifact / scanner-table audit outputs
├── eval_outputs/       Test metrics, confusion matrices, Grad-CAM grids per checkpoint
├── eval_run/           Research log: evaluation harnesses, phase A–J write-ups, RESUME_STATE.md
├── ml_logs/            Training-curve PNGs and history JSONs
├── diagnosis_outputs/  Generalization diagnostic figures
├── test/               External-data generalization probe harness (not pytest)
├── demo/               Scripted terminal commands + expected output for live demo
├── documents/          Course deliverables (PPTX, paper PDF)
└── logs/               modality_gate_debug.log (written by backend at runtime)
```

Ignore `.tools_tmp/` and `.claude/` (tooling scratch). The root also holds 8 overlapping Markdown reports: `PROJECT_CONTEXT.md` is the newest and most accurate; `PROJECT_HANDOFF.md` is partly stale.

---

## 4. Entry points and control flow

**Backend entry point:** `backend/main.py` defines `app`. Run it from inside `backend/` (imports are bare module names like `import config`).

**Startup** — `startup_load_models()` (`main.py:171`):
1. Calls `init_db()` (`db.py`), which runs `CREATE TABLE IF NOT EXISTS`.
2. For each name in `MODEL_NAMES`, calls `models.load_model_bundle()` (`models.py:123`). This does the following:
   - Loads the checkpoint with `weights_only=True`.
   - Picks the architecture: `XRVClassifier` if `is_xrv_backbone()`, otherwise `ml/model.py:build_model`.
   - **Re-evaluates the full `data/<name>/val` split.** Raises `ReproductionCheckFailed` if accuracy differs from the stored `val_acc` by more than 0.001, which aborts startup.
3. Builds an `OODDetector` for `xray` and `ct`.
4. Calls `seg_utils.load_seg_bundle()`, which runs the same reproduction check (dice/IoU on `seg_data/val`).
5. Logs whether Groq, Google OAuth and SMTP are configured.

Startup is slow: the test suite takes ~2 minutes, mostly this step.

**Prediction request** — `POST /predict/{xray|ct}` → `_run_prediction()` (`main.py:300`):
1. `_read_and_validate_image()` rejects, in order:
   - files over 10 MB → 413
   - empty or undecodable files → 400
   - blank images (grayscale std < 10) → 400
   - colorful images (mean channel spread > 20) → 400
2. `preprocess_image()` (`gradcam_utils.py:32`) applies ImageNet normalization: grayscale→3 channels, 224×224.
3. **Sanity gate:** `_sanity_predict()` rejects anything that isn't a scan → 422.
4. **Modality gate:** `_gate_predict()` → 422 if confidence < `MODALITY_GATE_CONFIDENCE_THRESHOLD` (0.90), or if the predicted modality ≠ the route. Every call is logged to `logs/modality_gate_debug.log`.
5. **X-ray only:** `seg_utils.compute_lung_crop()` runs the U-Net mask, takes its bounding box, crops the original image, and squashes it to the training size. It falls back to `border_crop` if the mask is degenerate. The image is then re-preprocessed with `preprocess_image_xrv()` (1 channel, range ~[-1024, 1024]).
6. Under `bundle.lock`:
   - `OODDetector.check()` computes the L2 distance between the penultimate-layer embedding and the training centroid; above threshold → 422.
   - `predict_with_heatmap[_xrv]()` runs Grad-CAM (`ml/gradcam.py`).
7. Adds a low-confidence warning if confidence < `MIN_CONFIDENCE[modality] + 0.15`.
8. `report.generate_report()` → Groq, or the template on any exception.
9. If a valid bearer token is present, `scans_routes.save_scan()` stores the result.
10. Returns `PredictionResponse`.

**Frontend entry:** `frontend/src/main.jsx` → `App.jsx`. This wraps everything in `ThemeProvider`, `AuthProvider` and `BrowserRouter`.

**Main user path:**
1. `/` shows `LandingPage` when logged out, `MainApp` when logged in.
2. In `MainApp` the user picks a modality and a file. The `File` object is passed via router state to `/analyzing`.
3. `AnalyzingPage` calls `api.predict()` with an `AbortController`. The progress steps shown are simulated timers.
4. On success, the result goes via router state to `/results`, where `ResultsPage` shows `ResultsDisplay` plus a `ChatWidget` given the prediction context.

**Offline flows** (command-line scripts):
- Data prep: `scripts/prepare_xray.py`, `prepare_ct_scan.py`, `prepare_modality_gate.py`, `build_sanity_dataset.py`
- Training: `ml/train.py`, `ml/train_xray_txrv.py`
- OOD thresholds: `scripts/compute_ood_thresholds.py`
- Evaluation: `ml/evaluate.py`

---

## 5. Data model

**Schema:** `backend/db.py:16-49`, applied at startup. There is no migration framework: schema changes mean editing `SCHEMA` and deleting or altering `app.db` by hand.

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT,                 -- NULL for Google-only accounts
  email_verified INTEGER NOT NULL DEFAULT 0,   -- informational only, gates nothing
  google_id TEXT UNIQUE,
  display_name TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')));

CREATE TABLE tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id),
  token TEXT UNIQUE NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('verify','reset')),
  expires_at TEXT NOT NULL,           -- ISO-8601 with UTC offset
  used INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')));

CREATE TABLE scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id),
  modality TEXT NOT NULL,             -- 'xray' | 'ct' (not constrained)
  predicted_class TEXT NOT NULL,
  confidence REAL NOT NULL,
  class_probabilities TEXT NOT NULL,  -- JSON string
  heatmap_png_base64 TEXT NOT NULL,   -- full data: URI, ~100s of KB each
  report TEXT, report_source TEXT,    -- 'groq' | 'template_fallback'
  created_at TEXT NOT NULL DEFAULT (datetime('now')));
```

Relationships: `users` 1:N `tokens`, `users` 1:N `scans`. Foreign keys are enabled on each connection; there are no cascades.

**API models:** `backend/schemas.py` holds `PredictionResponse`, `ChatRequest`, `ChatResponse` and `PredictionContext`. Auth request/response models live inline in `auth_routes.py:39-102`.

**Checkpoint format:** a dict with `model_state_dict`, `class_names` (or `classes`), `backbone` (or `architecture`) and `val_acc`. Seg checkpoints carry `val_dice` / `val_iou` instead.

**Datasets** (`datasets/README.md`, `data/README.md`):
- **X-ray:** COVID-19 Radiography DB + TB Chest Radiography DB (Kaggle). Split 15,373 / 1,922 / 1,922 (train/val/test).
- **CT:** IQ-OTH/NCCD. Split 829 / 104 / 103, grouped so near-duplicate slices never cross splits. 61 images from a different acquisition protocol were excluded.
- Folder layout is `ImageFolder` style: `data/<name>/{train,val,test}/<Class>/`.

---

## 6. Interfaces

**HTTP API** (FastAPI; OpenAPI docs at `/docs`):

| Method & path | Auth | Rate limit | Purpose |
|---|---|---|---|
| POST `/predict/xray`, `/predict/ct` | optional Bearer | 20/min per IP | multipart field `file`; returns `PredictionResponse` |
| POST `/chat` | none | none | `{message, history[], prediction_context}` → `{reply, source}` |
| GET `/`, `/health` | none | none | status; `/health` lists loaded models and OAuth/email flags |
| POST `/auth/signup` | none | 30/hour | creates user, sends verify email, **returns a session** immediately |
| POST `/auth/login` | none | 20/min | `{email, password}` → `{access_token, user}` |
| GET `/auth/me` | Bearer | none | current user |
| POST `/auth/change-password` | Bearer | none | exists and is tested, but has no UI (deliberate) |
| GET `/auth/verify/{token}` | none | none | marks verified, redirects to `FRONTEND_URL/email-verified` |
| POST `/auth/forgot-password` | none | 5/hour | always returns the same generic message |
| POST `/auth/reset-password/{token}` | none | none | `{new_password}` |
| GET `/auth/google/login`, `/auth/google/callback` | none | none | OAuth; callback redirects to `FRONTEND_URL/oauth-callback?token=<jwt>` |
| GET `/scans/history` | Bearer | none | all of the user's scans, newest first, no pagination |

**Error shapes:**
- Rejections are `{"detail": "<human message>"}`. The frontend shows `detail` verbatim.
- Pydantic validation errors return 422 with a `detail` array; `AuthContext.apiPost` joins the `msg` fields.
- Unhandled exceptions are sanitized to a generic 500 (`main.py:220`).

**Frontend routes** (`App.jsx`):
- Public: `/`, `/login`, `/signup`, `/forgot-password`, `/reset-password/:token`, `/oauth-callback`, `/email-verified`
- Behind `RequireAuth`: `/history`, `/profile`, `/analyzing`, `/results`

**CLIs:**
- `ml/train.py --data_dir --backbone {densenet121,resnet18} --epochs`
- `ml/train_xray_txrv.py`
- `ml/train_ct_monai.py` (RadImageNet, not adopted)
- `ml/evaluate.py --data_dir --checkpoint --output_dir --run_name`
- `ml/smoke_test.py`
- `scripts/predict_cli.py` (see Rough edges)
- `scripts/compute_ood_thresholds.py`

**External APIs consumed:**
- Groq `POST https://api.groq.com/openai/v1/chat/completions` (15 s timeout)
- Google OAuth token and userinfo endpoints
- Gmail SMTP with STARTTLS

---

## 7. Configuration

**`backend/.env`** is loaded by `backend/config.py`. It is the only place backend code reads configuration; `groq_client.py` reads `os.environ` directly after dotenv has loaded. All values are redacted here.

| Key | Required? | Controls |
|---|---|---|
| `GROQ_API_KEY` | optional | LLM report + chat; absent → template text |
| `JWT_SECRET` | effectively required | HS256 signing. **If missing, `config.py` generates one and appends it to `.env` at import time.** |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | optional (both needed) | Google sign-in; missing → 501 |
| `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` | optional (both needed) | real email; missing → links printed to backend log |
| `FRONTEND_URL` | optional (default `http://localhost:5173`) | redirect targets, reset links |
| `BACKEND_URL` | optional (default `http://127.0.0.1:8000`) | verify links, OAuth redirect URI (must match Google Console) |
| `DB_PATH` | optional (default `backend/app.db`) | SQLite file; tests override it |
| `MODALITY_GATE_CONFIDENCE_THRESHOLD` | optional (default 0.90) | gate uncertainty cutoff |

**Frontend:** `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`, `frontend/src/api.js:1`).

**Hard-coded values that behave like config:**
- CORS origins (`main.py:158`)
- `MAX_UPLOAD_BYTES` 10 MB (also mirrored in `UploadArea.jsx`)
- Blank and color thresholds
- JWT expiry 7 days, verify-token expiry 24 h, reset-token expiry 1 h (`auth_utils.py`)
- `GROQ_MODEL` (`groq_client.py:20`)
- Checkpoint paths (`main.py:86`)

Setup notes for OAuth and SMTP are in `backend/README_AUTH.md`.

---

## 8. Build, test, run, deploy

```powershell
# Python deps (repo root). Also required, because it's missing from the file:
pip install -r requirements.txt
pip install torchxrayvision==1.5.4

# Backend (must run from backend/; needs checkpoints/, data/, seg_data/ on disk)
cd backend
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # -> frontend/dist
npm run lint       # oxlint; currently warnings only

# Tests
cd backend
python -m pytest   # pytest.ini: testpaths = tests
```

**Tests:**
- pytest with FastAPI `TestClient`, 26 tests: 13 auth, 2 health, 11 predict.
- **No mocks for models.** The real checkpoints and data load, so tests require the full local dataset.
- `conftest.py` sets `DB_PATH` to `backend/tests/_test.db` before importing the app, and monkeypatches email to console-only.
- Verified today (2026-09-15): **26 passed, 3 warnings** (FastAPI `on_event` deprecation) in 119 s.
- No coverage tooling is configured, so coverage is unknown.
- There are no frontend unit tests. `frontend/test_*.cjs` are manual Playwright scripts that drive a running stack and take screenshots.

**Model evaluation evidence** (`CLASSIFIER_EVALUATION_REPORT.md`, 2026-09-10):

| Model | Offline test accuracy | Through live HTTP |
|---|---|---|
| X-ray | 96.57% | 91.05% (13 rejections, COVID-19 precision 0.78) |
| CT | 98.06% | 98.06% |

No external-dataset evaluation exists: `test/external_probe/` contains placeholders only.

**CI / deploy:** none, deliberately. The project is local-only, with no Docker and no hosting. The demo runbook is in `demo/terminal{1,2,3}_commands.txt`.

---

## 9. Conventions in play

- **Docstrings explain *why*.** Module docstrings are long design rationales that often cite the failure that motivated the code. Match that style: justify guards with the specific failure they catch.
- **Reproduction checks are sacred.** Any new served checkpoint needs a matching `data/<name>/val` split and a stored `val_acc` that reproduces within 0.1 pp, or the app won't start.
- **Preprocessing must match training exactly.** The X-ray model uses torchxrayvision normalization. CT and the gates use ImageNet normalization. Mixing them fails silently; `compute_lung_crop` must end in `crop_and_resize`.
- **Concurrency:** every use of a model goes through `bundle.lock` (a `threading.Lock`) because Grad-CAM registers hooks on shared models. OOD check and Grad-CAM must run in *one* lock hold.
- **Errors:**
  - Raise `HTTPException` with a user-facing `detail` string: 400 for bad input, 413 for oversize, 422 for a guardrail rejection (sanity, modality, OOD), 401 for auth, 501 for unconfigured OAuth.
  - External calls (Groq, SMTP) catch *all* exceptions and degrade to a fallback, logging a warning.
- **Logging:** stdlib `logging`, named loggers under `ct_xray_api.*`, and a separate file logger for gate decisions.
- **Auth:**
  - Stateless JWT (HS256, 7 days) in the `Authorization: Bearer` header; the frontend stores it in `localStorage["auth_token"]`.
  - Use `get_current_user` or `get_current_user_optional` as FastAPI dependencies.
  - Email verification does **not** gate anything, by design.
- **DB access:** raw parameterized SQL inside `with get_db() as conn:`, which commits on exit. Rows become dicts via `sqlite3.Row`.
- **Imports:**
  - The backend adds `ml/` and `scripts/` to `sys.path` and imports bare names (`from model import build_model`, `from prepare_xray import border_crop`).
  - torchxrayvision also registers a module called `model`. Imports of `train_xray_txrv` are therefore **deferred inside functions** on purpose (`models.py:98-105`). Don't hoist them to module top level.
- **Frontend:**
  - Function components and hooks, fetch wrappers that throw `ApiError(message, status)`.
  - Pages pass data (`file`, `result`) through `navigate(..., {state})`, not global state, so refreshing `/results` bounces to `/`.
- **Naming:** checkpoints use `<modality>_<desc>_phase<NN>[_vN].pt`; "phase" numbers refer to the development log, not to anything in the code.

---

## 10. Current state

- **Git: there is no repository.** `.git/` was deliberately deleted in a storage cleanup around 2026-09-10 (stated in `PROJECT_CONTEXT.md`). A fresh `.gitignore` exists. So there is no branch, no commit history and no diff; "uncommitted work" doesn't apply. Everything below comes from file timestamps and in-repo logs.
- **Recent activity by theme** (from mtimes and `eval_run/`):
  - **2026-09-06 → 09-10: CT generalization investigation.** A long chain of time-boxed phases (`eval_run/ct_*.md`, `phaseA…phaseJ_*.md`, running log `eval_run/RESUME_STATE.md`):
    - Whole-slice CT retrains (phase36/37 checkpoints) all still learned shortcuts instead of lesions.
    - A LIDC-IDRI nodule-crop CT-volume pipeline was built and briefly deployed as `/predict/ct/volume`. It scored 28.6% on external data (`phaseJ`, NO-GO) and was then **removed from the codebase entirely**.
    - Leftovers: `checkpoints/ct_lidc_nodulecrop_v4_stack3.pt` and `eval_run/phaseE/`.
  - **2026-09-08: X-ray crop fix** (`seg_utils.py`, `eval_run/xray_cropfix_applied.md`). Live X-ray accuracy on original uploads went from 0.883 to 0.950.
  - **2026-09-06: X-ray model swap** to torchxrayvision DenseNet121; OOD stats recomputed for it.
  - **2026-09-10:** final evaluation report, `main.py` last edited, storage cleanup, Git removal.
  - **2026-09-14:** manual dev runs (backend log shows signups and history calls).
- **Code markers:** no TODO / FIXME / HACK / XXX comments in `backend/`, `ml/`, `scripts/`, `frontend/src/`, `test/` or `eval_run/`.
- **Tests:** all 26 pass, none skipped. There are 3 oxlint warnings (unused `useRef` in `AnalyzingPage.jsx`, two fast-refresh export warnings) plus one `set-state-in-effect` warning in `AuthContext.jsx`.
- **Dependencies:**
  - `npm audit`: 0 vulnerabilities. `npm outdated`: minor bumps only (react 19.3.0, vite 8.3.0, oxlint 1.83.0, …).
  - Python: torch 2.14 and PyJWT 2.14 are available. No Python vulnerability scan has been run (pip-audit isn't installed), so that status is unknown.
- **Deliberate, settled scope decisions** (do not "fix"): no deployment/CI, in-memory rate limiting, no change-password UI, no email-verification gating, CT dataset won't be expanded.

---

## 11. Rough edges

1. **`torchxrayvision` is missing from `requirements.txt`.** A clean install cannot start the backend: the X-ray bundle imports it.
2. **The CT classifier doesn't actually detect lesions.** The project's own research shows this (`eval_run/ct_framing_normalization_experiment.md`):
   - Grad-CAM lung-field enrichment is 1.00× (no preference for the lungs), and ~23% of heatmap mass lands on constant background.
   - Externally sourced CT images are called "Malignant" ~79% of the time.
   - The 98% test accuracy is in-domain (IQ-OTH/NCCD) only. Treat CT output as a demo of a pipeline, not of detection.
3. **The live X-ray path loses 5.5 pp versus offline evaluation** (96.6% → 91.1%). Serving the same test images makes COVID-19 precision fall to 0.78. The cause is the segmentation-crop and preprocessing path; it is only partly fixed.
4. **The demo CLI is stale.** `scripts/predict_cli.py:42-44` still loads `xray_best_model_phase35_v2.pt` and gate v1, but `backend/ood_stats/xray_ood.pt` was recomputed for the torchxrayvision model. A prior check (2026-09-11) found it rejects essentially all real X-rays as OOD. `demo/terminal2_commands.txt` likewise evaluates the old, unserved X-ray checkpoint.
5. **Retraining the X-ray model breaks the backend.** `ml/train_xray_txrv.py:352` saves `backbone="txrv_<weights>"`, but `models.is_xrv_backbone()` only matches the prefix `densenet121-res224`. A fresh checkpoint would route to `build_model` and fail with "Unsupported backbone". The served checkpoint has `densenet121-res224-all`, so it was produced or edited some other way (exact provenance unknown).
6. **External-probe harness mislabels rejections.** `test/scripts/test_external.py:classify_rejection` buckets OOD rejections as `SANITY_GATE`: the OOD message ("This does not look like a…", `main.py:409`) never contains "out of distribution".
7. **Startup is heavy and data-coupled.**
   - Every boot re-evaluates ~2,000+ validation images across 6 models.
   - The app cannot start at all without `data/` and `seg_data/`.
   - The test suite cannot run on a machine without the full datasets.
8. **Auth soft spots:**
   - `/chat` is unauthenticated and unrate-limited, so anyone reaching the port can spend the Groq quota.
   - `/auth/reset-password` and `/auth/verify` have no rate limit.
   - Changing or resetting a password doesn't invalidate existing JWTs or outstanding reset tokens.
   - The JWT travels in a URL query string on the OAuth redirect and sits in `localStorage`.
   - The Google callback links to any existing account with the same email (`auth_routes.py:346`).
   - SMTP sends synchronously inside the signup request.
9. **Frontend auth quirks:**
   - `AuthContext.jsx:73` deletes the token on *any* `/auth/me` failure, including backend-down, so restarting the backend logs users out.
   - `LoginPage` ignores the `?error=` values the OAuth callback sends.
   - The backend supports anonymous prediction, but the UI requires login to reach the upload screen.
10. **Stale or misleading comments and names:**
    - `main.py:1-3` docstring names the old X-ray checkpoint.
    - The `MIN_CONFIDENCE` comment (`main.py:136-142`) calls it the mismatch detector; it now only drives a "low confidence" warning.
    - `conftest.py` says "4 checkpoints".
    - `ml/train_ct_monai.py` doesn't use MONAI.
    - `eslint-disable` comments remain although the linter is oxlint.
    - `ml/seg_model.py`'s U-Net was reverse-engineered from weights; no training script exists.
11. **Storage and scale:**
    - Full base64 PNG heatmaps are stored per scan in SQLite.
    - `/scans/history` returns everything unpaginated.
    - One lock per model serializes all inference.
    - `URL.createObjectURL` previews are never revoked.
12. **Repo clutter:**
    - 8 overlapping root reports (some contradict each other), root-level PNGs and `test.zip`.
    - Logs, `app.db` and misc `*_fp_test_list.txt` files inside `backend/`.
    - ~20 screenshots in `frontend/`; the stock Vite `frontend/README.md`.
    - ~40 unorganized one-off scripts; `datasets/` duplicates `raw/` + `data/`; experimental checkpoints mixed with served ones.
    - Knowing which files are live requires reading `main.py:86`.
13. **Minor issues:**
    - `@app.on_event` is deprecated.
    - `backend/clear_db.py` uses the relative path `'app.db'`, so it must run from `backend/`.
    - No DB migrations.
    - CORS is hard-coded.
    - `config.py` writes to `.env` as an import side effect.

---

## 12. Glossary

- **Modality:** image type, `xray` or `ct`. Each has its own route, classifier, sanity gate and OOD stats.
- **Sanity gate:** per-modality ResNet18 binary classifier (`scan` / `not_scan`). Trained against photos, documents and screenshots to reject non-medical uploads.
- **Modality gate:** ResNet18 X-ray-vs-CT classifier. Rejects a CT sent to the X-ray route (and vice versa), or any upload it's unsure about.
- **OOD (out-of-distribution) detector:** rejects inputs whose penultimate-layer embedding lies farther from the training-set **centroid** than the 99th-percentile training distance.
- **Reproduction check:** startup re-evaluation of a checkpoint on local val data; a mismatch aborts boot.
- **Bundle (`ModelBundle` / `SegBundle`):** loaded model + class names + device + lock.
- **Grad-CAM / heatmap:** gradient-weighted activation map overlaid on the input to show influential regions.
- **XRV / txrv:** torchxrayvision, a library of chest-X-ray-pretrained models with its own normalization (±1024, single channel).
- **RadImageNet:** medical-image-pretrained weights, tried for CT and not adopted.
- **Lung crop / border crop:** U-Net-mask bounding-box crop of X-rays; `border_crop` is the fixed-margin fallback.
- **Shortcut learning:** the model keying on non-pathology cues such as dataset source, scanner table, framing, text tint or white boxes instead of disease features.
- **Leakage audit:** perceptual-hash + SSIM grouping so near-duplicate CT slices don't straddle train/val/test (`scripts/ct_leakage_audit.py`).
- **Protocol-mismatch exclusion:** 61 CT images from a different acquisition dropped from the CT dataset.
- **TB artifacts:** tint and white-box marks correlated with the Tuberculosis label in the raw TB dataset; cleaned by `scripts/tb_clean.py`.
- **IQ-OTH/NCCD:** the Iraqi lung-cancer CT dataset (Benign/Malignant/Normal). **LIDC-IDRI**, **LUNGx:** public CT nodule datasets used in the abandoned volume experiment.
- **phase35_v2, phase36, phase37:** development-log iteration labels baked into checkpoint names.
- **Condition A / B / C:** the evaluation report's offline test / live-HTTP test / external-data setups.
- **report_source:** `groq` (LLM-written) or `template_fallback`.
- **CT volume pipeline:** removed experimental multi-slice DICOM feature (`/predict/ct/volume`); does not exist in current code.

---

## Open questions

- How was `xray_txrv_densenet121_best.pt`'s `backbone` field set to `densenet121-res224-all` when the training script writes `txrv_…`? Hand-edited, or a different script version?
- Where and how was the lung-segmentation U-Net trained? No script exists in the repo.
- Is there a canonical copy of the checkpoints and `data/` for a fresh machine? They're gitignored and Git is gone; a Drive backup is mentioned elsewhere, but its location is unknown.
- Should Git be re-initialized before the planned GitHub upload, and which of the 8 root reports is canonical? (`PROJECT_CONTEXT.md` appears newest.)
- Is anonymous prediction meant to be exposed in the UI, or is the backend's optional-auth path vestigial?
