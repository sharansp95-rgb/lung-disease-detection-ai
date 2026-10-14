# Project Status

**Generated:** 2026-08-25. Every claim below was verified directly against the running code, live checkpoints, and on-disk data at the time of writing (checkpoint metadata read via `torch.load`, dataset counts via direct directory listing, routes via grep against the actual route decorators, config via the live `/health` endpoint) — not recalled from memory or copied from earlier notes.

**Correction (2026-09-05):** X-ray test accuracy corrected from the previously stated **98.23%** to **97.87%** (0.9786680541103018) — read directly from `eval_outputs/xray_phase35_v2/xray_phase35_v2_test_metrics.json`. The incorrect figure had propagated into the IEEE paper draft and has been corrected there too.

**Update (2026-09-06):** the X-ray disease classifier was swapped to a torchxrayvision-pretrained backbone (see §2 below) after a documented investigation showed it substantially reduces the acquisition-style shortcut described in the paper's Section V-C, at a small, disclosed accuracy cost. The CT classifier was investigated for the same kind of swap (RadImageNet-pretrained backbone) but **not adopted** — two separate training attempts either collapsed the Benign class entirely or, after fixes, collapsed Normal instead, netting worse overall accuracy than the original model in both cases; `checkpoints/ct_best_model_phase35_v2.pt` remains the active CT model. Verified directly: the new X-ray checkpoint's startup reproduction check passes (diff 0.0000), the full `backend/tests/` pytest suite passes (26/26), and a manual before/after Grad-CAM comparison on a real test image confirms the heatmap renders correctly (not just that it avoids crashing).

---

## 1. Overview

A dual-modality medical imaging tool: a deep-learning pipeline that classifies **chest X-rays** (COVID-19 / Normal / Pneumonia / Tuberculosis) and **chest CT scans** (Benign / Malignant / Normal), with Grad-CAM explainability heatmaps showing which image regions drove each prediction. A FastAPI backend serves six model bundles (two disease classifiers, a modality-mismatch gate, a per-modality sanity gate pair that rejects non-scan uploads, and a lung-segmentation model used to improve input preprocessing) behind a React frontend with full user accounts, scan history, an AI chat assistant, and light/dark theming. Explicitly framed throughout as a research/decision-support prototype, not a diagnostic service.

---

## 2. Models

All six are loaded at backend startup (`backend/main.py`) and each undergoes a **reproduction check** before the app is allowed to start: the checkpoint's claimed metric is independently re-computed against the live validation data, and a mismatch beyond tolerance raises and blocks startup rather than silently serving a wrong model. Paths below are the literal values in `backend/main.py`'s `CHECKPOINTS`/`SEG_CHECKPOINT` dict, cross-checked against this session's actual startup log.

### X-ray disease classifier
- **Checkpoint:** `checkpoints/xray_txrv_densenet121_best.pt` (active as of 2026-09-06; the previous `xray_best_model_phase35_v2.pt` is kept in the repo as a fallback, no longer referenced by `backend/main.py`)
- **Architecture:** DenseNet-121, **torchxrayvision-pretrained** (`densenet121-res224-all` — trained on real chest X-rays pooled from NIH/PadChest/CheXpert/MIMIC-CXR/RSNA, not ImageNet natural photos), transfer learning (`ml/train_xray_txrv.py`)
- **Input:** single-channel (not 3-channel), `(2*(x/255)-1)*1024` normalization (torchxrayvision's own scheme, range ≈[-1024,1024] — distinct from both ImageNet mean/std and the CT model's normalization), `XRayCenterCrop`→`XRayResizer(224)`
- **Classes (order):** `COVID-19`, `Normal`, `Pneumonia`, `Tuberculosis`
- **Accuracy:** checkpoint claims val_acc **0.9646**; reproduces to **0.9646** against `data/xray/val` (diff 0.0000) on every startup. Test accuracy **0.9657** (vs. the previous checkpoint's 0.9787 — a disclosed ~1.3-point trade-off, not a regression to fix)
- **Why adopted:** a center-blank occlusion probe (paper Section V-C) showed the previous checkpoint's COVID-19 accuracy fell only 1.7 points (91.9%→90.2%) when the entire central lung field was blanked — near-total reliance on peripheral/acquisition-style cues rather than pathology. The same test on this checkpoint shows a 79.8-point collapse (94.4%→14.6%, below the 25% chance floor) under identical blanking, confirming genuine dependence on lung-region content for that class. Normal/Pneumonia/Tuberculosis confound levels were largely unchanged by the swap (see `eval_outputs/xray_txrv_densenet121/` and the CT-swap investigation below for the full comparison — a parallel RadImageNet-backbone swap was attempted for the CT model and *not* adopted, since it made accuracy worse without cleanly fixing its own shortcut).
- **Serving-path changes required by this swap (verified working, not assumed):** `backend/models.py`'s `load_model_bundle` now branches on the checkpoint's `backbone` field to build the right architecture and run the matching reproduction-check preprocessing; `backend/gradcam_utils.py` gained `preprocess_image_xrv`/`predict_with_heatmap_xrv` (the old ImageNet-based `denormalize` would have silently mis-rendered a single-channel, differently-scaled tensor rather than crash); the OOD detector's centroid/threshold (`backend/ood_stats/xray_ood.pt`) was recomputed against this model's own embedding space via `scripts/compute_ood_thresholds.py`, since the old model's OOD stats are for a completely different, incompatible feature space. The CT model, both gate models, and both sanity gates are all unaffected — they still use the original ImageNet-based pipeline unconditionally.

### CT disease classifier
- **Checkpoint:** `checkpoints/ct_best_model_phase35_v2.pt` — **unchanged** this session
- **Architecture:** DenseNet-121, transfer learning
- **Classes (order):** `Benign`, `Malignant`, `Normal`
- **Accuracy:** checkpoint claims val_acc **0.9904**; reproduces to **0.9904** (diff 0.0000)
- **RadImageNet-backbone swap investigated and rejected (2026-09-06):** the same kind of domain-pretrained-backbone swap that worked for X-ray was attempted here (RadImageNet-pretrained DenseNet-121, MIT-licensed, preprocessing verified against the official repo's source before use — see `ml/train_ct_monai.py`). Two full training attempts failed for different reasons: attempt 1 (standard class-weighted loss, 15 epochs) converged to a degenerate solution that never predicted `Benign` at all (0% recall, confirmed flat on train/val/test alike, not a not-enough-epochs issue); attempt 2 (macro-recall checkpoint selection, 35 epochs, no warmup, gradient clipping, Benign oversampling) fixed `Benign` recall (100%) but only by collapsing `Normal` instead (recall 95%→29%), netting *worse* overall test accuracy (68.9%) than either the first attempt (82.5%) or the original model (~99.0% val). Conclusion: RadImageNet's general (non-lung-specific) pretraining combined with this project's small, imbalanced 829-image CT training set was insufficient to cleanly separate all three classes regardless of imbalance-handling strategy. Full write-up (including the honest negative-result paragraph) is in the paper draft's Results/Discussion section on CT backbone retraining; supporting eval artifacts at `eval_outputs/ct_best_model_phase35_v2/` (old-model occlusion baseline) — the rejected checkpoints themselves were deleted after this investigation concluded, not kept alongside the production one.

### Modality gate
- **Checkpoint:** `checkpoints/modality_gate_best_model.pt`
- **Architecture:** ResNet-18
- **Classes (order):** `ct`, `xray` — not a disease label, decides which classifier route an image actually belongs to
- **Accuracy:** checkpoint claims val_acc **1.0000**; reproduces to **1.0000** (diff 0.0000)
- **Training data:** the X-ray and CT train splits combined and relabeled by modality (15,373 xray + 829 ct images)
- **Purpose:** runs before the disease classifier on every `/predict/*` call; exists because the OOD detector alone was confirmed (during development) to not reliably catch a real CT scan submitted to the X-ray route or vice versa.

### Per-modality sanity gate (X-ray, CT)
- **Checkpoints:** `checkpoints/xray_sanity_best.pt`, `checkpoints/ct_sanity_best.pt`
- **Architecture:** ResNet-18, binary (`scan` / `not_scan`)
- **Purpose:** runs before the modality gate on every `/predict/*` call. The modality gate is a closed-set xray-vs-ct classifier with no "neither" option, so a non-medical upload (a photo, a scanned document, a screenshot) still gets confidently routed into one of its two known classes — verified during development: a random photo scored 92%+ "xray" from the gate and passed the OOD check too, since its feature distance and softmax confidence landed inside the range real training images occupy. This gate closes that gap with a separately trained binary classifier per modality (real scans vs. a pool of photos/documents/screenshots as negatives). Has a dedicated passing regression test (`test_predict_rejects_colorful_non_scan_image` plus the sanity-gate rejection path in `backend/tests/test_predict.py`).

### Lung segmentation (U-Net)
- **Checkpoint:** `checkpoints/xray_seg_model.pt`
- **Architecture:** 4-level U-Net trained from scratch, 1 input channel (grayscale) → 1 output channel (mask logit). No training script for this model exists in this repo — it was trained externally against `seg_data/`, and its architecture was reverse-engineered from the checkpoint's `state_dict` into `ml/seg_model.py` (verified via a strict `load_state_dict` with zero missing/unexpected keys).
- **Not a classifier** — no class labels; outputs a binary lung mask used to crop X-ray uploads before classification.
- **Accuracy:** checkpoint claims val_dice **0.9890** / val_iou **0.9786**; reproduces to **dice 0.9830 / iou 0.9671** against `seg_data/val` (2,116 images) — within this project's tolerance (1pt dice / 2pt IoU) but, unlike the other five models, not an exact diff-0.0000 match. Most likely a minor resize/interpolation difference between the (unavailable) external training pipeline and this repo's inference-time preprocessing assumption (plain [0,1] scaling, no mean/std normalization — the assumption that made the reproduction check pass at all).
- **Training data:** all 4 classes of the COVID-19 Radiography Dataset (COVID, Lung_Opacity, Normal, Viral Pneumonia) — deliberately includes Lung_Opacity even though the disease classifier excludes it, since segmentation doesn't care about diagnostic label.

---

## 3. Known limitations

**🔴 OPEN BUG (2026-09-06, unresolved): the new X-ray checkpoint is reportedly predicting COVID-19 for most/all uploaded images in live manual testing.** This was reported by the user after the torchxrayvision-backbone swap (see §2) was deployed and both servers restarted; it has **not** been independently reproduced, root-caused, or fixed as of this writing — do not treat it as understood or resolved. It contradicts the startup reproduction check (diff 0.0000 against `data/xray/val`) and the passing `pytest` suite's real-sample prediction test, which means whatever is wrong is likely specific to the live `/predict/xray` request path on real user-uploaded images (e.g. an upload-side preprocessing difference, a gate/OOD interaction, or something about the images being tested) rather than the model or reproduction check itself — but this is a hypothesis, not a diagnosis. **Flagged here explicitly so it is not mistaken for a working feature; investigate before relying on X-ray predictions.**

**OOD detection is partial by design.** The out-of-distribution detector (`backend/ood_detector.py`) rejects inputs whose feature-space distance from a training-set centroid exceeds the 99th-percentile training distance — it catches genuinely alien input (screenshots, non-medical images). It does **not**, by itself, catch a real medical image of the wrong modality (that gap is closed by the separate gate model above, not by OOD). It also does **not** catch an in-scope-but-untrained class: a genuine chest X-ray showing Lung Opacity (a real category in the source dataset, excluded from the classifier's 4 trained classes) passes the OOD check fine and gets classified into one of the 4 known classes. Verified via dedicated investigation this session: predictions split ~50/50 between COVID-19 and Normal with varying confidence (75–99.7%), never Pneumonia or Tuberculosis — consistent with genuine model uncertainty on an out-of-scope input, not a bug.

**Segmentation-crop integration causes a small, disclosed Tuberculosis regression.** On a 60-image stratified test sample, switching X-ray preprocessing from a uniform border-crop fallback to the trained segmentation model's lung-boundary crop raised overall accuracy from 88.3% to 95.0%, but Tuberculosis-class accuracy dropped from 93% to 87% (13/15 vs 14/15 correct). Root cause: the TB Chest Radiography source dataset has no segmentation masks, so the classifier's Tuberculosis training images were built exclusively via a uniform 8% border crop — the classifier never saw TB images cropped tightly to the lung boundary. The segmentation crop is anatomically more accurate, but that shifts TB inputs slightly outside the distribution the classifier actually learned. There is no reliable signal at inference time to detect "this looks TB-source-style" and route around it, so this is an accepted, documented trade-off (net +6.7 points overall) rather than an unresolved bug.

**CT dataset is small.** 1,036 images total (829 train / 104 val / 103 test) after a patient-leakage cluster-aware resplit and a protocol-mismatch exclusion (down from an original 1,097 — see §6). The 104-image validation set means single-digit-percentage swings in val accuracy should be read as noise, not signal.

~~Google OAuth is configured but not end-to-end browser-tested.~~ **Update (2026-08-26): confirmed working.** The full round trip — a real user consenting on Google's side and landing back at `/auth/google/callback` — has been exercised with an actual Google account and logs in successfully.

**Groq model deviates from the original spec.** Originally specified as `llama-3.3-70b-versatile`; that model no longer exists on Groq's API (confirmed via the account's model list). Currently uses `openai/gpt-oss-120b` instead (`backend/groq_client.py`).

**Class imbalance is real but was addressed, not ignored.** X-ray: Normal outnumbers Tuberculosis ~22x and Pneumonia ~10x. CT: Malignant outnumbers Benign ~4.2x. Both inherited from source data. Handled via inverse-frequency-weighted `CrossEntropyLoss` at training time (`ml/train.py`, `compute_class_weights`), not by discarding or duplicating data.

---

## 4. Architecture

### Backend (FastAPI, `backend/`)

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Friendly service info (not part of the app UI — avoids a bare 404 when hit directly) |
| `/health` | GET | Status, loaded models, OAuth/email config flags |
| `/predict/xray`, `/predict/ct` | POST | Run the full prediction pipeline (see below); auth optional — logged-in calls save to scan history |
| `/chat` | POST | Chatbot, grounded in the current prediction context if any |
| `/auth/signup`, `/auth/login` | POST | Account creation / session issuance (bcrypt + JWT) |
| `/auth/me` | GET | Current user (protected) |
| `/auth/change-password` | POST | Backend endpoint exists and works; **no frontend UI currently calls it** (removed during a profile-page redesign) |
| `/auth/verify/{token}` | GET | Email verification link target |
| `/auth/forgot-password`, `/auth/reset-password/{token}` | POST | Password reset flow; rejects reusing the current password |
| `/auth/google/login`, `/auth/google/callback` | GET | Google OAuth redirect flow |
| `/scans/history` | GET | Authenticated user's saved scans |

**Prediction pipeline** (`_run_prediction` in `main.py`), per call to `/predict/xray` or `/predict/ct`:
1. Read and validate the uploaded image
2. **X-ray only:** segmentation model generates a lung mask → crop to that bbox (falls back to a uniform border-crop if the mask is empty or covers an implausible fraction of the frame)
3. Resize/normalize
4. Modality gate — reject (422) if the image looks like the wrong modality for the route
5. OOD check — reject (422) if feature-space distance from the training centroid exceeds threshold
6. Classifier + Grad-CAM (each model's own `threading.Lock` held across this whole sequence — a real concurrency bug here, forward-hook corruption from overlapping requests, was found and fixed earlier in development)
7. Report generation (Groq, falls back to a template)
8. If authenticated, save the scan to history
9. Return prediction, confidence, full class probabilities, heatmap, report

### Frontend (React + React Router, `frontend/src/`)

| Route | Access | Page |
|---|---|---|
| `/` | public shows landing page; logged-in shows the upload page | `LandingPage.jsx` / `MainApp.jsx` |
| `/login`, `/signup`, `/forgot-password`, `/reset-password/:token`, `/oauth-callback` | public | respective auth pages |
| `/analyzing` | protected | loading screen (real request + simulated step checklist) |
| `/results` | protected | full-width diagnosis + heatmap/explanation panels |
| `/history` | protected | standalone scan history page |
| `/profile` | protected | account details + scan history (current profile page design) |

Flow: upload on `MainApp` → navigate to `/analyzing` (fires the real API call in the background while showing progress) → on success, navigate to `/results`; on failure, an error state with retry, not a silent hang.

Key shared components: `AppNavbar.jsx` (persistent top nav — theme toggle + profile avatar), `ChatWidget.jsx` (floating chat with FAQ quick-reply chips), `ModalitySelector.jsx`, `UploadArea.jsx`, `ResultsDisplay.jsx`, `ThemeToggle.jsx`.

State: `AuthContext.jsx` (token/user, localStorage-persisted) and `ThemeContext.jsx` (dark/light, localStorage-persisted) as React Context providers wrapping the whole app.

---

## 5. Features built

**Working and verified this session or earlier:**
- Full authentication: signup/login (bcrypt password hashing + JWT sessions), email verification and password reset over real Gmail SMTP (HTML + plaintext multipart emails, falls back to console-logging the link if SMTP isn't configured), reset rejects reusing the current password, Google OAuth implemented server-side (see limitation above)
- Scan history: every authenticated prediction is saved (modality, class, confidence, full probability breakdown, heatmap, report); anonymous users can still get predictions without anything being saved
- Chat assistant: floating widget, FAQ quick-reply chips, Groq-backed with a template fallback, explicitly scoped to explaining the tool's own output (not medical advice)
- Light/dark theme toggle, persisted, applied consistently across every page
- Grad-CAM heatmaps with a heatmap/original image toggle
- Plain-language report generation (Groq + template fallback), both carrying the same non-diagnostic disclaimer
- Modality-mismatch gate + OOD rejection + segmentation-guided crop, layered ahead of classification

**Discussed/scaffolded but not fully wired up:**
- Change-password: backend endpoint fully implemented and tested; no frontend UI by design, since `forgot-password` already covers the same need

---

## 6. Data

Full, detailed provenance and audit trail lives in **`data/README.md`** — this section is a verified-current summary, not a replacement for it.

### X-ray (`data/xray/`) — verified current counts, 19,217 total
| split | COVID-19 | Normal | Pneumonia | Tuberculosis |
|---|---|---|---|---|
| train | 2,856 | 10,949 | 1,070 | 498 |
| val | 357 | 1,369 | 134 | 62 |
| test | 357 | 1,368 | 134 | 63 |

Sources: Kaggle `tawsifurrahman/covid19-radiography-database` (COVID-19, Normal, Pneumonia) + `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` (Normal, Tuberculosis), deduped by SHA-256 content hash, merged where both datasets contribute to `Normal`.

**TB artifact cleanup (applied, verified):** a full-dataset scan found 53.1% of Tuberculosis images (vs 0.37% of Normal) carried a color-tint and/or redaction-box artifact correlated with the label — a shortcut-learning risk. Tinted images were desaturated, small redaction boxes inpainted, large ones (≥5% of frame) excluded outright. Applied to both classes for consistency.

**Crop mismatch (found, fixed twice):** the model was trained on cropped input (mask-guided lung-bbox crop for COVID-source classes, uniform 8% border crop for TB-source classes) but live `/predict/xray` uploads were originally served raw/uncropped — confirmed via a direct raw-vs-cropped test to flip real diagnoses (a genuine COVID-19 case misclassified as Tuberculosis at 98.6% confidence). Fixed first with a universal border-crop, then upgraded to the trained segmentation model's lung-boundary crop (see §2/§3 for the residual TB trade-off).

### CT (`data/ct_scan/`) — verified current counts, 1,036 total
| split | Benign | Malignant | Normal |
|---|---|---|---|
| train | 96 | 400 | 333 |
| val | 12 | 50 | 42 |
| test | 12 | 50 | 41 |

Source: Kaggle `hamdallak/the-iqothnccd-lung-cancer-dataset` (IQ-OTH/NCCD, 110 patient cases). No patient-ID metadata survives in this download (filenames are just per-class sequential slice indices), so patient grouping had to be inferred.

**Patient-leakage fix (applied):** perceptual-hash + SSIM near-duplicate detection found 111 clusters (adjacent scan slices) with 47 of them split across train/val/test under the original per-image split — 35 of 109 test images had a near-duplicate sibling in train, enough on its own to explain the model's original 100% test accuracy. Resplit to keep every cluster intact within one split; zero cross-split clusters remain, verified directly.

**Protocol-mismatch exclusion (applied):** after the leakage fix, Grad-CAM still showed shortcut reliance on the scanner table structure, concentrated in Malignant (89.3% table-visible vs 99.8%/100% for Normal/Benign). Cross-checked against image dimensions: the same ~60 Malignant images were both non-standard resolution and table-absent — evidence of a genuinely different acquisition protocol for that subset, not a cropping artifact. Excluded (61 images total) rather than cropped, since cropping would have treated a symptom, not the cause.

### Segmentation data (`seg_data/`) — verified current counts, 21,165 total
| split | covid | lung_opacity | normal | viral_pneumonia |
|---|---|---|---|---|
| train | 2,893 | 4,810 | 8,154 | 1,076 |
| val | 362 | 601 | 1,019 | 134 |
| test | 361 | 601 | 1,019 | 135 |

Built this session from the COVID-19 Radiography Dataset's real per-image lung masks (all 4 classes, 100% mask coverage, 0 skipped/broken pairs), resized to 256×256, 80/10/10 stratified split.

---

## 7. File/folder structure

```
sdp/
├── PROJECT_STATUS.md       this file
├── requirements.txt        Python deps (generated this session — none existed before)
├── backend/                FastAPI app
│   ├── main.py              routes, prediction pipeline, startup/reproduction checks
│   ├── auth_routes.py       /auth/* endpoints
│   ├── scans_routes.py      /scans/* endpoints + save_scan()
│   ├── models.py            classifier bundle loading + reproduction check
│   ├── seg_utils.py         segmentation model loading + lung-crop logic
│   ├── ood_detector.py, gradcam_utils.py, report.py, chat.py, groq_client.py
│   ├── auth_utils.py, email_utils.py, config.py, db.py
│   ├── ood_stats/            centroid+threshold .pt per modality (xray_ood.pt recomputed 2026-09-06 for the new X-ray backbone's embedding space)
│   ├── .env                 real secrets (Groq/JWT/Google/Gmail) — gitignored
│   └── app.db                SQLite (users, tokens, scans) — wiped clean 2026-09-06 at user request, recreated empty on next startup
├── frontend/                React + Vite app (src/pages, src/components — see §4)
├── ml/                      model architectures + training/eval scripts
│   ├── model.py (classifiers), seg_model.py (U-Net), dataset.py, train.py, evaluate.py, gradcam.py
│   ├── train_xray_txrv.py    torchxrayvision-backbone X-ray training (active model's source of truth)
│   └── train_ct_monai.py     RadImageNet-backbone CT training (investigated, not adopted — see §2)
├── scripts/                 one-off data-prep and audit scripts (prepare_*.py, *_audit.py, *_clean.py), plus:
│   ├── compute_ood_thresholds.py           regenerates backend/ood_stats/*.pt; now backbone-aware (branches per checkpoint's architecture)
│   ├── test_new_xray_checkpoint.py         reusable Table-IV + center-blank-occlusion eval for any X-ray checkpoint
│   ├── test_new_ct_checkpoint.py           same, for any RadImageNet-preprocessed CT checkpoint (kept despite that path being rejected — reusable if revisited)
│   └── test_old_ct_checkpoint_occlusion.py occlusion baseline for the active (ImageNet-backbone) CT model, cited in the paper's CT-rejection writeup
├── checkpoints/             8 .pt files: 6 actively loaded by backend/main.py (xray_txrv_densenet121_best.pt, ct_best_model_phase35_v2.pt, modality_gate_best_model_v2.pt, xray_sanity_best.pt, ct_sanity_best.pt, xray_seg_model.pt) + 2 unreferenced-but-intentionally-kept (xray_best_model_phase35_v2.pt, the pre-swap X-ray fallback; modality_gate_best_model.pt, an older pre-existing gate version, predates this session). Rejected CT RadImageNet checkpoints from this session's investigation were deleted, not kept alongside.
├── data/                    xray/, ct_scan/ (processed, model-ready) + README.md (full provenance) + backup zips
├── seg_data/ , seg_data.zip verified-current segmentation train/val/test data + backup
├── raw/                     original Kaggle downloads, pre-processing (reproducible source of truth for data/ and seg_data/) — a Shenzhen+Montgomery X-ray external-source merge was trialed here this session and fully reverted (see data/README.md); nothing from that attempt remains
├── audit/, ml_logs/, eval_outputs/   data-quality audit outputs and training logs from earlier phases (this session's rejected-CT-attempt artifacts removed; active-model and paper-cited eval outputs kept)
└── .gitignore               excludes .env, *.db, node_modules, checkpoints/*.pt, data/, raw/
```

---

## 8. What's NOT done

- **No production deployment, by design.** This project is scoped to local-only use — CORS is hardcoded to `localhost:5173`/`127.0.0.1:5173`; no hosting, CI/CD, process manager, or production secrets handling beyond a local `.env` file.
- **Change-password** has no frontend entry point despite a working backend endpoint (§5). Intentional — `forgot-password` already covers the same user need, so this endpoint is left unused rather than duplicated in the UI.
- **Email verification is not enforced.** Signup sends a real verification link and sets `email_verified` on click, but nothing currently checks that flag — a new account gets a valid session token in the same response as signup, before the link is ever clicked. The flow is real and working end-to-end, just informational rather than access-gating at this time.
- **Segmentation model has no training script in this repo** — it was trained externally; only its architecture (reverse-engineered) and inference-time usage live here. If it ever needs retraining, that pipeline would need to be rebuilt.

**Corrections to this section as of 2026-08-26** (the rest of this file was last verified 2026-08-25 and has drifted in a few places since):
- A real `pytest` suite exists at `backend/tests/` (`conftest.py`, `test_auth.py`, `test_health.py`, `test_predict.py`) and passes.
- Per-IP rate limiting is implemented on prediction and auth routes via `slowapi` (`backend/rate_limit.py`) — in-memory, which is intentional and sufficient since this project only ever runs single-process locally, not a gap.
- **Google OAuth has been exercised end-to-end** with a real Google account and confirmed working — no longer an open item.

**Corrections to this section as of 2026-09-06:**
- **X-ray backbone swap is done, but the resulting live-prediction bug (§3) is NOT done.** The checkpoint swap, serving-path changes, reproduction check, and pytest suite all pass — but a reported live issue (predicting COVID-19 for most/all uploads) is open and unresolved. Do not treat the X-ray swap as fully finished/production-ready until this is investigated.
- **CT RadImageNet backbone swap was attempted and explicitly not adopted** (§2) — this is a completed, closed investigation with a documented negative result, not an open item, but worth noting here so it isn't mistaken for unstarted work.
- **The local user database was intentionally wiped** (`backend/app.db` deleted, recreated empty) at user request during this session's manual testing — not a bug; mentioned here so a reader doesn't wonder why account/scan-history data present in an earlier session no longer exists.
