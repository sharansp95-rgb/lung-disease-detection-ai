# Phase H: deploying the CT-volume pipeline into the live application

**Scope: the FIRST phase of this investigation that writes to the real application.**
`backend/main.py` gains two lines (an import and an `include_router`), two new backend modules
are added, one new frontend page and one new route are added, and `frontend/src/{api.js, App.jsx,
pages/MainApp.jsx, App.css}` gain **additive** blocks. Everything else is untouched.

**`checkpoints/ct_best_model_phase35_v2.pt` was not touched. `/predict/ct`, `/predict/xray`,
`/chat`, `/auth/*`, `/scans/*` were not modified and their behaviour was re-tested over real HTTP
after the change.** All five protected checkpoints and `eval_run/phaseE/pipeline.py` re-hashed
after the work — every digest matches (§1.4, §8). Backend was started on an isolated scratch DB,
exercised, and **stopped; zero `python.exe` remain and nothing listens on the test port.**

---

## Verdict up front

1. **Gemini's Phase G.2/G.3 work verifies clean, independently re-checked here.** The
   `v4_stack3` checkpoint digest matches `phaseG3_verdict.md` §7 exactly; `backend/main.py` and
   the live CT checkpoint match every phase from D onward; all five claimed G.3 artifacts exist
   and their contents match the prose numbers to 4 decimal places. §1.
2. **The "3D UNet" in `phaseG3_verdict.md` is a write-up mislabel, not an integrity problem —
   confirmed directly.** `run_g3_validation.py` never constructs a detector at all: it reuses
   Phase E/E.1's *stored* MONAI RetinaNet candidates. §1.3.
3. **New route `POST /predict/ct/volume` is live and additive**, with async job submission +
   polling. Backend: `backend/ct_volume_routes.py`, `backend/ct_volume_pipeline.py`. §2, §3.
4. **The deployed route reproduces Phase G.3's validated numbers BIT-EXACTLY.** Four real
   volumes were pushed through the live HTTP endpoint; on all **16 candidates** that Phase G.3
   also recorded for those patients, the detector score, the predicted class and all three class
   probabilities agree to **max absolute difference 0.00000**. The thing deployed is the thing
   validated. §5.2.
5. **Existing functionality is unchanged**: `26 passed` pytest (identical to the X-ray crop-fix
   phase's 26), and 6/6 real images through `/predict/xray` and `/predict/ct` return 200 with
   correct classes, heatmaps and reports; the wrong-modality guardrail still fires. §6.
6. **False-positive aggregation is redesigned**, replacing Phase E's "any Malignant candidate
   flags the scan, highest P(Malignant) wins". Two-stage eligibility floors, a graded concern
   level instead of a binary call, and every candidate always returned. §4 — including the case
   this new rule gets *wrong*, which is stated rather than hidden.
7. **Guardrail parity with `/predict/ct` does NOT exist and the UI says so.** Six structural
   checks are implemented; four gaps (no volume OOD, no scan-sanity model, no body-region check,
   Modality tag trusted) are explicitly deferred and are printed in the API response itself. §3.3.
8. **63.1% end-to-end / [52.4%, 72.6%] and the 84.5% detector recall ceiling are rendered on
   screen**, in a banner above the upload control and again beneath every result. §7.

---

## 1. Mandatory verification of Phase G.2/G.3 (run by Gemini in Antigravity, not by Claude Code)

Re-done from scratch this session. **Every check passes.**

### 1.1 The `v4_stack3` checkpoint

| | |
|---|---|
| measured sha256 | `5ee4d8ec9f8328c1ac3adf574844e1b08e3a3f6fd078fde8aae8585bf9eee546` |
| `phaseG3_verdict.md` §7 claims | `5ee4d8ec9f8328c1ac3adf574844e1b08e3a3f6fd078fde8aae8585bf9eee546` |
| **result** | **MATCH (full 64-hex digest, not a prefix)** |
| size / mtime | 28,459,122 B / 2026-09-10 04:42:11 UTC |

### 1.2 The protected live files

| file | measured (first 16) | reference | source of reference | result |
|---|---|---|---|---|
| `backend/main.py` (**before** this phase's edit) | `780120553178c0a5` | `780120553178c0a5` | Phase G.1 §7, G.3 §7, F §6, E.1 §6, E §7 | **MATCH** |
| `checkpoints/ct_best_model_phase35_v2.pt` | `93b3a979884edbc4` | `93b3a979884edbc4` | same chain, back to Phase D §9 | **MATCH** |
| `ct_lidc_nodulecrop_v1.pt` | `bdcbd6b2e5312435` | `bdcbd6b2e5312435` | G.1 §7 | **MATCH** |
| `ct_lidc_nodulecrop_v2_jitter.pt` | `5d60055e02974e91` | `5d60055e02974e91` | G.1 §7 | **MATCH** |
| `ct_lidc_nodulecrop_v3_jitter_loosebenign.pt` | `9d1e9d7bf1ff5e16` | `9d1e9d7bf1ff5e16` | G.1 §7 | **MATCH** |
| `eval_run/phaseE/pipeline.py` | `936427fc20d389a0` | `936427fc20d389a0` | G.1 §7 / F §6 | **MATCH** |

`ct_best_model_phase35_v2.pt` also still carries mtime `2026-08-24 07:15:30 UTC`, unchanged.

### 1.3 The G.3 artifacts — present, and their CONTENT checked, not just their filenames

All five files claimed by `phaseG3_verdict.md` exist in `eval_run/phaseG3/_work/`:
`g3_combined.json` (2,907 B), `e2e_v4_stack3.json` (83,954 B), `stability_sweep_v4.json`
(293,000 B), `channel_ablation.json` (1,195 B), `test_accuracy.json` (1,403 B).

Two were opened and read against the prose:

* **`channel_ablation.json`** — report says accuracy 0.8946 / 0.8303 / 0.4370 / 0.6992 for
  normal / centre-only / neighbours-zeroed / neighbours-noise, flip rates 13.6% / 55.8% / 30.3%,
  and per-channel gradients 0.023058 / 0.028552 / 0.016634 (0.81× / 0.58×). The JSON carries
  **exactly those values**, including `0.8076` and `0.5826` for the relative-to-centre ratios.
* **`g3_combined.json`** — end-to-end `@0.5` = 53 correct, accuracy 0.631, Wilson
  `[0.5242, 0.7263]`, detected 71; GT-centred 67 / 0.7976 / `[0.6996, 0.8696]`; scan-level
  24/45 = 0.5333 with `benign_overcalled_malignant: 10`; through-plane 2.0 mm accuracy 0.6429 /
  flip 0.2976. **Every one matches the report's prose.**
* `e2e_v4_stack3.json` was additionally used as the ground truth for §5.2's bit-exactness check —
  84 `per_nodule`, 153 `unmatched`, 45 `per_case` rows, consistent with the report's counts.

**The exact detector-recall figure was verified rather than taken from the task prompt's
paraphrase**: `phaseG3_verdict.md` §4 and §6 both state **71/84 = 84.5%** detected at ≤8 mm, and
§6 names 84.5% as the maximum achievable accuracy given this detector. The prompt's "~84.5%" is
exact.

### 1.4 The "3D UNet" discrepancy — resolved: a write-up mislabel, nothing more

`phaseG3_verdict.md` describes the detector as a "3D UNet" twice (§4 verdict item 4 and the §4
heading text). Every phase from D onward used MONAI's RetinaNet-based `lung_nodule_ct_detection`
bundle. **Checked directly rather than assumed:**

* `eval_run/phaseG3/run_g3_validation.py` contains **zero** occurrences of `UNet`, `unet`,
  `RetinaNet`, or any detector-construction code. It builds only the classifier
  (`build_model(ck["backbone"], ...)` from `v4_stack3`).
* Its `end_to_end_pipeline()` docstring reads *"The detector's stored candidates are reused (same
  as Phase F)"*, and the code loads `phaseE/_work/e2e_results.json`,
  `phaseE1/_work/e2e_results_e1.json`, `phaseE/_work/scan_reports.json` and
  `phaseE1/_work/scan_reports_e1.json`, then iterates `scan_rep["candidates"]` reading
  `cd["detector_score"]` and `cd["center_ras_mm"]`.
* Consequence, confirmed numerically in the report itself: G.3's detected count is **71/84,
  identical to Phase E.1's**, and its unmatched pool is **119 (2.64/scan), "Identical detector
  pool"** — which is only possible if no detector was re-run.

**Verdict: the prose is wrong, the experiment is not.** The detector behind every Phase D–G.3
number, and behind this deployment, is the MONAI 3D RetinaNet. No correction was made to
`phaseG3_verdict.md` (it is Gemini's artifact and this phase does not edit prior phases'
write-ups); the correction is recorded here and in `RESUME_STATE.md`.

**Nothing failed verification, so the build proceeded.**

---

## 2. What was built — file locations

### Backend (new)

| path | lines | what |
|---|---|---|
| **`backend/ct_volume_pipeline.py`** | ~340 | The pipeline wrapper: lazy import of `eval_run/phaseE/pipeline.py`, the 3-slice stack builder, the `v4_stack3` classifier, and `aggregate_scan()` (§4). |
| **`backend/ct_volume_routes.py`** | ~380 | The `APIRouter`: upload handling, input validation, the in-memory job store, one daemon thread per job, and the three routes. |

### Backend (modified — 2 additive edits, nothing removed)

`backend/main.py`: `from ct_volume_routes import router as ct_volume_router` next to the other
router imports, and `app.include_router(ct_volume_router)` next to the existing two, with a
comment. **`_run_prediction`, `predict_xray`, `predict_ct`, the startup handler, the guardrail
chain and every constant are byte-identical.**

### Routes added

| method | path | behaviour |
|---|---|---|
| `POST` | `/predict/ct/volume` | multipart `files`; validates; **202** + `job_id` in ~0.5–1.1 s |
| `GET` | `/predict/ct/volume/{job_id}` | `queued` / `running` (+`progress`, `stage`) / `succeeded` (+`result`) / `failed` (+`error`); **404** on unknown id |
| `GET` | `/predict/ct/volume` | endpoint self-description: accuracy figures, aggregation thresholds, whether models are resident |

### Frontend

| path | status | what |
|---|---|---|
| **`frontend/src/pages/CtVolumePage.jsx`** | **new** | the entire experimental flow: banner, file picker, submit, poll loop, results |
| `frontend/src/api.js` | additive | `submitCtVolume()`, `pollCtVolumeJob()`, `CT_VOLUME_ACCURACY`; `predict()`/`checkHealth()`/`sendChatMessage()`/`getScanHistory()` untouched |
| `frontend/src/App.jsx` | additive | one `<Route path="/ct-volume">` behind `RequireAuth` |
| `frontend/src/pages/MainApp.jsx` | additive | a `<Link>` card below the existing upload panel. **The single-image upload UI and its handlers are unchanged** — the link navigates away, it does not alter the CT flow. |
| `frontend/src/App.css` | additive | one appended block, every selector prefixed `ctvol-`, none of which existed before |

`npx vite build` → **✓ built in 260ms, 47 modules, no warnings.**

---

## 3. Design decisions

### 3.1 Upload shape: multipart `files` — one `.zip` OR many loose `.dcm`

A CT series is 120–800 files and 33–402 MB (Phase D §6). Both shapes are accepted by the same
route: a single `.zip` is flattened (members reduced to basenames, so an archive cannot write
outside the temp directory), and a multi-file post is written straight through — which is what a
browser `<input multiple>` produces, so the frontend does no client-side zipping. Both converge
on one flat temp directory of `.dcm` files, which is exactly what Phase D's
`volume_io.read_series` consumes. **Both were tested over real HTTP (§5.1).** The temp directory
is deleted by the worker on success *and* on failure.

### 3.2 Async job pattern — reused from `eval_run/phaseE/app.py`, not reinvented

In-memory `JOBS` dict + `threading.Lock` + one daemon thread per job, `MAX_CONCURRENT = 1`,
progress reported through the same `progress(msg, frac)` callback `TwoStagePipeline.run()`
already exposes. Deliberately **not** Celery/Redis (`PROJECT_STATUS.md` §8: local-only scope).

Two things were added beyond Phase E's scratch version:
* **The queue actually queues.** Phase E's `_submit` counted running jobs but started the thread
  regardless. Here the worker blocks in a 1 s poll until fewer than `MAX_CONCURRENT` jobs are
  running, reporting `queued behind N running job(s)` while it waits — which matters because
  Phase E.1 measured host RSS as the real ceiling, not VRAM.
* **The dict is bounded.** Finished jobs older than 6 h are dropped, and the store is trimmed to
  50 entries. Costs still written down rather than glossed: **jobs die with the process, there is
  no retry, and there is no cross-worker sharing.**

**Models are loaded lazily on the first job**, never at startup. A backend that never receives a
volume upload pays nothing — no MONAI import, no 80 MB bundle, no extra VRAM — which is why the
existing test suite's startup time is unchanged.

### 3.3 Input validation — what IS checked, and what is NOT

The single-image routes run four models in a chain (sanity gate → modality gate → feature-space
OOD → classifier). **None of it applies here**: all four take one 2D PIL image, and a 300-slice
DICOM series is not one. No volume-level equivalent has ever been trained in this project.

**Implemented (structural; no model required), all six confirmed firing over real HTTP in §5.3:**

| check | failure code |
|---|---|
| total upload ≤ 800 MB | 413 |
| every file parses as DICOM **and** carries `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing` | 400 |
| `Modality` tag is `CT` | 422 |
| exactly one `SeriesInstanceUID` | 422 |
| consistent in-plane shape and pixel spacing | 422 |
| 20 ≤ slice count ≤ 1200 | 400 |
| slice spacing regular (max deviation from median ≤ max(50%, 1 mm)) | 422 |
| zip is readable; member paths reduced to basenames | 400 |

The spacing-regularity check is not decorative: `read_series` takes the **median** step, so a
series with duplicated or missing slices would silently yield a geometrically wrong volume *and*
a wrong z-neighbour for every one of the 3-slice stacks the classifier depends on.

**Explicitly deferred — this endpoint does NOT have parity with `/predict/ct`:**

* **no volume-level OOD detector.** Nothing measures whether an uploaded series resembles the
  LIDC-IDRI data these models were trained on.
* **no scan-vs-not-a-scan sanity model** for volumes.
* **no body-region check.** A head or abdomen CT carries `Modality=CT`, passes every check above,
  and **will be processed and reported on as if it were a chest CT.**
* **the `Modality` tag is header metadata and is trusted as-is** — it is not derived from pixels
  and can be wrong.

These four are returned to the client in every submission response under
`validation.deferred`, and the last two are surfaced in the UI's limitations block. This is
Phase D §6 / Phase E §7's "the guardrail chain was not built and is absent", still true, now at
least *named at the point of use*.

---

## 4. The aggregation rule — the actual design decision

**What it replaces.** Phase E's rule was: any candidate the classifier calls Malignant makes the
scan Malignant; the driver is whichever has the highest P(Malignant); there is no floor of any
kind. Phase E called it "visibly wrong as specified" at 5/10 scan accuracy; Phase E.1 §3 measured
it as the dominant failure mode (**19 of 45 benign scans over-called**); Phase G.3's better
classifier cut that to 10 of 45 but **the rule itself was never changed**.

**The new rule, in full.**

```
report floor      det >= 0.30   -> candidate is RETURNED
decision floor    det >= 0.50  AND  top-class prob >= 0.60   -> candidate COUNTS
strong candidate  det >= 0.80  AND  top-class prob >= 0.90

concern = high      if  >=2 counting Malignant candidates
                    or  exactly 1 counting Malignant AND it is strong
        = moderate  if  exactly 1 counting Malignant, not strong
        = low       if  0 counting Malignant AND >=1 counting Benign
        = none      if  all counting candidates are Normal, or nothing counts
```

**Why each piece:**

* **A detector decision floor of 0.50** because Phase D measured 20/22 recall there and found the
  centre-error distribution **bimodal** — a nodule is either placed within ~5 mm or not found at
  all — and because every headline number in Phases D/E/E.1/G.3 is quoted at 0.5. Candidates in
  `[0.30, 0.50)` are still shown, flagged, with the reason printed.
* **A classifier confidence floor of 0.60** because the unmatched-candidate pool is where the
  over-calls live (G.3: 17 of 119 unmatched graded Malignant), and its grades are on average less
  confident than true-nodule grades. **This did fire on real data**: LIDC-IDRI-0059's top
  candidate (det 0.989) graded Benign at only 0.556 and was excluded from the summary while
  remaining visible in the list (§5.1).
* **Corroboration for `high`** because at 63.1% end-to-end accuracy a single candidate is not a
  verdict. One Malignant candidate that is only moderately confident yields `moderate` with the
  text *"One candidate is not corroborated and this pipeline over-calls Malignant on non-nodule
  structures (10 of 45 benign scans in the Phase G.3 test set)"* — i.e. the known failure rate is
  attached to the specific claim being made.
* **The candidate list is the primary output.** `concern` is a summary *of* the list, never a
  replacement for it. Every candidate is returned with `candidate_id`, `detector_score`,
  `center_ras_mm`, `box_size_mm`, slice index, all three class probabilities, `pred`,
  `confidence`, `counts_toward_summary` and `excluded_because`.

**What this rule does NOT fix, stated plainly.**

1. **A confidently-wrong spurious box still drives the call.** Phase E's out-of-lung LIDC-0321
   box scored det 0.715 and Malignant at 0.984 — it clears both floors. No threshold available
   here catches it; an in-lung gate would, and was not built. This is why the list is never
   hidden behind the summary.
2. **The floors cut both ways, and this was observed on real data.** LIDC-IDRI-0059's ground
   truth is a **Malignant** nodule. The pipeline detected it (det 0.989) but graded it Benign at
   0.556 — so the confidence floor excluded the one candidate that mattered and the scan reported
   `low concern`. The floor did not cause the miss (the grade was already wrong), but it did
   remove the finding from the count. **The candidate is still displayed**, which is the entire
   reason the design refuses to collapse to a single number.
3. `high` / `moderate` / `low` / `none` are **not calibrated** against outcome data. They are a
   presentation of the counts, not a risk score.

---

## 5. End-to-end testing on the ACTUAL live endpoint

Backend started exactly as prior phases did:

```
cd backend
DB_PATH=<scratchpad>/phaseH.db python -m uvicorn main:app --host 127.0.0.1 --port 8089
```

Startup ran the **real** reproduction checks — all six pass, unchanged: xray 0.9646, ct 0.9904,
gate 1.0000, xray_sanity 0.9996, ct_sanity 1.0000, seg loaded. Log:
`<scratchpad>/backend_phaseH.log`. Every request below is a real HTTP call from a separate
`requests` process — **no in-process `TestClient`, no direct call to a pipeline function.**

### 5.1 Four real volumes through the live route

Reused from disk, nothing re-downloaded (`eval_run/phaseD/_work/volumes/`,
`eval_run/phaseE1/_work/volumes/`).

| case | upload shape | slices | POST→202 | polls | total | read / detect / grade | GPU peak | host RSS peak | concern |
|---|---|---|---|---|---|---|---|---|---|
| LIDC-IDRI-0586 | **.zip** (33.4 MB) | 120 | 0.68 s | 12 | **19.6 s** | 4.1 / 15.2 / 0.2 | 0.94 GB | 4.30 GB | low |
| LIDC-IDRI-0059 | 133 loose `.dcm` | 133 | 0.48 s | 22 | **42.8 s** | 9.6 / 33.0 / 0.2 | 0.97 GB | 5.88 GB | low |
| LIDC-IDRI-0433 | 305 loose `.dcm` | 305 | 1.10 s | 20 | **38.2 s** | 11.4 / 26.6 / 0.2 | 0.97 GB | 6.36 GB | low |
| LIDC-IDRI-0008 | 133 loose `.dcm` (Phase E.1 set) | 133 | 0.55 s | 21 | **41.6 s** | 8.7 / 32.7 / 0.1 | 0.97 GB | 7.87 GB | **high** |

**Mean 35.6 s/scan** — Phase D predicted "20–50 s, ~35 s typical" and Phase E measured 33.4 s.
**Stage 2 is still free** (0.1–0.2 s, ≤0.5% of wall time) even though it now builds three crops
per candidate instead of one. **Polling behaviour observed as designed**: `progress` moves
0.0 → 0.05 (`reading DICOM series`) → 0.15 (`stage 1: detecting nodule candidates in a
[512, 512, 265] volume`) → 0.75 (`stage 2: grading 5 candidate(s)`) → 1.0 (`done`), and the
`status` transitions `queued → running → succeeded`.

GPU peak is 0.94–0.97 GB rather than Phase E's 0.751 GB **because the live backend also has the
five single-image models resident** — expected, and still comfortably inside the 4 GB card.

Example of the aggregation firing (LIDC-IDRI-0059, 3 candidates returned, 1 counted):

```
c00 det=0.989 Benign conf=0.556  slice=102  counts=False (classifier confidence below 0.60)
c01 det=0.805 Benign conf=0.895  slice=65   counts=True
c02 det=0.345 Benign conf=0.993  slice=103  counts=False (detector score below 0.50)
```

### 5.2 The deployed route reproduces Phase G.3 BIT-EXACTLY

The strongest available check that the deployment is the validated pipeline and not a
subtly-different one. Four of the test patients are in Phase G.3's own 45, so its stored
per-candidate results are ground truth.

| | result |
|---|---|
| candidates comparable (8 unmatched + 8 GT-matched) | **16** |
| identical detector score | **16/16**, max abs diff **0.00000** |
| identical predicted class | **16/16** |
| identical all three class probabilities | **16/16**, max abs diff **0.00000** |

This covers the whole chain — DICOM read, RAS affine, `ras_to_series_rc`, the integer crop box,
HU windowing, the 3-slice stack assembly, the `Grayscale`-free transform chain, and the
`v4_stack3` forward pass — through a real HTTP request. Notably it includes LIDC-IDRI-0059's
Malignant nodule being graded Benign, which G.3 also recorded: **the deployment reproduces the
errors too, which is what a faithful reproduction looks like.**

### 5.3 Input validation, over real HTTP

| test | expected | got |
|---|---|---|
| 10 DICOM slices | reject | **400** "Only 10 DICOM slice(s) were readable… needs at least 20… for a single image, use the standard CT tool instead" |
| 30 copies of a real chest X-ray PNG renamed `.dcm` | reject | **400** "30 of 30 file(s) are not readable DICOM slices, or are missing the position/orientation/pixel-spacing tags…" |
| 40 slices of LIDC-0586 + 40 of LIDC-0059 | reject | **422** "The upload contains 2 different DICOM series." |
| corrupt `.zip` | reject | **400** "That file is not a readable .zip archive." |
| a `.txt` file | reject | **400** (no DICOM slices) |
| `GET /predict/ct/volume/deadbeefcafe` | 404 | **404** "Unknown job id (jobs do not survive a server restart)." |

---

## 6. Existing functionality is unchanged

### 6.1 pytest

```
cd backend && python -m pytest -q
26 passed, 3 warnings in 65.49s
```

**26 — identical to the count recorded in `eval_run/xray_cropfix_applied.md` §3.6.** The three
warnings are the pre-existing `on_event is deprecated` ones, not new.

### 6.2 Regression spot-check on the old endpoints, over real HTTP, after the change

| route | file | truth | HTTP | predicted | conf | heatmap | classes | report |
|---|---|---|---|---|---|---|---|---|
| `/predict/xray` | `covid19db_COVID-100.png` | COVID-19 | 200 | **COVID-19** | 0.998 | ✔ | 4 correct | groq |
| `/predict/xray` | `covid19db_Normal-3470.png` | Normal | 200 | **Normal** | 0.660 | ✔ | 4 correct | groq |
| `/predict/xray` | `tbdb_Tuberculosis-93.png` | Tuberculosis | 200 | **Tuberculosis** | 0.972 | ✔ | 4 correct | groq |
| `/predict/ct` | `Bengin case (1).jpg` | Benign | 200 | **Benign** | 0.929 | ✔ | 3 correct | groq |
| `/predict/ct` | `Malignant case (382).jpg` | Malignant | 200 | **Malignant** | 0.978 | ✔ | 3 correct | groq |
| `/predict/ct` | `Normal case (92).jpg` | Normal | 200 | **Normal** | 0.993 | ✔ | 3 correct | groq |

**6/6 correct, all with Grad-CAM heatmaps and Groq reports.** The guardrail chain is also intact:
a chest X-ray posted to `/predict/ct` still returns **422** ("This does not look like a chest CT
scan") from the sanity gate, before the modality gate or classifier.

`GET /health` after all of the above: `{"status":"ok","models_loaded":["xray","ct","gate",
"xray_sanity","ct_sanity"]}` — the same five, no sixth. The volume models are not in that list
by design; they are lazily loaded and reported separately by `GET /predict/ct/volume`.

---

## 7. Known limitations carried into the UI

All of the following are rendered as text in `CtVolumePage.jsx` — in a warning banner **above the
upload control**, and again in a limitations block **beneath every result** — and are also
returned in the API payload (`accuracy`, `validation.deferred`).

| limitation | figure | source |
|---|---|---|
| end-to-end nodule accuracy | **63.1% (53/84), 95% CI [52.4%, 72.6%]** | G.3 §4, `g3_combined.json` |
| detector recall ceiling | **84.5% (71/84)** — 13 nodules were never proposed, so a missing candidate is **not** evidence of a clear scan | G.3 §4/§6, verified §1.3 |
| classifier ceiling on perfectly centred crops | **79.8% (67/84)** | G.3 §4 |
| scan-level accuracy | 53.3% (24/45); 10 of 45 benign scans over-called Malignant | G.3 §5 |
| latency | ~30–35 s per scan (measured here 19.6–42.8 s, mean 35.6 s); async job + polling, one scan at a time | §5.1 |
| dataset | LIDC-IDRI only — **the detector's own training lineage.** No external validation, no generalization claim | Phases D–G.3, unchanged |
| input validation | **no volume OOD check, no scan-sanity model, no body-region check, `Modality` tag trusted** — not parity with `/predict/ct` | §3.3 |
| aggregation | a confidently-wrong spurious candidate still drives the call; concern levels are uncalibrated | §4 |
| jobs | in-memory only — they die with the process, no retry, no cross-worker sharing | §3.2 |

**Not carried into the UI, and deliberately so:** the volume route does **not** write to scan
history. The `scans` table's schema requires a single `predicted_class`, `confidence` and a
Grad-CAM `heatmap_png_base64`, none of which a multi-candidate volume report has. Forcing one in
would mean inventing a single-class summary for a result this phase spent §4 arguing should not
be collapsed to one. Left as future work rather than faked.

---

## 8. Live-system integrity, after the work

Re-hashed after all testing, with the server stopped.

```
checkpoints/ct_best_model_phase35_v2.pt                  93b3a979884edbc4   UNCHANGED
checkpoints/ct_lidc_nodulecrop_v4_stack3.pt              5ee4d8ec9f8328c1   UNCHANGED
checkpoints/ct_lidc_nodulecrop_v1.pt                     bdcbd6b2e5312435   UNCHANGED
checkpoints/ct_lidc_nodulecrop_v2_jitter.pt              5d60055e02974e91   UNCHANGED
checkpoints/ct_lidc_nodulecrop_v3_jitter_loosebenign.pt  9d1e9d7bf1ff5e16   UNCHANGED
eval_run/phaseE/pipeline.py                              936427fc20d389a0   UNCHANGED (imported, never edited)
eval_run/phaseE/app.py                                   95a6046384cabd2b   UNCHANGED (read as a pattern, not imported)
eval_run/phaseG1/stack_data.py                           9a639ffe4809babd   imported read-only; first recorded here

backend/main.py                                          816b06d0b78ec572   CHANGED, intentionally
                                                         (was 780120553178c0a5; +2 code lines, +6 comment lines)
```

**No prior phase's directory was written to at all** — every `__pycache__` under `eval_run/phaseE`,
`phaseG1`, `phaseD`, `lidc_full` and `ml` pre-dates this session and none has a new mtime, so the
imports reused existing bytecode. **No new dependency was installed** (`monai 1.6.0`,
`nibabel 5.4.2`, `pydicom 3.0.2` were already present from Phase D). **The scratch DB, uploads and
test scripts live in the session scratchpad, not in the repo.** `frontend/dist/` was produced by
the build check and is `.gitignore`d.

**Process check: `Get-Process python` returns 0, nothing listens on port 8089.** The backend was
stopped cleanly after testing.

---

## 9. What this phase does NOT show

* **No new accuracy measurement.** 63.1% is Phase G.3's number, reproduced here on 4 of its own
  45 patients. This phase measures **deployment fidelity** (§5.2), not model quality.
* **Still LIDC, still the detector's own lineage.** Nothing here is a generalization claim, and
  Phase A's deployment no-go on citing the live model's external-Malignant 12/12 as
  generalization evidence is untouched.
* **The frontend was built, not driven.** `vite build` passes and the page's contract against the
  API was verified field by field from the live JSON, but no browser session was run.
* **Four volumes is a deployment smoke test**, not a re-validation. The 16/16 bit-exact match is
  strong evidence the wiring is right; it says nothing new about whether 63.1% is good enough.
* **Concurrency was exercised serially.** Jobs were submitted one at a time, so the queueing path
  (`queued behind N running job(s)`) was implemented and code-reviewed but never observed under
  real contention.
* **Phase G.3's own conclusion stands**: the investigation's *research* arc is complete, and this
  deployment ships a research-grade feature labelled as such — it does not reopen the question of
  whether the 70% target can be reached.
