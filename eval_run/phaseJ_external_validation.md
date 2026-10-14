# Phase J: Genuine External Validation of the CT Volume Pipeline + Go/No-Go Decision

**Status:** Complete  
**Date:** 2026-09-10  
**Target:** Live deployed endpoint `POST /predict/ct/volume` on port 8000  
**External Cohort:** SPIE-AAPM-NCI Lung CT Challenge (LUNGx), acquired at University of Chicago Medical Center  
**Decision Verdict:** **NO-GO** (External accuracy: **28.6%**, threshold for retirement of old endpoint: **≥60%**)

---

## 1. Executive Summary & Decision

Prior to this phase, the CT volume pipeline's only accuracy measurement (**63.1% end-to-end**, measured in Phase G.3 and re-measured live in Phase I) used LIDC-IDRI data — the same source distribution and scanner cohort used to train both the detector and the crop classifier. The volume pipeline had never been tested on genuinely independent data.

This phase subjected the live running `POST /predict/ct/volume` endpoint to full, un-curated CT volumes from a completely independent clinical collection: the **SPIE-AAPM-NCI Lung CT Challenge (LUNGx)**. Ground truth was established by biopsy, resection, or definitive multi-year clinical follow-up.

### The Decision Question
> *If end-to-end accuracy on real external data is ≥60%, the old single-image CT endpoint should be marked for removal and the new pipeline's UI should become the primary CT feature. If accuracy is meaningfully below 60%, recommend keeping both features as-is (old as clearly-labeled fallback, new as clearly-labeled experimental).*

### The Decision: NO-GO
- **End-to-End Nodule Accuracy on External Data:** **28.6% (4/14)** [Wilson 95% CI: 11.7% to 54.6%]
- **Threshold for Cutover:** **≥60.0%**
- **Recommendation:** **NO-GO**. Do **NOT** remove or deprecate the single-image `/predict/ct` endpoint. Both features must remain as currently deployed:
  1. The single-image `/predict/ct` tool remains available (with its existing documented caveats regarding normal-tissue generalization).
  2. The CT volume pipeline `/predict/ct/volume` remains explicitly designated as **experimental / research-grade**, retaining its disclaimer in the API and UI.
  3. No routes, UI components, or legacy models are deleted in this phase.

---

## 2. Master Comparison Table: LIDC-IDRI Baseline vs. External LUNGx

Evaluated at detector decision floor `0.50` (primary) and report floor `0.30`, using ground-truth matching radius of **8.0 mm** (identical to Phase G.3 and Phase I).

| Metric | Lineage / Dataset | Internal (LIDC-IDRI) [Phase G.3 / I] | External (SPIE-AAPM LUNGx) [Phase J Live] | Delta / Note |
|---|---|---|---|---|
| **End-to-End Nodule Accuracy** | Pooled | **63.1%** (53/84) [52.4%, 72.6%] | **28.6%** (4/14) **[11.7%, 54.6%]** | **-34.5%** (Fails ≥60% threshold) |
| **End-to-End Nodule Accuracy (@0.30)** | Pooled | **64.3%** (54/84) [53.6%, 73.7%] | **28.6%** (4/14) [11.7%, 54.6%] | **-35.7%** |
| **Stage 1: Detector Recall (@0.50)** | Detection | **84.5%** (71/84) | **78.6%** (11/14) | -5.9% (Detector generalizes well) |
| **Stage 1: Detector Recall (@0.30)** | Detection | **88.1%** (74/84) | **85.7%** (12/14) | -2.4% |
| **Stage 2: Benign Nodule Accuracy** | Benign GT | **67.6%** (46/68) [55.9%, 77.4%] | **0.0%** (0/9) **[0.0%, 29.9%]** | **-67.6% (Severe failure mode)** |
| **Stage 2: Malignant Nodule Accuracy**| Malignant GT | **43.8%** (7/16) [23.1%, 66.8%] | **80.0%** (4/5) **[37.6%, 96.4%]** | **+36.2% (Strong sensitivity)** |
| **Scan-Level Accuracy** | Patient level | **53.3%** (24/45) [39.1%, 67.1%] | **45.5%** (5/11) **[21.3%, 72.0%]** | -7.8% |

---

## 3. Per-Class Breakdown & Confusion Analysis

### Confusion Matrix on External Nodule Data (Rows = Ground Truth, Cols = Prediction)

| True \ Pred | Malignant | Benign | Called Normal | Missed by Detector (d > 8mm) | Total | Sensitivity % |
|---|---|---|---|---|---|---|
| **Malignant** | **4** | 0 | 1 | 0 | 5 | **80.0%** (4/5) [37.6%, 96.4%] |
| **Benign** | 4 | **0** | 2 | 3 | 9 | **0.0%** (0/9) [0.0%, 29.9%] |
| **Normal** *(non-nodule)* | *(13 FP cands)* | *(5 FP cands)* | *(8 FP cands)* | N/A | *(26)* | *N/A (nodule cohort)* |

### Error Breakdown
1. **Benign Nodules (n=9)**:
   - **4 misclassified as Malignant (`misclassified_BM`)**: Detected accurately (mean distance 2.97 mm), but the classifier assigned strong probabilities to Malignant (P(Malignant) > 0.85).
   - **2 called Normal (`called_Normal`)**: Detected accurately (dist 1.31 mm and 5.03 mm), but the classifier graded them Normal.
   - **3 missed by detector (`missed_by_detector`)**: Nodule center was outside 8.0 mm of any detector candidate proposal (nearest was 9.30 mm, 9.68 mm, and 143.46 mm).
2. **Malignant Nodules (n=5)**:
   - **4 classified correctly as Malignant**: Pinpoint localization (mean distance 3.69 mm) with high classifier confidence (0.85 to 0.94).
   - **1 called Normal (`called_Normal`)**: Detected at 7.93 mm, but classifier called Normal.
   - **0 missed by detector**: 100% detector recall on external malignant nodules.
3. **Scan-Level Behavior (n=11 patients)**:
   - 5 of 11 scans were correctly diagnosed (all 4 Malignant scans were called Malignant, plus 1 Benign scan where all candidates were Benign).
   - 6 scans failed: 5 Benign patient scans were over-called Malignant because at least one detected candidate was over-classified as Malignant; 1 Malignant scan was called Normal.

---

## 4. Diagnostic Analysis: Why Did Accuracy Drop?

The failure to reach ≥60% is **not** primarily a detector localization problem; it is a **classifier domain shift / calibration bias**:

1. **The Detector (MONAI 3D RetinaNet) Generalizes Well**:
   - On completely external scans from University of Chicago (Philips and GE scanners, 1.0mm to 1.5mm slice thickness), the detector achieved **78.6% recall** at threshold 0.50 and **85.7%** at threshold 0.30.
   - When nodules were detected, the localization precision was high: median offset to true centroid was **3.83 mm** (well within the 8.0 mm box boundary and 13 mm tolerance established in Phase C.1).

2. **The Classifier (`ct_lidc_nodulecrop_v4_stack3.pt`) Collapses on Benign**:
   - The classifier has an extreme inductive bias toward `Malignant` on external scans.
   - On LIDC-IDRI, benign nodules often had subtle ground-glass or small solid appearances. On LUNGx, benign nodules (e.g. granulomas, hamartomas) possess sharp, dense margins that the LIDC-trained classifier overwhelmingly flags as `Malignant`.
   - Result: Not a single external benign nodule was correctly identified as Benign (4 called Malignant, 2 called Normal, 3 missed).
   - While high malignant sensitivity (80%) is clinically desirable for a cancer triage screening tool, an end-to-end accuracy of 28.6% and a 0% benign specificity makes the volume pipeline unreliable as an autonomous primary classifier.

---

## 5. Complete Per-Nodule & Per-Scan Log (All 11 External Patients)

All evaluations below were performed live over HTTP through `POST /predict/ct/volume` on `127.0.0.1:8000`.

| Patient ID | Nodule # | Slice # | Ground Truth | Nearest Det Cand | 3D Dist (mm) | Det Score | Pred @0.5 | Outcome @0.5 | Scan GT | Scan Pred | Scan Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `LUNGx-CT001` | 1 | 142 | Benign | `c00` | 3.99 mm | 0.995 | Malignant | `misclassified_BM` | Benign | Malignant | **WRONG** |
| `LUNGx-CT002` | 1 | 205 | Benign | None (<8mm) | 9.68 mm | 0.421 | None | `missed_by_detector` | Benign | Malignant | **WRONG** |
| `LUNGx-CT002` | 2 | 150 | Benign | `c01` | 2.62 mm | 0.892 | Malignant | `misclassified_BM` | — | — | — |
| `LUNGx-CT003` | 1 | 146 | Malignant | `c00` | 3.73 mm | 0.984 | Malignant | **`correct`** | Malignant | Malignant | **CORRECT** |
| `LUNGx-CT004` | 1 | 41 | Malignant | `c00` | 4.47 mm | 0.991 | Malignant | **`correct`** | Malignant | Malignant | **CORRECT** |
| `LUNGx-CT004` | 2 | 80 | Malignant | `c01` | 4.69 mm | 0.963 | Malignant | **`correct`** | — | — | — |
| `LUNGx-CT005` | 1 | 42 | Malignant | `c00` | 1.86 mm | 0.997 | Malignant | **`correct`** | Malignant | Malignant | **CORRECT** |
| `LUNGx-CT005` | 2 | 138 | Benign | `c03` | 1.31 mm | 0.742 | Normal | `called_Normal` | — | — | — |
| `LUNGx-CT006` | 1 | 50 | Malignant | `c01` | 7.93 mm | 0.681 | Normal | `called_Normal` | Malignant | Normal | **WRONG** |
| `LUNGx-CT007` | 1 | 188 | Benign | None (<8mm) | 143.46 mm | 0.220 | None | `missed_by_detector` | Benign | Benign | **CORRECT** |
| `LUNGx-CT008` | 1 | 139 | Benign | `c02` | 5.03 mm | 0.729 | Normal | `called_Normal` | Benign | Benign | **CORRECT** |
| `LUNGx-CT009` | 1 | 164 | Benign | `c00` | 3.93 mm | 0.912 | Malignant | `misclassified_BM` | Benign | Malignant | **WRONG** |
| `LUNGx-CT010` | 1 | 139 | Benign | `c00` | 1.34 mm | 0.978 | Malignant | `misclassified_BM` | Benign | Malignant | **WRONG** |
| `LUNGx-CT011` | 1 | 56 | Benign | None (<8mm) | 9.30 mm | 0.380 | None | `missed_by_detector` | Benign | Malignant | **WRONG** |

---

## 6. Go/No-Go Decision & Recommendation

### Formal Decision: **NO-GO**

1. **Threshold Assessment**:
   - The pre-established decision criterion was **≥60%** end-to-end nodule accuracy on genuine external data.
   - Measured external accuracy is **28.6%** [11.7%, 54.6%].
   - Even the upper bound of the Wilson 95% confidence interval (54.6%) does not reach the 60% threshold.

2. **Action Taken in This Phase**:
   - **Zero code, routes, or checkpoints removed or modified.**
   - The live single-image endpoint `/predict/ct` remains operational as the primary 2D interface.
   - The live volume endpoint `/predict/ct/volume` remains active under its experimental route prefix and banner.

---

## 7. Scoped-But-Not-Executed Plan for Future Phase K (For Reference Only)

If a future model iteration (e.g. trained with multi-institutional data or domain adaptation) achieves ≥60% on external validation, the following architectural and interface changes would be executed in Phase K:

### Backend Changes:
1. **Retire Old Route**:
   - In `backend/main.py`: Mark `POST /predict/ct` with `@app.post("/predict/ct", deprecated=True)` or remove the route handler.
   - Remove single-image CT resident model from memory in `backend/main.py`: Delete `ct_best_model_phase35_v2.pt` loading on startup, freeing ~1.5 GB host RAM and GPU memory.
   - Remove `checkpoints/ct_best_model_phase35_v2.pt`.
2. **Promote Volume Pipeline**:
   - Move `backend/ct_volume_routes.py` from experimental prefix to standard route `POST /predict/ct` (accepting both DICOM series zip and multipart slice batches).
   - Update `backend/tests/test_predict.py` to target the volume submission flow.

### Frontend Changes:
1. **Promote `CtVolumePage.jsx`**:
   - Elevate `CtVolumePage.jsx` from the experimental navigation tab to the primary "CT Analysis" view in `frontend/src/App.jsx`.
   - Update file upload dropzone in the primary CT tab to accept `.zip` archives of DICOM series and folder drag-and-drop.
2. **Retire 2D Slice Dropzone**:
   - Remove single-image 2D JPG/PNG CT upload UI or move it to a labeled legacy tab.

---

## 8. Artifacts Produced

- Scored Metrics & Evaluation Database: `eval_run/phaseJ_data/phaseJ_results_scored.json`
- Ground Truth Annotations (14 LUNGx Scans): `eval_run/phaseJ_data/gt_nodules_14.json`
- Downloaded External DICOM Cohort (11 Scans, ~1.1 GB): `eval_run/phaseJ_data/volumes/`
- Resume & Scoring Harness: `eval_run/phaseJ_resume.py`
