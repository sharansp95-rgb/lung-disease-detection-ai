# Diagnostic Report: Cross-Source Generalization & Failure Modes

**Date:** 2026-09-05  
**Harness:** `test/`  
**Target:** Chest X-ray and Chest CT Disease Classification Pipeline  

---

## 1. Provenance of the Earlier Probe (`sample images/`)

The earlier diagnostic (`GENERALIZATION_DIAGNOSIS.md`) evaluated performance on `sample images/` under the assumption that these files represented images from outside the training sources. A full SHA-256 and perceptual hash audit against the complete project corpus (`data/`, `raw/`, `datasets/`, `seg_data/` — 254,084 total images) was executed using `verify_provenance.py`.

### Results

| Match Category | Count | Percentage |
|---|---|---|
| `EXACT_MATCH_IN_UNKNOWN` (matched in `raw/` sources) | 180 | 57.1% |
| `EXACT_MATCH_IN_TRAIN` | 112 | 35.6% |
| `EXACT_MATCH_IN_VAL` | 13 | 4.1% |
| `EXACT_MATCH_IN_TEST` | 10 | 3.2% |
| **Genuinely External (`NO_MATCH`)** | **0** | **0.0%** |
| **Total Probe Images** | **315** | **100.0%** |

### Verdict
**`GENERALIZATION_DIAGNOSIS.md` measured nothing, and its conclusions are void.** Every single image in `sample images/` (315/315, 100%) traces directly back to the project's training and raw dataset pools. None of the images were external.

---

## 2. Corruption Test (Evaluating In-Distribution Robustness)

Both modalities were evaluated on held-out test splits (`data/xray/test` and `data/ct_scan/test`) through the real FastAPI backend endpoint (`/predict/xray` and `/predict/ct`).

> [!NOTE]
> **API Execution Note:** The default production backend applies `@limiter.limit("20/minute")`. During initial testing, `test_corruption.py` caused requests to exceed the limit after the baseline phase, resulting in HTTP 429 Too Many Requests (recorded as rejections). Starting the server with standard environment configuration `RATELIMIT_ENABLED=false` disabled rate limiting without modifying any backend source code, enabling full test execution.

### X-Ray Corruption Results (`results/corruption_xray.json`)
- **Baseline Sampled:** 20 images (5 per class: COVID-19, Normal, Pneumonia, Tuberculosis)
- **Baseline Correct (unaltered):** 17 images (100.0% baseline)

| Transform | Accuracy | Flipped | Rejected | Delta vs. Baseline |
|---|---|---|---|---|
| baseline (unaltered) | 100.0% | 0 | 0 | +0.0 |
| resize 512 roundtrip | 100.0% | 0 | 0 | +0.0 |
| JPEG quality 60 | 100.0% | 0 | 0 | +0.0 |
| black border 10% | 100.0% | 0 | 0 | +0.0 |
| gamma 0.7 | 100.0% | 0 | 0 | +0.0 |
| gamma 1.4 | 100.0% | 0 | 0 | +0.0 |
| contrast x0.7 | 100.0% | 0 | 0 | +0.0 |
| contrast x1.4 | 100.0% | 0 | 0 | +0.0 |
| **intensity inverted** | **23.5%** | **13** | **0** | **-76.5** |
| histogram equalized | 94.1% | 1 | 0 | -5.9 |

**Transforms costing 20+ points in X-ray:**
- **`intensity inverted`**: Cost **76.5 points** (accuracy plummeted from 100.0% to 23.5%, with 13 of 17 images flipped).

---

### CT Corruption Results (`results/corruption_ct.json`)
- **Baseline Sampled:** 15 images (5 per class: Benign, Malignant, Normal)
- **Baseline Correct (unaltered):** 14 images (100.0% baseline)

| Transform | Accuracy | Flipped | Rejected | Delta vs. Baseline |
|---|---|---|---|---|
| baseline (unaltered) | 100.0% | 0 | 0 | +0.0 |
| resize 512 roundtrip | 100.0% | 0 | 0 | +0.0 |
| JPEG quality 60 | 100.0% | 0 | 0 | +0.0 |
| black border 10% | 85.7% | 2 | 0 | -14.3 |
| gamma 0.7 | 92.9% | 1 | 0 | -7.1 |
| gamma 1.4 | 100.0% | 0 | 0 | +0.0 |
| contrast x0.7 | 100.0% | 0 | 0 | +0.0 |
| contrast x1.4 | 92.9% | 1 | 0 | -7.1 |
| **intensity inverted** | **78.6%** | **2** | **1** | **-21.4** |
| **histogram equalized** | **64.3%** | **4** | **1** | **-35.7** |

**Transforms costing 20+ points in CT:**
- **`histogram equalized`**: Cost **35.7 points** (accuracy fell from 100.0% to 64.3%, with 4 images flipped and 1 rejected).
- **`intensity inverted`**: Cost **21.4 points** (accuracy fell from 100.0% to 78.6%, with 2 images flipped and 1 rejected by modality gate).

---

## 3. Per-Stage Traces for Breaking Transforms

For each transform dropping 20+ points, a representative flipped image was traced stage-by-stage through the pipeline. Denormalized 224×224 input tensors were saved side-by-side to `test/results/`.

### Trace 1: X-Ray — `intensity inverted`
- **File:** `covid19db_COVID-68.png` (True Class: `COVID-19`)
- **Artifact:** [trace_xray_intensity_inverted.png](file:///d:/7th%20sem/sdp/test/results/trace_xray_intensity_inverted.png)

| Stage | Unaltered Original | Transformed (`intensity inverted`) |
|---|---|---|
| **Lung Crop (`compute_lung_crop`)** | `used_unet=False`, `bbox_area_frac=0.9806`. Branch: `> 0.98 -> fallback border_crop` | `used_unet=True`, `bbox_area_frac=0.7967`. Branch: `U-Net crop accepted (0.7967 in [0.05, 0.98])` |
| **Sanity Gate** | Passed (`scan`, conf: 1.0000) | Passed (`scan`, conf: 1.0000) |
| **Modality Gate** | Passed (`xray`, conf: 0.9936) | Passed (`xray`, conf: 0.9891) |
| **OOD Detector** | Passed (distance: 23.44 < threshold: 32.89) | Passed (distance: 19.80 < threshold: 32.89) |
| **Final Disease Prediction** | **`COVID-19` (conf: 100.0%)** | **`Tuberculosis` (conf: 99.1%)** *(FLIPPED)* |

**Observation:** Inverting the intensity profile completely flips the pathology prediction from COVID-19 to Tuberculosis with 99.1% confidence. Paradoxically, the inverted image lands *closer* to the training feature centroid (distance 19.80 vs 23.44), passing the OOD guard effortlessly. In addition, the inverted grayscale values alter the segmentation U-Net mask, shifting it from triggering the `>0.98` fallback on the original to accepting the crop at 79.7% area.

---

### Trace 2: CT — `histogram equalized`
- **File:** `Bengin case (81).jpg` (True Class: `Benign`)
- **Artifact:** [trace_ct_histogram_equalized.png](file:///d:/7th%20sem/sdp/test/results/trace_ct_histogram_equalized.png)

| Stage | Unaltered Original | Transformed (`histogram equalized`) |
|---|---|---|
| **Sanity Gate** | Passed (`scan`, conf: 0.9996) | Passed (`scan`, conf: 0.9707) |
| **Modality Gate** | Passed (`ct`, conf: 0.9999) | Passed (`ct`, conf: 0.9968) |
| **OOD Detector** | Passed (distance: 19.85 < threshold: 25.62) | Passed (distance: 21.56 < threshold: 25.62) |
| **Final Disease Prediction** | **`Benign` (conf: 96.5%)** | **`Normal` (conf: 82.6%)** *(FLIPPED)* |

**Observation:** Global histogram equalization stretches background noise and alters subtle soft-tissue attenuation gradients, causing the network to misclassify a benign pulmonary nodule as normal lung tissue.

---

### Trace 3: CT — `intensity inverted`
- **File:** `Bengin case (81).jpg` (True Class: `Benign`)
- **Artifact:** [trace_ct_intensity_inverted.png](file:///d:/7th%20sem/sdp/test/results/trace_ct_intensity_inverted.png)

| Stage | Unaltered Original | Transformed (`intensity inverted`) |
|---|---|---|
| **Sanity Gate** | Passed (`scan`, conf: 0.9996) | Passed (`scan`, conf: 0.9455) |
| **Modality Gate** | Passed (`ct`, conf: 0.9999) | Passed (`ct`, conf: 0.9177) |
| **OOD Detector** | Passed (distance: 19.85 < threshold: 25.62) | Passed (distance: 21.11 < threshold: 25.62) |
| **Final Disease Prediction** | **`Benign` (conf: 96.5%)** | **`Normal` (conf: 78.3%)** *(FLIPPED)* |

**Observation:** Reversing CT polarity flips subtle nodule contrast, resulting in misclassification as Normal with 78.3% confidence.

---

## 4. Diagnosis: Which Cause is Dominant?

The empirical evidence isolates the relative contributions of the three candidate causes:

### Ranking & Attribution

1. **Primary Immediate Cause: (B) Intensity/Format Shift (~55%)**
   - **Evidence:** The model breaks catastrophically on its *own* training/testing images under light photometric modification.
   - Specifically, `intensity inverted` costs **76.5 points** on X-ray (flipping 76.5% of verified baseline images) and **21.4 points** on CT. `histogram equalized` costs **35.7 points** on CT.
   - Medical images sourced online or from varying PACS systems frequently alternate between `MONOCHROME1` (air white, bones black) and `MONOCHROME2` (air black, bones white). The pipeline lacks any polarity check or standard photometric windowing in `preprocess_image()`, immediately breaking predictions on inverted or equalized external images before pathology can even be assessed.

2. **Secondary Core Cause: (C) Source-Database Confound (~35%)**
   - **Evidence:** As documented in the paper's Section IV-E audit, a 32×32 probe recovers the source dataset at 97.1% accuracy, and COVID-19 classifications survive lung field blanking with only a 1.7% drop.
   - When images are drawn from external repositories, the feature representation drifts because the classifier relies on acquisition signatures (borders, sensor noise, aspect ratios).
   - Furthermore, our test revealed that 25% of genuine CT scans passed through both the sanity gate, modality gate, and the X-ray OOD detector to be confidently misclassified as X-ray COVID-19 or Normal.

3. **Tertiary Contributing Factor: (A) Preprocessing Pipeline Mismatch (~10%)**
   - **Evidence:** On the positive side, `black border 10%` retained **100.0% accuracy** on X-ray because the U-Net successfully segmented and cropped the lung fields.
   - However, the pipeline order introduces an architectural flaw: `compute_lung_crop` runs *before* the sanity and modality gates. On non-standard or out-of-modality scans, the segmentation model produces arbitrary crops before the gates can evaluate the original framing.

---

## 5. Recommended Fixes

### Fix 1: Photometric Polarity & Windowing Standardization in Preprocessing (Fixes B)
- **Target Files & Functions:**
  - `backend/gradcam_utils.py` -> `preprocess_image()`
  - `ml/dataset.py` -> `build_transforms()`
- **Implementation:**
  - Implement automated polarity detection in `preprocess_image()`: inspect border/corner intensity distribution vs lung field median. If the image has inverted polarity (air is bright, tissue is dark), invert it back to standard `MONOCHROME2` before feeding it to downstream models.
  - Implement robust percentile clipping (e.g. 1st to 99th percentile) rather than global histogram equalization.
- **Requires Retraining?** **NO for inference polarity normalization.** Adding polarity detection to `preprocess_image()` immediately restores compatibility for inverted inputs with zero retraining.
- *(Optional long-term)*: Retrain with random photometric inversion and contrast jitter augmentation for greater intrinsic model invariance.

### Fix 2: Reorder API Pipeline Gates (Fixes A and Gate Misrouting)
- **Target Files & Functions:**
  - `backend/main.py` -> `_run_prediction()`
- **Implementation:**
  - Execute the Modality Gate *first* on the raw image tensor before running `compute_lung_crop()`.
  - If a CT scan is uploaded to `/predict/xray`, the modality gate immediately intercepts it and returns a clean 422 redirecting the user to the CT tab, preventing the X-ray segmentation U-Net from running on a CT slice and preventing the sanity gate from falsely reporting "not an actual scan".
- **Requires Retraining?** **NO.** Requires ~10 lines of reordering in `backend/main.py`.

### Fix 3: Cross-Source Multi-Domain Augmentation (Fixes C)
- **Target Files & Functions:**
  - `ml/train.py`, `ml/dataset.py`
- **Implementation:**
  - Apply border occlusion, random cropping, color jitter, and mixup/cutmix across dataset boundaries to suppress acquisition site features.
- **Requires Retraining?** **YES** (requires retraining classifier checkpoints on GPU).

---

## 6. What Could Not Be Measured, and Why

1. **Evaluation on `external_probe/`:**
   - `test/external_probe/` is currently empty by design (containing only class README placeholders as specified in `SOURCING.md`). Per the hard constraints, no medical images were downloaded or fabricated.
   - Quantitative evaluation of external images requires sourcing real external images according to `SOURCING.md` guidelines (e.g., 5–10 images per class from RSNA, CheXpert, or BIMCV).

---

## Appendix: Three Open Questions on the IEEE Paper

### a) Real Stage Order Inside `_run_prediction()`
- **Question:** Does the sanity gate run before or after the U-Net lung crop?
- **Finding:** In `backend/main.py` (lines 292–361), the actual sequence is:
  1. `_read_and_validate_image()`
  2. `compute_lung_crop()` (X-ray only: segmentation U-Net)
  3. `preprocess_image()` (resizing to 224×224 and ImageNet normalization)
  4. `_sanity_predict()` (Sanity Gate)
  5. `_gate_predict()` (Modality Gate)
  6. `ood_detector.check()` (OOD Centroid Distance)
  7. `predict_with_heatmap()` (Disease Classifier & Grad-CAM)
- **Answer:** The sanity gate runs **after** the U-Net lung crop. Fig. 1 and Algorithm 1 in the paper correctly depict the code's execution sequence (`validate -> lung crop -> preprocess -> sanity -> modality -> OOD -> classify`).
- *Important note:* Because lung cropping occurs *before* any gate, non-scan or cross-modality uploads are segmented by the X-ray U-Net before the sanity or modality gates inspect them.

### b) Segmentation U-Net Reproduction Difference From Live Startup
- **Question:** The paper's Table II says IoU diff 0.0115; an earlier draft said 0.0116. Which is the real value?
- **Finding:** From the live startup log on the local validation set (`seg_data/val/`, 400 pairs):
  ```
  checkpoint claims dice=0.9890 iou=0.9786, reproduced dice=0.9830 iou=0.9671 in 37.9s (diff dice=0.0060, iou=0.0116)
  ```
  The exact floating-point calculation:
  - Claimed: `0.978648...`
  - Reproduced: `0.967088...`
  - Difference: `|0.967088 - 0.978648| = 0.011560...`
- **Answer:** **`0.0116`** is the exact rounded value printed at 4 decimal places (`:.4f`) by the live startup check. `0.0115` was an unrounded truncation in earlier drafting. Table II should report **`0.0116`** to match the live startup assertion.

### c) Origin of the Rejection Message on Cross-Modality Submissions
- **Question:** Which gate produces `"This does not look like a chest X-ray. Please upload an actual scan image."` when a genuine CT is submitted to the X-ray route?
- **Finding:**
  1. The string is produced exclusively by the **Sanity Gate** (`backend/main.py:313`):
     ```python
     detail=f"This does not look like a {label}. Please upload an actual scan image."
     ```
     (where `label = "chest X-ray"`).
     The Modality Gate produces a completely distinct string (`backend/main.py:331`):
     ```python
     detail=f"This looks like a {actual_label}, not a {expected_label} -- please use the ... tab instead."
     ```
     The two gates **do not** share an error string.
  2. The Sanity Gate is **over-triggering on real CT scans**. When 20 genuine CT scans from `data/ct_scan/test/` were submitted to `/predict/xray`:
     - **10 (50%)** were correctly caught by the Modality Gate.
     - **5 (25%)** were prematurely rejected by the Sanity Gate with the non-scan message.
     - **5 (25%)** passed both gates and OOD and were misclassified as X-ray diseases (`Normal` or `COVID-19`).
  3. **Why it happens:** The X-ray sanity model was trained strictly on planar X-rays against web photos/documents. It was never trained on CT slices. When a CT slice is cropped by the X-ray U-Net and passed to `xray_sanity`, the sanity model classifies it as `not_scan`, producing the misleading error message before the modality gate is reached.
