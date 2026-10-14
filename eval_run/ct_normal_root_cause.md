# Root cause: CT "Normal" class collapses on external images (0/9), while Malignant holds (12/12)

Diagnosis only — no model, checkpoint, or preprocessing code was modified.

## 1. Image inspection

| | In-dataset (IQ-OTH/NCCD), all 3 classes | External Malignant (n=12) | External Normal (n=9) |
|---|---|---|---|
| Size | Exactly 512×512, always | 360–490 × 251–340 (variable) | 360–940 × 251–627 (variable) |
| Aspect ratio | 1.00 (square) | 1.22–1.45 | 1.28–1.50 |
| Mode | RGB | RGBA/RGB | RGBA/RGB |
| Min pixel value | **37, every single image** (never true black) | 0 | 0 |
| % pixels near-black (<10) | 0.0% always | 31.8–37.3% (tight band) | **0.5–51.6% (wildly variable)** |

Two things stand out:
- The in-dataset's `min=37` floor is a **dataset-wide signature present in all three classes** (Normal, Benign, Malignant all show it) — not something specific to Normal. So "no true black" vs "has true black" cannot by itself explain why Normal fails but Malignant doesn't; both external classes have true black.
- External Malignant's near-black percentage is a **tight, consistent band (32–37%)** across all 12 images — these look like a consistent crop/window style. External Normal's near-black percentage is **erratic (0.5% to 51.6%)** — these 9 images look far less consistent in framing/zoom/windowing with each other than the Malignant set is. This asymmetry matters more than the raw pixel stats (see §3).
- Aspect-ratio distortion is essentially **identical** between the external Malignant (1.22–1.45) and external Normal (1.28–1.50) groups — same source dataset, same squashing when resized to a square input. This rules out "aspect-ratio squashing" as a Normal-specific cause; it's a shared confound that Malignant tolerates and Normal doesn't.

## 2. Preprocessing pipeline

`ml/dataset.py::build_transforms` (used for both training and the live `/predict/ct` route, confirmed via `backend/gradcam_utils.py`'s `preprocess_image`): `Grayscale(3ch) → Resize((224,224)) → ToTensor → Normalize(ImageNet mean/std)`.

- **No HU windowing, no scanner-specific contrast normalization, no aspect-ratio-preserving crop/pad** — a naive resize-to-square is applied uniformly regardless of source or class.
- Confirmed by running the transform directly on `normal_22.png` (360×263) and `normal_7.png` (940×627): both get squashed to 224×224 and their post-transform tensor min/max lands at the same clipped extremes (-2.12/2.64) as the in-dataset image — the transform runs identically. **This preprocessing step does not treat Normal differently from Malignant** — it's class-agnostic, so it cannot be the differentiating cause, though it is a generic weakness (see recommendation below).

## 3. Grad-CAM comparison

| Image | Pred (conf) | Attention location |
|---|---|---|
| In-dataset Normal case (112) | Normal (0.51 — barely correct) | Spread across both lung fields, bilateral |
| In-dataset Normal case (127) | Normal (0.92) | Concentrated on lung parenchyma, both lungs |
| External Normal `normal_22` | Malignant (0.98 — confidently wrong) | **Mediastinum / central great-vessel region**, almost no lung parenchyma |
| External Normal `normal_7` | Malignant (0.42 — barely wrong) | **Mediastinum + area above the body (table/border)**; visible lung vasculature gets almost no attention |
| External Normal `normal_11` | Benign (0.56 — barely wrong) | **Heart/mediastinal shadow**, center of image |
| External Malignant `adenocarcinoma_000108` | Malignant (0.95, correct) | Peripheral chest-wall/pleural region (plausible lesion location) |
| External Malignant `large_cell_000108` | Malignant (0.78, correct) | Broad/diffuse across both lungs and border |

**All three external Normal misclassifications show attention shifted away from lung parenchyma toward the mediastinum/central chest structures** — a pattern not seen in either in-dataset Normal example, both of which show attention on the lungs themselves. This looks like an **anatomical slice-level mismatch**: IQ-OTH/NCCD's Normal class appears to be built from a canonical mid-lung-parenchyma slice per case, while these external "Normal" images include slices closer to the aortic-arch/great-vessel level — a level the model has apparently never seen labeled Normal, so the visually "busy" central structures at that level get read as suspicious.

This is a "reasonable-looking attention, wrong label" pattern for the confident miss (`normal_22`) — the heatmap isn't noise or border artifacts, it's coherently focused on a real anatomical structure, just the wrong one for a Normal call. That points at a **generalization/coverage gap (learned template of one slice level), not a calibration/threshold bug.**

## 4. Confidence levels of the 9 misclassifications

| Image | Pred | Confidence | True-class (Normal) probability |
|---|---|---|---|
| normal_10 | Malignant | 0.93 | 0.014 |
| normal_22 | Malignant | 0.98 | 0.002 |
| normal_11 | Benign | 0.56 | 0.420 (close second) |
| normal_21 | Malignant | 0.54 | 0.095 |
| normal_23 | Malignant | 0.69 | 0.100 |
| normal_24 | Benign | 0.70 | 0.166 |
| normal_6 | Malignant | 0.53 | 0.038 |
| normal_7 | Malignant | 0.42 | 0.234 |
| normal_8 | Malignant | 0.47 | 0.209 |

**Mixed, not uniform** — 2 of 9 are confidently wrong (>0.9: `normal_10`, `normal_22`), the rest are narrow/borderline decisions (0.42–0.70), several with the true Normal class as a fairly close second place (`normal_11`, `normal_7`, `normal_8`). Also notable: even the two **correct** in-dataset Normal examples pulled for Grad-CAM were not all confident — one landed at 0.51 (barely above the decision line). So the in-dataset 100% (25/25) recall was not built on wide margins; it was a boundary the model reliably clears in-distribution but by a modest margin in several cases, consistent with Normal being a comparatively fragile, narrowly-learned decision rather than a robust concept, further destabilized once the input distribution shifts even slightly.

## Most likely explanation

A combination of two things, in order of contribution:

1. **Anatomical/slice-level template mismatch** (primary, per §3): the Normal class was learned from a narrow, homogeneous slice-selection style specific to IQ-OTH/NCCD (mid-lung parenchyma, uniform windowing with `min=37` floor). External Normal images include different slice levels and windowing styles; the model has no learned concept of "normal" at those levels and defaults toward Malignant/Benign when central-chest structures look visually busy.
2. **Normal is an inherently weaker, negative-defined signal** (per §4): unlike Malignant, which is driven by a salient positive local feature (a mass/lesion) that survives resizing, aspect distortion, and moderate domain shift, "Normal" requires the *absence* of any such feature to be recognized — a much more diffuse, template-dependent decision that transfers far less robustly across acquisition style.

Preprocessing distortions (naive squash-to-square resize, lack of HU/contrast normalization) are real, generic weaknesses in the pipeline, but they are **ruled out as the differentiating cause** here — they apply identically to the external Malignant images, which still classify correctly.

## Fixable via preprocessing alone, or needs retraining?

**Partially fixable via preprocessing, but the core issue needs retraining with more diverse Normal-class data.**

- A scanner-agnostic contrast/windowing normalization (e.g., percentile-based rescaling instead of relying on whatever 8-bit range the source export used) and an aspect-ratio-preserving resize+pad could plausibly **rescue several of the borderline misses** — `normal_11`, `normal_7`, `normal_8`, `normal_6` all have non-trivial Normal probability (0.20–0.42) already; nudging the input distribution closer to what the model was trained on could tip some of these back to Normal.
- It will **not** fix the confidently-wrong cases (`normal_10` at 0.93, `normal_22` at 0.98) or the underlying slice-level gap — no amount of contrast/aspect normalization teaches the model what a normal great-vessel-level or upper-mediastinal slice looks like if it has never seen one labeled Normal. That requires **retraining with a more anatomically diverse Normal class** (multiple slice levels, ideally multiple scanners/sources), not a preprocessing patch.

**No fix has been applied.** Awaiting confirmation before attempting either the preprocessing changes or a retraining plan.
