# X-ray live-crop fix — root cause, change, and validation

Resolves **N1** in `VERIFICATION_REPORT_2.md` §4.2/§4.3 ("live X-ray accuracy is ~78–84%, not 96.57%,
and the gap is created by the serving path, not the model").

Date: 2026-09-08 · Files changed: `backend/seg_utils.py` **only** · Backups: `eval_run/xray_cropfix_backups/`

---

## 1. Root cause

`backend/seg_utils.py::compute_lung_crop()` returned the **raw lung bounding box** — an arbitrary,
almost-never-square rectangle. The classifier's preprocessing
(`ml/train_xray_txrv.py::build_xrv_transform`) begins with:

```python
arr = xrv.utils.normalize(arr, XRV_MAXVAL, reshape=True)
arr = _center_crop(arr)     # xrv.datasets.XRayCenterCrop()  <-- crops to the SHORTER side
arr = _resizer(arr)         # -> 224x224
```

`XRayCenterCrop` crops a non-square input down to a centred square of its shorter side. So the live
path handed the model a **centre-cropped lung field with the outer lung margins amputated**.

The training images never hit that path. `scripts/prepare_xray.py` finishes *every* crop with:

```python
def crop_and_resize(arr, size=TARGET_SIZE):
    return Image.fromarray(arr).resize((size, size), Image.LANCZOS)   # non-aspect-preserving squash
```

Every file under `data/xray/{train,val,test}/` is therefore already 224×224 square, with the **whole**
crop squashed in and nothing discarded — which makes `XRayCenterCrop` a no-op on training data and
destructive on live data. The two implementations were meant to share logic (`seg_utils.py` already
imports `border_crop`, `mask_bbox`, `MASK_PAD_PX`, `TB_BORDER_FRAC` from `prepare_xray.py`) but the
final `crop_and_resize` step was the one piece never carried across.

### Measured, per-image (`eval_run/xray_cropfix_backups/cropdiff_meta.json`)

All 11 sampled live crops were non-square:

| file | raw upload | live crop | what `XRayCenterCrop` kept | lost |
|---|---|---|---|---|
| `tbdb_Tuberculosis-85.png` | 512×512 | **384×312** | 312×312 | 72 px of width (19%) |
| `tbdb_Normal-312.png` | 299×299 | **248×185** | 185×185 | 63 px of width (25%) |
| `tbdb_Tuberculosis-5.png` | 512×512 | **404×478** | 404×404 | 74 px of height |
| `covid19db_Normal-879.png` | 299×299 | **245×197** | 197×197 | 48 px of width |
| `covid19db_COVID-3467.png` | 299×299 | **219×268** | 219×219 | 49 px of height |
| `tbdb_Tuberculosis-171.png` | 512×512 | 390×382 | 382×382 | 8 px |
| `tbdb_Tuberculosis-121.png` | 512×512 | 432×472 | 432×432 | 40 px of height |
| `tbdb_Normal-2581.png` | 299×299 | 219×193 | 193×193 | 26 px of width |
| `covid19db_Normal-111.png` | 299×299 | 227×190 | 190×190 | 37 px of width |
| `covid19db_Viral Pneumonia-49.png` | 299×299 | 272×259 | 259×259 | 13 px |
| `covid19db_Viral Pneumonia-491.png` | 299×299 | 222×259 | 222×222 | 37 px of height |

### Show, don't tell — the crop-diff PNGs

`eval_run/xray_cropfix_backups/cropdiff_<name>.png`, one per image above. Six panels, left to right:

1. **raw upload** — the original source file
2. **TRAIN crop (prepare_xray)** — what `prepare_xray.py` builds from that same raw file
3. **→ model sees (train)** — panel 2 through the XRV transform (identical to panel 2; centre-crop is a no-op)
4. **LIVE crop** — `compute_lung_crop()` output before the fix, with its actual pixel dimensions
5. **→ model sees (LIVE, cropped!)** — panel 4 through the XRV transform: **visibly zoomed in, lung margins gone**
6. **→ model sees (FIXED)** — the crop squashed to 224×224 first, which restores panel 3's framing

The clearest cases are `cropdiff_tbdb_Normal-312.png` (248×185 → 185×185; both outer lung fields cut off)
and `cropdiff_tbdb_Tuberculosis-85.png` (384×312 → 312×312).

### Why the failures were false COVID-19

The classifier was fitted on a source-dependent framing mixture — COVID-19/Pneumonia/covid19db-Normal
built with a tight mask lung bbox, Tuberculosis/tbdb-Normal built with a uniform 8% border crop
(`TB_BORDER_FRAC`, no masks in the TB source). Framing is a learned cue. The un-squashed live crop was
*tighter than any training framing* — tighter even than the tight class — so it pushed every image toward
the tight-framing classes, and COVID-19 is the largest of those. Hence "every non-COVID flip becomes
COVID-19" in §4.3.

### Reconciling with `PROJECT_STATUS.md` §3 (the 60-image study that found the crop *helped*)

Not a contradiction. That study ran on the **old ImageNet-backbone checkpoint**
(`xray_best_model_phase35_v2.pt`), whose preprocessing has no `XRayCenterCrop`, and its baseline was a
*universal border crop*, not "no crop". Its disclosed TB regression (93%→87%) is the class-conditional
framing effect described above — real, and still present after this fix, but much smaller than the
geometric bug that dominated the 240-image study.

---

## 2. The change

`backend/seg_utils.py` — the only file modified. Two hunks.

```diff
-from prepare_xray import border_crop, mask_bbox, MASK_PAD_PX, TB_BORDER_FRAC  # noqa: E402
+from prepare_xray import (  # noqa: E402
+    border_crop, crop_and_resize, mask_bbox, MASK_PAD_PX, TARGET_SIZE, TB_BORDER_FRAC,
+)
```

```diff
 # in compute_lung_crop(), all three return paths:
     if mask_small.max() == 0:
         cropped = border_crop(gray, TB_BORDER_FRAC)
-        return Image.fromarray(cropped), False
+        return crop_and_resize(cropped, TARGET_SIZE), False
     ...
     if bbox_area_frac < MIN_BBOX_AREA_FRAC or bbox_area_frac > MAX_BBOX_AREA_FRAC:
         cropped = border_crop(gray, TB_BORDER_FRAC)
-        return Image.fromarray(cropped), False
+        return crop_and_resize(cropped, TARGET_SIZE), False
     ...
     cropped = gray[orig_r0:orig_r1, orig_c0:orig_c1]
-    return Image.fromarray(cropped), True
+    return crop_and_resize(cropped, TARGET_SIZE), True
```

plus a docstring explaining why the squash is load-bearing. `crop_and_resize` and `TARGET_SIZE` are
**imported from `prepare_xray.py`**, not reimplemented, so the serving path and the data-build path stay
byte-identical by construction. `scripts/prepare_xray.py` needed no correction — it was already right;
the serving path was the side that had drifted.

`backend/main.py` was **not modified**. No parameters were tuned, no thresholds moved.

### Alternatives measured and rejected (`eval_run/results_cropfix_variants.json`, `results_cropfix_expand.json`)

| candidate | raw-upload acc | note |
|---|---|---|
| **squash to 224 (shipped)** | **0.9500** | best COVID recall (0.933), zero new parameters, 1× inference |
| 2-view ensemble (lung + 8% border), geometric mean | 0.9500 | same overall, COVID recall drops 0.933→0.850, 2× inference |
| 2-view ensemble, arithmetic mean | 0.9458 | worse |
| 3-view ensemble (+ whole image) | 0.9125 | much worse on originals |
| squash + 5% bbox expansion | 0.9625 | rejected: knife-edge (0.904 at 10%, 0.825 at 30%), COVID recall 0.883 — a tuned constant fitted to a 240-image sample |

The shipped fix is the only candidate that is a *principled train/serve alignment* rather than a fitted
constant, and it has the best class balance.

### Backups

| file | backup |
|---|---|
| `backend/seg_utils.py` | `eval_run/xray_cropfix_backups/seg_utils.py.orig` |
| `backend/main.py` (unmodified, backed up anyway) | `eval_run/xray_cropfix_backups/main.py.orig` |
| `scripts/prepare_xray.py` (unmodified, backed up anyway) | `eval_run/xray_cropfix_backups/prepare_xray.py.orig` |
| pre-fix results | `results_pathdiff_BEFORE.json`, `results_rawprobe_BEFORE.json` |

---

## 3. Validation

### 3.1 The 240-image offline-vs-live comparison (identical script, identical seed-3 sample)

`eval_run/run_pathdiff.py` → `run_pathdiff_after.py`; `results_pathdiff_after.json`.

| class | offline (= how 96.57% was measured) | live BEFORE | live AFTER |
|---|---|---|---|
| COVID-19 | 0.9333 | 0.9000 | 0.9000 |
| Normal | 0.9833 | 0.7833 | **0.9000** |
| Pneumonia | 0.9500 | 0.8333 | **0.9500** |
| Tuberculosis | 1.0000 | 0.6833 | **0.9333** |
| **overall (n=240)** | **0.9667** | **0.8000** | **0.9208** |

Caveat carried forward from §4.3: these files are *already* crop-preprocessed, so the live path
double-crops them. This is the pessimistic framing, not the realistic one.

### 3.2 Original, uncropped uploads — what a real user actually sends

`eval_run/run_rawprobe.py` → `run_rawprobe_after.py`; `results_rawprobe_after.json` (240 originals, seed 11).

| class | live BEFORE | live AFTER | claimed |
|---|---|---|---|
| COVID-19 | 0.9333 | 0.9333 | 0.9440 |
| Normal | 0.8000 | **0.9500** | 0.9722 |
| Pneumonia | 0.9333 | **0.9667** | 0.9403 |
| Tuberculosis | 0.8667 | **0.9500** | 1.0000 |
| **overall (n=240)** | **0.8833** | **0.9500** | **0.9657** |

The no-crop control is unchanged and still confirms the crop is necessary: 0.8042 overall, COVID-19
recall **0.40**. Removing the crop remains the wrong fix.

### 3.3 Full 4-class confusion matrix through the LIVE `/predict/xray` endpoint (post-fix)

`eval_run/cropfix_live_run.py` → `results_cropfix_live.json`. Real HTTP, backend on port 8077 with an
isolated scratch DB, same seed-3 240-image sample, rate-limit paced. Rows = truth.

| true \ pred | COVID-19 | Normal | Pneumonia | Tuberculosis | rejected 422 (OOD) |
|---|---|---|---|---|---|
| **COVID-19** (60) | **54** | 1 | 3 | 2 | 0 |
| **Normal** (60) | 3 | **54** | 1 | 2 | 0 |
| **Pneumonia** (60) | 1 | 2 | **57** | 0 | 0 |
| **Tuberculosis** (60) | 4 | 0 | 0 | **49** | 7 |

| class | recall (over served) | precision | recall BEFORE (§4.2, live HTTP) |
|---|---|---|---|
| COVID-19 | 0.900 | 0.871 | 0.954 |
| Normal | 0.900 | 0.947 | 0.725 |
| Pneumonia | 0.950 | 0.934 | 0.920 |
| Tuberculosis | 0.925 | 0.925 | 0.652 |
| **overall** | **0.9185** (233 served) | | ≈0.78 est. |

Counting the 7 OOD rejections as failures: 0.8917. The live HTTP number (0.9185) agrees with the offline
serving-path simulation (0.9208) to within 0.2 pt, confirming the simulation was faithful. The dominant
false-COVID attractor is gone: COVID-19 is now predicted for 62/233 served images (27%), down from
350/436 (80%).

### 3.4 Tuberculosis recall — dataset and external

| source | n | served | correct | recall |
|---|---|---|---|---|
| `data/xray/test/Tuberculosis` (seed-3 60, live HTTP) | 60 | 53 | 49 | **0.925** over served / 0.817 counting the 7 OOD rejections (was **0.652**) |
| `data/xray/test/Tuberculosis` (originals, offline serving path) | 60 | 60 | 57 | **0.950** (was 0.867) |
| `eval_run/external_samples/xray/tb/` (live HTTP) | 10 | 6 | 6 | **1.000** over served / 0.600 counting rejections |

The 4 external-TB rejections are HTTP **400** from the colour-content validator
(*"This image has color content that a real X-ray or CT scan wouldn't have"*) — those files carry the
burned-in colour tint documented in `data/README.md`'s TB audit. That gate runs in
`_read_and_validate_image()` **before** any crop, so it is unchanged by this fix and behaved identically
before it.

### 3.5 OOD-detector side effects (`eval_run/results_cropfix_ood.json`)

The OOD check runs on the post-crop tensor, so the fix moves it. Mean feature-space distance from the
training centroid (threshold 8.72):

| class | before | after |
|---|---|---|
| COVID-19 | 5.028 | **4.878** |
| Normal | 3.351 | **2.910** |
| Pneumonia | 4.729 | **4.425** |
| Tuberculosis | 6.212 | 6.293 |

Rejections over the 240 sample: **5 → 7**, all Tuberculosis. Three of four classes moved *closer* to the
training distribution, as expected. TB is flat and remains the class nearest the threshold — that is the
pre-existing **N3** finding, not something this fix introduced, but it does cost 2 extra TB rejections on
60 images while fixing 15 TB misclassifications on the same set.

### 3.6 Regression checks

- **pytest**: `26 passed, 3 warnings in 61.53s` — unchanged from before the fix.
- **Startup reproduction check** (fresh backend, isolated scratch DB, port 8077) — **all 6 pass**:

| checkpoint | claims | reproduced | diff |
|---|---|---|---|
| xray (`xray_txrv_densenet121_best.pt`) | 0.9646 | 0.9646 | 0.0000 |
| ct | 0.9904 | 0.9904 | 0.0000 |
| gate (`modality_gate_best_model_v2.pt`) | 1.0000 | 1.0000 | 0.0000 |
| xray_sanity | 0.9996 | 0.9996 | 0.0000 |
| ct_sanity | 1.0000 | 1.0000 | 0.0000 |
| seg (U-Net) | dice 0.9890 / iou 0.9786 | dice 0.9830 / iou 0.9671 | within tolerance |

Log: `eval_run/cropfix_backend_startup.log`. The server was stopped after the run; no backend process
was left running.

---

## 4. Verdict — honest

**Substantially fixed, not fully closed.**

- On **original uploads** — the realistic case, what a real user sends — live accuracy went
  **0.883 → 0.950** against a documented **0.9657**. The gap narrowed from 8.2 pt to **1.6 pt**. That is
  genuinely close, not a marginal tick.
- On the **pre-cropped `data/xray/test/` split** — the double-crop framing the §4.3 measurement used —
  live went **0.800 → 0.921** against the offline **0.967**. That closes **12.1 of the 16.7 pt** gap;
  **4.6 pt remain**. Over real HTTP the same sample gives **0.9185** (0.8917 if OOD rejections count as
  failures). So on that specific comparison, live accuracy **approaches but does not reach** 96.57%.
- The headline failure mode is resolved: false-COVID-19 predictions fell from 80% of served images to
  27%, and every class recall is now ≥0.90 over served images (it was 0.652 for TB, 0.725 for Normal).

### What is still unresolved

1. **The class-conditional crop mismatch cannot be fixed at serving time.** Tuberculosis and the
   tbdb-sourced half of Normal were trained *only* on the uniform 8% border crop, because the TB source
   has no segmentation masks; COVID-19 and Pneumonia were trained *only* on the tight mask crop. Live
   inference has no signal to tell which framing an upload "should" get. This is the residual ~4.6 pt and
   the remaining 3 TB → COVID-19 errors. The real fix is a data fix — rebuild `data/xray/` with a
   consistent crop policy across all four classes (e.g. run the U-Net over the TB source too) and
   retrain — which is out of scope here since it changes weights.
2. **N3 (OOD rejecting genuine TB test images) is untouched and marginally worse**: 5 → 7 rejections on
   240 images. The threshold (99th percentile over training images) is calibrated on a distribution TB
   sits at the edge of.
3. **The external-TB colour gate** rejects 4/10 samples at HTTP 400. Pre-existing, unrelated to the crop,
   arguably correct behaviour given the documented TB-source tint artifact.
4. `used_segmentation` is still discarded at the call site in `main.py` (**N4**) — deliberately left
   alone, since fixing it means editing `main.py`, which this task was asked to keep otherwise intact.

### Confirmations

- **No model weights or checkpoints were touched.** `git status --porcelain checkpoints/` is empty.
  The only ML artifact read was for evaluation.
- **The pre-existing uncommitted `backend/main.py` changes are fully intact.** `main.py` was not edited
  at all this session (0 edits; the file is byte-identical to `eval_run/xray_cropfix_backups/main.py.orig`).
  Verified still present: the gate-reordering (`compute_lung_crop` runs *after* the sanity/modality gates),
  the `preprocess_image_xrv` / `predict_with_heatmap_xrv` backbone dispatch via `is_xrv_backbone()`, the
  `xray_txrv_densenet121_best.pt` / `modality_gate_best_model_v2.pt` checkpoint paths, the modality-gate
  confidence guardrail and the `logs/modality_gate_debug.log` instrumentation. `git diff --stat` still
  shows all 78 changed lines in `main.py`.
- The `weights_only=False → True` change visible in `git diff backend/seg_utils.py` is **pre-existing**
  (it was already uncommitted at session start), not part of this fix.

---

## 5. Artifacts

| path | what |
|---|---|
| `eval_run/xray_cropfix_backups/cropdiff_*.png` | side-by-side train-crop vs live-crop vs what-the-model-sees, 11 images |
| `eval_run/xray_cropfix_backups/cropdiff_meta.json` | measured crop dimensions per image |
| `eval_run/xray_cropfix_backups/*.orig` | pre-edit file backups |
| `eval_run/cropdiff_evidence.py` | generates the crop-diff PNGs |
| `eval_run/cropfix_variants.py` / `results_cropfix_variants.json` | the candidate-fix sweep |
| `eval_run/cropfix_expand_sweep.py` / `results_cropfix_expand.json` | bbox-expansion sweep (rejected) |
| `eval_run/run_pathdiff_after.py` / `results_pathdiff_after.json` | post-fix 240-image offline-vs-live |
| `eval_run/run_rawprobe_after.py` / `results_rawprobe_after.json` | post-fix original-upload probe |
| `eval_run/cropfix_live_run.py` / `results_cropfix_live.json` | post-fix live HTTP confusion matrix + external TB |
| `eval_run/cropfix_ood_check.py` / `results_cropfix_ood.json` | OOD-distance before/after |
| `eval_run/cropfix_backend_startup.log` | startup reproduction-check log |
