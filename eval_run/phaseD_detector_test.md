# Phase D (measured): MONAI `lung_nodule_ct_detection` run against Phase B/C's own LIDC volumes

**Scope: measure and report. The two-stage pipeline was NOT built** — that is Phase E, and the
constraint against starting it was honoured regardless of how the numbers came out.
`checkpoints/ct_best_model_phase35_v2.pt`, `backend/main.py`, `data/ct_scan/` and Phase B's
`eval_run/lidc_full/labels_manifest.csv` are untouched (re-hashed in §9). Everything written this
phase is confined to `eval_run/phaseD/`. One new Python dependency was installed: `nibabel`.

---

## Verdict up front

**The localization question is settled, and not narrowly. The detector's centre error is ~2.8 mm
against a ~13 mm tolerance — a 4.7× margin. The 3D-volume input requirement is real but is NOT the
blocker the survey feared: on this hardware it costs ~135 lines of new code, 0.69 GB of VRAM and
~35 s per scan. What the detector actually costs you is *recall*, not precision of placement.**

1. **Localization: PASS, comfortably.** Across 22 annotated test-split nodules in 10 patients,
   **20/22 were found and placed within 8 mm** at the chosen operating point, mean 3D centre error
   **2.75 mm**, median **2.45 mm**, **worst matched case 5.39 mm**. Phase C.1's `v2_jitter`
   classifier holds 0.783 nodule accuracy at 8.5 mm and 0.725 at 12.7 mm of displacement; **every
   matched detection lands inside the flat part of that curve**, not the tail. Even the worst
   (5.39 mm) sits below the 8.5 mm point where `v2` is still at 0.783.
2. **Detection rate: 20/22 (90.9%) at score ≥ 0.5, 21/22 (95.5%) at score ≥ 0.3.** The two failures
   are *misses*, not near-misses (§4): one nodule is detected at 2.17 mm but scores 0.467, just
   under the threshold; the other is genuinely not found. **There is no population of "found but
   badly placed" detections at any threshold tested.**
3. **False positives: 2.0 extra boxes per scan at score ≥ 0.5, and most are not errors.**
   17 of 20 unmatched detections sit **inside lung parenchyma**, 19 of 20 are **focally denser than
   their surroundings**, and **13 of 20 land within 4 mm of a LIDC reader ROI that Phase B's
   manifest filtered out** — i.e. they are genuine annotated nodules, not hallucinations. The
   unexplained rate is **0.70/scan**, and exactly one box across all 10 whole-chest scans is both
   outside lung tissue and unexplained.
4. **VRAM: not a blocker at the training patch size, and a hard one at the config default.** With
   the bundle's native 192×192×80 patch, peak allocation is **0.69 GB of 4.00 GB** on every one of
   the 10 volumes, and inference is **15–35 s**. With `configs/inference.json`'s shipped default of
   512×512×192 the peak is
   **9.85 GB** — it only completes because Windows' driver spills to shared system memory, at 4.4×
   the runtime. **The shipped config does not fit this machine; the training patch fits with room
   to spare and finds the same top box.** CPU fallback works too (211 s vs 15 s, ~14× slower,
   byte-identical detections) but was not needed.
5. **The integration verdict is the one thing that got *worse* on contact with reality, and it is
   not about the detector.** Loading a volume was easy (§6). What is not easy is that
   `/predict/ct` is not one classifier — it is a *chain of four single-2D-image models* (sanity
   gate, modality gate, OOD detector, classifier) plus Grad-CAM, every one of which is defined on a
   single PIL image. A volume endpoint does not extend that chain; it needs its own.

**A two-stage detector + Phase C classifier pipeline is feasible. It is not a research risk any
more — it is a scoped engineering job, and §7 says what it is.** It is still not a ~10-days-before-
an-exam job, and this phase does not recommend starting it now.

---

## 1. What was actually downloaded and run

**Model.** MONAI Model Zoo bundle `MONAI/lung_nodule_ct_detection` pulled from Hugging Face:
16 files, of which `models/model.pt` is **83.7 MB** (59 s). Apache 2.0, exactly as the survey said.

**The minimal working setup turned out to be genuinely minimal.** `monai` 1.6.0 was already
installed in this project's environment; `pydicom`, `scipy` and `numpy` too. The only package that
had to be added is **`nibabel` 5.4.2** (MONAI's `Orientation` transform imports it). No ITK, no
ignite training loop, no MONAI Label, no 3D Slicer. `models/model.pt` is a plain 190-tensor
`state_dict`, which loads with strict key matching into a network built from the four lines in
`configs/inference.json` that define it (`resnet50` 3D backbone → FPN feature extractor →
`RetinaNet` → `RetinaNetDetector`). `run_detector.py` is ~200 lines including argument parsing and
result serialisation.

**Data.** 12 series were selected from `labels_manifest.csv` rows with `split == "test"`
(`select_cases.py`, deterministic: prefer series carrying more annotated nodules, force ≥4 series
containing a Malignant nodule). **10 of the 12 completed downloading in the time available** — see
§8 — covering **22 annotated nodules (4 Malignant, 18 Benign, 4.03–18.59 mm)**. That meets the
brief's 10–15 case target.

| patient | slices | slice spacing | resampled shape | vendor | nodules | read + resample | inference |
|---|---|---|---|---|---|---|---|
| LIDC-IDRI-0657 | 413 | 0.62 mm | 512×512×207 | GE | 1 M | 15.3 s | 28.4 s |
| LIDC-IDRI-0828 | 255 | 1.25 mm | 455×455×255 | GE | 1 M | 4.8 s | 16.0 s |
| LIDC-IDRI-0059 | 133 | 2.50 mm | 512×512×265 | GE | 1 M | 4.8 s | 34.8 s |
| LIDC-IDRI-0670 | 172 | 1.80 mm | 441×441×247 | Siemens | 1 M | 5.7 s | 15.1 s |
| LIDC-IDRI-0182 | 133 | 2.50 mm | 512×512×265 | GE | 3 B | 4.5 s | 33.4 s |
| LIDC-IDRI-0586 | 120 | 2.50 mm | 420×420×239 | GE | 3 B | 5.9 s | 15.3 s |
| LIDC-IDRI-0933 | 179 | 1.80 mm | 484×484×257 | Siemens | 3 B | 4.2 s | 33.6 s |
| LIDC-IDRI-0648 | 764 | 0.50 mm | 471×471×306 | Siemens | 3 B | 16.2 s | 19.9 s |
| LIDC-IDRI-0433 | 305 | 1.00 mm | 607×607×244 | Philips | 3 B | 13.0 s | 27.7 s |
| LIDC-IDRI-0321 | 265 | 1.25 mm | 484×484×265 | GE | 3 B | 7.4 s | 33.7 s |

The sample is not homogeneous — **3 vendors, slice spacing 0.50–2.50 mm, 120–764 slices per
series** — and the detector's accuracy did not track any of those. Read+resample time tracks raw
slice count; inference time tracks **the number of sliding windows**, i.e. the *resampled* volume
shape, which is why the 764-slice case is among the fastest to infer on. At 192×192×80 with 0.25
overlap the observed cost is **≈0.44 s per window** (0059: 80 windows/34.8 s; 0433: 64/27.7;
0648: 45/19.9) — a usefully predictable number for a Phase E latency budget.

**Preprocessing mirrors the bundle exactly**: RAS orientation, resample to 0.703×0.703×1.25 mm,
`ScaleIntensityRange(-1024, 300) → [0, 1]`. Box selector parameters are the bundle's
(`score_thresh=0.02`, `nms_thresh=0.22`, `topk=1000`, `detections_per_img=300`); nothing was tuned.

**Determinism.** The bundle's docs warn it is non-deterministic (a max-pooling layer). Run twice on
LIDC-IDRI-0182 on this GPU, the output was **identical**: same box count, max |score difference|
**0.0**, max |centre difference| **0.0 mm**. Reported because it was checked, not assumed.

---

## 2. VRAM — the finding the brief specifically asked for

RTX 3050 Laptop, 4095 MB total, confirmed by `torch.cuda.get_device_properties`.

| sliding-window ROI | peak allocated | wall time (LIDC-0059) | top detection |
|---|---|---|---|
| **192×192×80** (the bundle's *training* patch) | **0.69 GB** | **33 s** | 0.989 @ 2.71 mm |
| 512×512×192 (`configs/inference.json` default) | **9.85 GB** | 144 s | 0.989 @ 2.71 mm |
| 192×192×80 on **CPU** (LIDC-0586) | — | 211 s (vs 15 s GPU) | identical output |

**The shipped inference config does not fit a 4 GB card.** It did not raise `CUDA out of memory`
only because the Windows NVIDIA driver silently falls back to shared system memory — which is why
it "worked" at 4.4× the runtime. That is a trap worth naming: a naive run of `python -m monai.bundle
run` on this machine would appear to succeed while paging through host RAM.

**Dropping to the native training patch fixes it completely and costs nothing measurable.** Peak
falls 14×, to 17% of the card, and the top box is identical to three decimal places. Inference was
**15–35 s per volume across all 10 cases, at an unvarying 0.69 GB peak**, plus **4–16 s** to read
and resample the DICOM series. CPU fallback exists and produces byte-identical detections, so a
GPU-less deployment is possible at ~3.5 min/scan — but it was not needed here.

---

## 3. Coordinate conversion — validated before it was trusted

Phase A was burned once by an axis convention, so the manifest-pixel → world-mm conversion was
tested against real data twice before any error number was computed.

**Check 1 — Phase B's own contour-vs-annulus HU test** (`sanity_check_coords.py`). Convert each
annotation to RAS mm, map into the preprocessed volume, compare mean HU inside the lesion sphere
against a surrounding annulus. Run at the correct position and at the deliberately **row/col-swapped**
position:

| | contour − annulus HU (mean / median) | positive |
|---|---|---|
| **correct convention** | **+295.6 / +212.3** | **22/22** |
| row/col swapped | +0.1 / −4.0 | 10/22 |

**Check 2 — correlate against the actual Phase B crop PNGs** (`sanity_check_png.py`). Cut the 28 mm
axial patch the manifest position implies out of the preprocessed volume, window it identically
(WL −600 / WW 1500), and correlate it with the crop Phase B actually wrote for that nodule. All 8
dihedral orientations are scored so the flip relationship is measured, not assumed, and a null
distribution of 200 random in-lung patches per nodule says what a wrong location looks like.

**Mean best-variant r = 0.882 over all 22 nodules (min 0.709, several ≥ 0.98, one at 1.000), against
a null p95 of 0.08–0.63.** The winning orientation is `transpose + reverse both axes` on **20 of
22** — exactly what LPS→RAS predicts (DICOM +column → Left → −x in RAS). The two exceptions are
near-symmetric patches where two orientations tie within a few hundredths; they are not a different
geometry.

The conversion is right. The mm numbers below rest on it.

---

## 4. Localization accuracy — the headline

3D Euclidean distance in mm between the detector's predicted box centre (converted to RAS world
coordinates by the preprocessed volume's own affine) and the annotation centre. **3D, not in-plane**
— both were available, and the in-plane component is reported separately below because it is the
one that matters for a 2D crop.

**22 annotated nodules, 10 patients, score threshold 0.5:**

| metric | value |
|---|---|
| **detection rate ≤ 8 mm** | **20/22 (90.9%)** |
| detection rate ≤ 5 mm | 19/22 (86.4%) |
| **mean 3D centre error, matched** | **2.75 mm** |
| **median** | **2.45 mm** |
| p90 / **max** | 3.92 / **5.39 mm** |
| mean in-plane (x,y) error | 2.27 mm |
| mean z error | 1.33 mm |

Full sweep over score thresholds (rate = fraction of all 22 nodules found within that radius;
error statistics over matches within 20 mm):

| score | boxes/scan | ≤5 mm | ≤8 mm | ≤13 mm | ≤20 mm | ≤30 mm | mean err | median | max |
|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 7.30 | 0.909 | **0.955** | 0.955 | 0.955 | 1.000 | 2.60 | 2.32 | 5.39 |
| 0.20 | 5.70 | 0.909 | **0.955** | 0.955 | 0.955 | 1.000 | 2.70 | 2.32 | 5.39 |
| **0.30** | 4.80 | 0.909 | **0.955** | 0.955 | 0.955 | 1.000 | **2.72** | **2.41** | **5.39** |
| **0.50** | **4.00** | 0.864 | **0.909** | 0.909 | 0.909 | 0.955 | **2.75** | **2.45** | **5.39** |
| 0.70 | 3.30 | 0.773 | 0.818 | 0.818 | 0.818 | 0.864 | 2.78 | 2.45 | 5.39 |
| 0.90 | 2.20 | 0.636 | 0.682 | 0.682 | 0.682 | 0.773 | 2.83 | 2.32 | 4.51 |

**The shape of that table is the most useful thing in this report after the mean error.** The
≤5 mm, ≤8 mm, ≤13 mm and ≤20 mm columns are *identical or near-identical at every threshold*, and
the worst matched error never exceeds 5.39 mm. **There is no population of "found but badly placed"
detections.** A nodule is either found and placed within ~5 mm, or not found at all. For a
two-stage pipeline that is a far better failure mode than a long tail of near-misses would have
been: the crop handed to the classifier is either right or absent, never subtly wrong.

**The two failures at score ≥ 0.5, examined individually:**

* **LIDC-IDRI-0433, 7.51 mm Benign — a threshold artifact, not a localization failure.** The
  detector *does* find it, at **2.17 mm**, but scores it **0.467** — just under 0.5. It reappears at
  score ≥ 0.3, which is why the ≤8 mm rate goes 0.909 → 0.955 there.
* **LIDC-IDRI-0648, 7.11 mm Benign — a genuine miss.** The nearest box at *any* score, down to the
  bundle's 0.02 floor, is 8.95 mm away and scores 0.022. Reading the crop pair for it in
  `figure_gt_vs_detection.png`: this nodule sits in a **cluster of adjacent nodules**, and the
  detector locked onto a neighbour 20 mm away instead. That is a plausible and specific failure
  mode, not noise.

Per nodule, at score ≥ 0.5 (misses in bold):

| patient | class | diam | 3D err | in-plane | z | box score |
|---|---|---|---|---|---|---|
| 0657 | M | 7.03 | 1.92 | 1.74 | 0.80 | 0.984 |
| 0828 | M | 8.75 | 2.66 | 1.93 | 1.83 | 0.989 |
| 0059 | M | 14.06 | 2.71 | 2.09 | 1.72 | 0.989 |
| 0670 | M | 10.29 | 3.05 | 2.28 | 2.01 | 0.607 |
| 0182 | B | 14.77 | **5.39** | 5.36 | 0.54 | 0.999 |
| 0182 | B | 7.03 | 1.98 | 1.90 | 0.59 | 0.670 |
| 0182 | B | 10.55 | 2.65 | 2.63 | 0.37 | 0.768 |
| 0586 | B | 6.91 | 2.49 | 1.91 | 1.60 | 0.837 |
| 0586 | B | 5.76 | 2.24 | 1.93 | 1.15 | 0.929 |
| 0586 | B | 4.03 | 1.89 | 1.78 | 0.63 | 0.940 |
| 0933 | B | 18.59 | 3.80 | 1.92 | 3.28 | 0.997 |
| 0933 | B | 7.97 | 3.28 | 1.89 | 2.67 | 0.995 |
| 0933 | B | 7.30 | 2.05 | 1.84 | 0.90 | 0.989 |
| **0648** | B | 7.11 | **20.30 (MISS)** | — | — | — |
| 0648 | B | 12.93 | 4.99 | — | — | 0.999 |
| 0648 | B | 5.82 | 2.30 | — | — | 0.914 |
| **0433** | B | 7.51 | **68.14 (miss @0.5; 2.17 @0.3)** | — | — | 0.467 |
| 0433 | B | 13.34 | 1.99 | — | — | 0.972 |
| 0433 | B | 8.34 | 2.21 | — | — | 0.992 |
| 0321 | B | 9.30 | 2.65 | — | — | 0.996 |
| 0321 | B | 7.30 | 2.32 | — | — | 0.956 |
| 0321 | B | 5.98 | 2.41 | — | — | 0.804 |

`figure_gt_vs_detection.png` renders, for all 22, the 28 mm crop at the ground-truth centre beside
the crop at the detector's centre. **The lesion is plainly present and near-centred in every
detector-centred crop for the 20 matched nodules**, including the 5.39 mm worst case; the two
failures are visible as obviously different anatomy.

### Against the ~13 mm tolerance: PASS, with a 4.7× margin

Phase C.1's measured tolerance curve for `ct_lidc_nodulecrop_v2_jitter.pt`:

| displacement | 0 mm | 4.2 mm | 8.5 mm | 12.7 mm | 15.6 mm |
|---|---|---|---|---|---|
| v2 nodule accuracy | 0.792 | 0.797 | 0.783 | **0.725** | 0.396 |

**Measured detector error on matched nodules: mean 2.75 mm, max 5.39 mm.** Every matched detection
lands between the 0 mm and 8.5 mm columns, where `v2` is flat at 0.78–0.80 — i.e. **in the region
where the classifier has not yet started paying for displacement at all**, not in the 12.7 mm
region where it is down to 0.725. The ~13 mm figure was the tolerance the classifier *needed*; the
detector beats it by a factor of nearly five, and beats even the tighter pre-jitter `v1` tolerance
(<4.2 mm) on 19 of the 20 matched nodules.

**The important consequence: relaxing localization further buys nothing.** Phase C.1 paid ~3 points
of centred accuracy for the jitter augmentation that widened tolerance from <4 mm to ~13 mm. On
this evidence the detector would have cleared the *un*-augmented `v1` requirement on 19 of 20 as
well. The bottleneck in a two-stage pipeline is **whether the detector fires at all**, not how well
it centres — which inverts the priority Phase C/C.1 assumed.

This is the number the survey could not produce from AP@IoU=0.1, and it is not borderline.

---

## 5. False positives — counted, then characterised

Raw counts at score ≥ 0.5: **40 detections across 10 scans (4.00/scan)** against 22 annotated
nodules, so **20 unmatched boxes (2.00/scan)**. Counting stops there only if you are willing to call
all 20 errors, and LIDC's annotation coverage does not support that — Phase B's manifest keeps only
clustered nodules that also cleared a lung-fraction floor, so a real nodule can easily be absent
from it. (The scale of that gap is worth seeing: these 10 series carry **9 to 121 reader ROIs
each**, against 1–3 nodules per series in Phase B's manifest.)

Each unmatched box was therefore checked three ways: is it inside lung parenchyma (Phase B's own
`lung_mask`, on the axial slice through the box centre); is it a focal density (Phase B's
contour-vs-annulus HU test); and is it near any LIDC reader ROI at all (`plan_full.json`'s `avoid`
list, which holds *every* reader ROI in the series, not just the ones the manifest kept).

| | count |
|---|---|
| unmatched detections at score ≥ 0.5 | **20** (2.00/scan) |
| …inside lung parenchyma | **17/20** |
| …focally denser than surroundings (HU diff +40 to +680) | **19/20** |
| …**within 4 mm of a LIDC reader ROI absent from Phase B's manifest** | **13/20** |
| **unexplained** | **7 (0.70/scan)** |

So **almost two-thirds of the "false positives" are demonstrably real annotated nodules that Phase
B filtered out** — 13 of 20, all within 4 mm of a reader ROI. Of the 7 unexplained, 6 are
lung-internal focal densities (mean lung fraction 0.94 in a 28 mm window, HU contrast +93 to +680)
that look like nodules no reader marked.

**Exactly one unmatched box in 10 scans is both outside lung tissue and unexplained** (LIDC-0321,
score 0.715, lung fraction 0.00, 78 mm from any reader ROI). The two other not-in-lung boxes both
sit within 1.5 mm of a reader ROI, i.e. they are juxtapleural nodules that Phase B's own lung mask
also excludes — the same class of lesion Phase C.1's loose-Benign follow-up was about.

So the answer to "does it flag noise or normal tissue as nodules on clean regions" is: **almost
never — one genuinely spurious box across 10 whole-chest scans.** (Lung-membership was only
computed for the 20 unmatched boxes; the 20 matched ones are on annotated nodules by construction.)

The honest caveat: without a radiologist, "plausible unannotated nodule" is an inference from
location and density, not a diagnosis. What can be stated without one is the negative — the
detector is essentially not firing on wrong anatomy — and the operating cost: **a two-stage pipeline
would hand the classifier ~4 crops per scan, of which ~3.3 correspond to something a reader
marked.** Note also that this rate is exactly what a classifier with a Normal class is *for*: Phase
C's "is there a nodule in this patch" head scores 0.931 on `v2_jitter`, so a handful of extra crops
per scan is a workload the second stage is already built to absorb.

---

## 6. The 3D/2D input mismatch, now that it has actually been done

### How much new code did the volume path actually take?

**Less than the survey implied.** The genuinely new volume-handling core is
**`volume_io.py`, 89 lines**, plus ~45 lines of affine/preprocess helpers in `run_detector.py` —
call it **~135 lines**. What those lines do that Phase B/C's single-slice path did not:

* read every `.dcm` in a directory rather than one file by SOPInstanceUID;
* sort slices along the true slice normal (`ImageOrientationPatient` cross product), never by
  filename or `InstanceNumber`;
* derive slice spacing from the median inter-slice step and record its standard deviation;
* build a 4×4 LPS affine from `ImageOrientationPatient` + `PixelSpacing` + `ImagePositionPatient`,
  then negate x and y for RAS;
* keep a SOP → z / SOP → `ImagePositionPatient` map so a 2D annotation can be lifted to 3D.

Everything else — `RescaleSlope`/`Intercept`, HU clipping, the lung window — is Phase B's code
unchanged. **The 3D step is a geometry problem, not an architecture problem, and it is a
day's work, not a rewrite.** The full phase came to 1,074 lines of Python including two downloaders,
two independent sanity checks, the analysis harness and the figure script.

### Is a volume endpoint a moderate addition or a restructure?

**Honest read after doing it: the detector is the easy half. The app is the hard half, and not for
the reason the survey guessed.**

The survey framed this as "the app accepts one image, a detector needs a volume." True but
incomplete. `POST /predict/ct` is not a thin wrapper around a classifier — `_run_prediction` runs a
**chain of four single-2D-image models before the classifier ever sees anything**:

1. `_read_and_validate_image` → a single PIL image;
2. a **sanity gate** (is this a scan at all, vs a photo/document/screenshot);
3. a **modality gate** (X-ray vs CT, resnet18);
4. a **feature-space OOD detector** (99th-percentile distance from the training centroid);
5. then the classifier, then Grad-CAM for the response.

Every one of those five is defined on a 224-ish 2D tensor. A DICOM series has no PIL image to hand
any of them, and the guardrails are not decoration — `backend/main.py`'s own comments record that
each was added after a specific tested failure. So a volume route cannot reuse the chain; it needs
its own equivalents (or an explicit, documented decision to run the existing gates on a
representative slice, which is a claim someone would have to justify).

**The upload/UI half is genuinely moderate.** A series upload is a zip or a multi-file form post,
and the frontend already has a file-upload flow to extend. The response shape changes from "one
prediction" to "a list of findings with 3D positions", which is a real but ordinary frontend change.

**So: not a fundamental restructure of the app, but not "add a detector step" either.** It is a
second, parallel prediction path — new endpoint, new input validation, new guardrails, new response
schema, new result view — that happens to share the classifier weights with the existing one. The
existing `/predict/ct` would stay exactly as it is.

---

## 7. Is a real two-stage pipeline feasible? Yes — and here is what Phase E would actually be

**Feasible, on the evidence measured here.** Every one of the four things that could have killed it
was tested and none did: localization error (2.75 mm vs 13 mm needed), detection rate (20/22 at
score 0.5, 21/22 at 0.3), false positives (2.0/scan, 13 of 20 demonstrably real nodules, exactly
one genuinely spurious box in 10 whole-chest scans), and hardware (0.69 GB of 4 GB, ~35 s/scan end to end).

What Phase E would concretely require, based on what this session actually observed rather than
what the survey guessed:

**Data flow.** `series (N DICOM files) → volume_io.read_series → MONAI preprocess (RAS, 0.703/0.703/
1.25 mm) → RetinaNetDetector sliding window @192×192×80 → boxes in RAS mm → for each box above 0.5,
convert the centre back to the ORIGINAL series' voxel frame and cut a 28 mm axial crop through
Phase B's existing `box_for`/`window`/`save_png` → v2_jitter classifier → per-finding
Benign/Malignant/Normal.` Note the crop must come from the **original** slice geometry, not the
resampled volume, or the crops stop being distributed like Phase B's training data — the
inverse-affine helper for that already exists in `run_detector.py`.

**Endpoint shape.** A new `POST /predict/ct/series` accepting a zip or multi-file upload; returns a
list of findings (`{center_mm, box_mm, detector_score, class_probabilities, crop_png}`), not a
single prediction. `/predict/ct` unchanged.

**New guardrails needed** (the single biggest item, per §6): a series-level equivalent of the
sanity gate, modality gate and OOD check. Cheapest defensible version is a set of structural
assertions on the DICOM series itself (Modality == CT, consistent `ImageOrientationPatient`, plausible
slice count and spacing, HU range) plus running the existing 2D gates on a sampled mid-lung slice —
but that is a design decision that needs writing down, not assuming.

**Latency, from measured numbers.** 4–16 s to read and resample the series (dominated by DICOM
parsing, single-threaded), 15–35 s detector inference on this GPU, plus ~1 s for a handful of 2D
crop classifications. **20–50 s per scan on this laptop GPU (~35 s typical); ~3.5 min on CPU.** That is far outside
the current endpoint's request/response budget, so Phase E needs a job/polling shape, not a
synchronous POST — worth knowing before designing the API.

**Model weight cost.** +83.7 MB for the detector, +`nibabel`. `monai` is already installed.

**Three things this phase learned that the survey could not have predicted:**
* The bundle's own inference config is unusable on 4 GB and silently degrades rather than failing —
  Phase E must pin `roi_size=[192,192,80]` explicitly.
* The detector's errors are bimodal, not a smooth tail (§4). **Phase E's failure mode is "missed
  nodule", not "badly-centred crop"** — so effort belongs on recall and the score threshold, not on
  making the classifier more translation-tolerant than `v2_jitter` already is. One of the two
  failures here was purely a threshold choice (0.467 vs a 0.5 cut).
* Detector inference costs a predictable **≈0.44 s per sliding window** on this GPU, and window
  count is set by the *resampled* shape — so latency is estimable per series before running it.

### What this does not show

* **10 patients / 22 nodules is a small sample**, and they are LIDC — the same source distribution
  the detector was trained on (LUNA16 is LIDC-derived). **This measures whether the detector can be
  wired to the classifier, not how it behaves on an unseen scanner or population.** Nothing here
  licenses a generalization claim.
* **The nodule mix is 18 Benign to 4 Malignant.** Selection favoured series carrying several
  annotated nodules, and multi-nodule series in this test split are overwhelmingly Benign. Nothing
  here says the detector is equally good at either — though all 4 Malignant nodules were found, at
  1.92–3.05 mm.
* The 2 remaining selected series did not finish downloading in time (§8). They were selected before
  any results were seen, so nothing was cherry-picked.
* **No end-to-end accuracy number exists.** The classifier was never run on a detector-proposed
  crop this session, by design. "Detector finds it within 2.75 mm" plus "classifier is 0.79 accurate
  at 0 mm and 0.78 at 8.5 mm" is not the same as a measured pipeline number, and should not be
  quoted as one.
* False-positive characterisation is anatomical, not clinical. §5 says why.
* Phase A's deployment no-go and the standing advice not to cite the live model's external-Malignant
  12/12 as generalization evidence are both untouched by this phase.

### Recommendation

**Do not start Phase E now.** The exam deadline argument from Phase C/C.1 is unchanged, and this
phase's own §6 adds work (the guardrail chain) that the survey had not accounted for. But the
investigation should **not** be closed as infeasible — that was the live question, and the answer
measured today is that the detector half works, on this hardware, at this accuracy, with this much
code. **Phase D closes as: feasible, costed, deferred.**

---

## 8. Download reality — reported plainly, per the brief

This was the only part of the phase that went badly, and it is the reason the sample is 10 and not
12.

* **`getImage` (whole-series zip)** worked for the first four series — 133 sl/34 MB/100 s;
  413 sl/120 MB/578 s; 133 sl/33 MB/70 s; 120 sl/35 MB/110 s — and then **made no visible progress
  for over an hour on five concurrent requests, with no error and no timeout firing** (the socket
  timeout never trips because bytes trickle in). Killed.
* Switched to **`getSOPInstanceUIDs` + parallel `getSingleImage` (8 workers)**, which is exactly
  Phase A/B's validated route issued for every slice rather than only the annotated ones. It
  recovered the remaining six cases one series at a time: 91–402 MB in 158–882 s each, no failures.
* Throughput varied by an order of magnitude over the session (~60 KB/s at worst, ~1 MB/s at best),
  so per-series wall time was **2.5–15 minutes**, unpredictably. That is the constraint, not the API.
* **Per-case size is 33–402 MB; 2,739 slices / 1,243 MB for all 10.** A few hundred MB per case, not
  multi-GB — the problem was throughput, not volume.

Downloading was stopped after the 10th series rather than waiting out the last two, per the brief's
instruction to stop rather than grind; the partial 11th was deleted. **All 12 cases were selected
before any result was seen** (`select_cases.py` is deterministic and was run first), so nothing was
cherry-picked, and 10 meets the brief's 10–15 target. The remaining two would have to behave
completely unlike the first ten to move a verdict resting on a 4.7× margin.

---

## 9. Live-system integrity

Re-hashed after all inference, not asserted:

```
checkpoints/ct_best_model_phase35_v2.pt      sha=93b3a979884edbc4  mtime=1787555730.4296012
backend/main.py                              sha=780120553178c0a5  mtime=1788684518.831581
eval_run/lidc_full/labels_manifest.csv       sha=95d734f64123a46f  3461 rows, unmodified
checkpoints/ct_lidc_nodulecrop_v1.pt         sha=bdcbd6b2e5312435  unmodified
checkpoints/ct_lidc_nodulecrop_v2_jitter.pt  sha=5d60055e02974e91  unmodified
data/ct_scan/                                n=1036  newest mtime 2026-08-23 20:53:00
```

The first two match the values recorded in `phaseC1_followups.md` §7 and `phaseC_lidc_training.md`
§8 exactly. **No checkpoint was written this phase.** Every write is confined to `eval_run/phaseD/`.
One environment change outside the repo: `pip install nibabel`.

**Disk added: ~1.5 GB, all under `eval_run/phaseD/` — 1.4 GB of it the downloaded DICOM series
(`_work/volumes/`, 2,739 slices) plus the 80 MB bundle.** The DICOM volumes are pure cache: delete
`_work/volumes/` and every result in this report still stands (`detections.json`, `analysis.json`
and the figure are all derived and stored).

The project's own `uvicorn main:app --reload` dev server was running before this phase started and
was not touched. **No process was left running** — both downloaders and the completion monitor were
stopped explicitly before this report was finalised.

---

## 10. Artifacts

All under `eval_run/phaseD/`. Nothing is referenced by `backend/`.

| path | what |
|---|---|
| `bundle/` | the MONAI `lung_nodule_ct_detection` bundle as downloaded (weights 83.7 MB, Apache 2.0) |
| `select_cases.py`, `_work/cases.json` | §1 deterministic test-split case selection |
| `download_volumes.py` | the `getImage` whole-series-zip route (worked for 4 series, then stalled — §8; killed before it wrote its summary log, so per-series `_complete.json` files under `_work/volumes/` are the record) |
| `download_volumes_perslice.py`, `_work/download_log_perslice.json` | the `getSingleImage`-per-slice route that finished the job |
| **`volume_io.py`** | **the 89-line DICOM-series → HU volume + RAS affine core (§6)** |
| `run_detector.py`, `_work/detections.json` | bundle network/detector built from the config, inference, box→world conversion |
| `sanity_check_coords.py`, `_work/coord_sanity.json` | §3 check 1 — contour-vs-annulus HU, correct vs row/col-swapped |
| `sanity_check_png.py`, `_work/png_sanity.json` | §3 check 2 — correlation against Phase B's own crop PNGs + null distribution |
| `analyze.py`, `_work/analysis.json` | §4/§5 — mm errors, detection rates, FP characterisation against `plan_full.json`'s full reader-ROI list |
| `make_figure.py`, `_work/figure_gt_vs_detection.png` | §4 — GT crop vs detector-centred crop for all 22 nodules |
| `_work/det_smoke_nativeroi.json`, `_work/det_cpu_0586.json`, `_work/det_repeat{1,2}.json` | §2 — the 512×512×192 VRAM test, the CPU fallback, the determinism check |
| `_work/volumes/` | the 10 downloaded DICOM series (cache, 1.4 GB, deletable) |

**Phase E was not started.**
