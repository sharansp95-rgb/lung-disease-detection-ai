# Phase D (survey only): candidate lung-nodule detectors

Research and documentation only. **No model was downloaded, installed, or run this session.**
No code changes, no live-system changes. This closes with a recommendation for what the next
session should test first, not a working pipeline.

## Context: what "integration" actually requires here

This project's live `/predict/ct` endpoint (`backend/main.py`) accepts one uploaded 2D image and
returns one prediction. `ct_lidc_nodulecrop_v1.pt`/`v2_jitter` classify a small 2D crop and, per
Phase C.1, now tolerate roughly ±13mm of localization error before accuracy collapses. A detector
would need to take an incoming scan and propose where to center that crop.

**The single biggest finding of this survey, true of every well-documented candidate below: none
of them take a single 2D image as input.** They all operate on full 3D CT volumes (a whole DICOM
series, dozens to hundreds of slices). Fitting any of them into this project's current
single-image upload flow is not "add a detector step" — it's a change to what the app accepts as
input in the first place (a volume/series upload, not a 2D image), which is a materially bigger
integration than the survey's "effort estimate" column implies for the detector itself.

**On the ~13mm tolerance comparison:** the two best-documented candidates below report accuracy as
mAP/AP at IoU=0.1 (a loose bounding-box-overlap threshold), not a mean localization error in mm.
These are not directly convertible without knowing the specific box sizes involved (IoU depends on
both boxes' dimensions, not just center offset) — stating an equivalent mm figure would be a
manufactured number, so this survey does not do that. What can be said: IoU=0.1 is a deliberately
loose criterion (an IoU that low tolerates large center offsets, especially for LUNA16's mostly
3-10mm nodules), so these models plausibly clear a 13mm tolerance, but confirming that requires
actually running one against real center-offset data — exactly the next session's job.

## Candidates

### 1. MONAI `lung_nodule_ct_detection` (Project MONAI official model zoo)
- **Architecture:** 3D RetinaNet (focal loss one-stage detector, Lin et al. 2017), based on the
  nnDetection LUNA16 approach with modifications.
- **Training data:** LUNA16 (LIDC-IDRI derived), official fold 0 of the 10-fold challenge split.
- **Reported accuracy:** mAP 0.852, mAR 0.998, AP(IoU=0.1) 0.858, AR(IoU=0.1) 1.0 (COCO-style
  metrics). No mm-based localization error reported.
- **Input:** single-channel 3D CT patches, 192×192×80 voxels, resampled to 0.703×0.703×1.25mm
  spacing. Ground-truth/output boxes in world (mm) coordinates.
- **License:** Apache 2.0 — permissive, no blocker for any use.
- **Weights:** directly downloadable from the MONAI Model Zoo / Hugging Face
  (`MONAI/lung_nodule_ct_detection`) — no training required to get a working model.
- **Known caveats (from its own docs):** non-deterministic due to a max-pooling layer (exact
  reproduction across runs not guaranteed); coordinate-system handling needs care
  (`affine_lps_to_ras` flag) — the kind of coordinate-convention issue this investigation has
  already been bitten by once (Phase A's xCoord axis check) and should not assume away.
- **Integration effort:** low for "get a model running and see if it detects nodules on a sample
  volume" (weights ready, permissive license, official docs). High for "make this the localizer
  behind `/predict/ct`" (3D volume input requirement, as above; also a heavier runtime dependency —
  full MONAI framework — than this project currently carries).

### 2. SANet (Slice-Aware Network, IEEE TPAMI 2021)
- **Architecture:** 3D region-proposal network with a slice-grouped non-local (SGNL) module plus a
  false-positive-reduction stage. Despite the "slice-aware" name, it processes full 3D volumes with
  slice-context awareness, not standalone 2D slices.
- **Training data:** the authors' own **PN9** dataset (8,798 scans / 40,439 annotated nodules) —
  **not LUNA16**. A different source distribution than everything else in this investigation.
- **Reported accuracy:** paper reports results on PN9; no CPM/FROC numbers surfaced in the repo
  itself for a quick comparison.
- **License:** **non-commercial research use only (with GPL-3.0 code)** — a real blocker if this
  project's use is anything beyond personal/academic (worth checking your actual use case against
  this before investing time here).
- **Weights:** pretrained weights available via direct download (Google Drive link in repo).
- **Integration effort:** similar 3D-volume mismatch as MONAI's candidate, plus the licensing
  restriction and a different (non-LUNA16) training distribution than the rest of this project's
  work — lower priority unless the license is acceptable for your use.

### 3. nnDetection (self-configuring medical detection framework)
- **What it is:** not a single pretrained model but a framework that auto-configures a detection
  architecture per task; reported to outperform prior LUNA16 approaches when trained on it.
- **Training data / weights:** motivated by and validated on LUNA16, but functions as a
  self-configuring pipeline — it does not appear to ship ready-to-use LUNA16 weights the way
  MONAI's bundle does; using it means running its own training/configuration process on LUNA16 (or
  another dataset) yourself.
- **Integration effort:** highest of the three — no quick "download and test," real training-run
  investment before you'd know if it's even competitive for this use case. Not a realistic
  next-session pick given the deadline; noted for completeness/future reference only.

### 4. Smaller/unreviewed GitHub repositories (noted, not recommended first)
A few individual/hobby repositories (e.g. `Karim-Sleiman/Lung-Nodule-Detection`,
`Fazil-kagdi/lung-nodule-segmentation-classification`) claim 2D-slice-oriented pipelines with
downloadable weights — which would be the best architectural fit if their claims hold up, since
they'd avoid the 3D-volume mismatch above. However: no peer review, no reported accuracy metrics
found in a search-only pass, and no way to assess reliability without actually running them.
**Worth a quick look precisely because they might sidestep the 3D-volume problem**, but rank below
the documented options until someone actually inspects the code and any reported numbers directly.

## Ranked recommendation

1. **MONAI `lung_nodule_ct_detection`** — best-documented, permissive license, weights ready now,
   trained on the same LUNA16/LIDC-IDRI lineage this whole investigation has used. **Test this
   first.** Concretely: download the bundle, run it against a handful of Phase B/C's own LIDC
   volumes (ground truth nodule locations already known from the manifest), and measure actual
   center-offset error in mm directly against real data — don't trust the IoU number as a proxy.
2. Quickly inspect the small 2D-oriented repos (#4) alongside #1 — if either has a working 2D
   approach with any credible reported accuracy, it's worth a look specifically because it avoids
   the volume-vs-slice architecture mismatch that every serious candidate otherwise shares.
3. SANet — only if the non-commercial license is acceptable for this project's actual use; the PN9
   training data and licensing are both friction relative to candidate 1.
4. nnDetection — not realistic before the exam deadline; requires a real training investment
   before it's even known to be competitive. Revisit later if the detector approach is still being
   pursued after other options are exhausted.

## Explicit scope note

**No model was downloaded, installed, or executed this session.** This is a documentation-only
survey, per the task's constraint to keep this fast and low-compute given limited session usage.
Testing candidate 1 against real data is deferred to a future session as a distinct next step, not
started here.
