# Phase B: LIDC-IDRI nodule-crop dataset — full-scale extraction

**Scope: extraction and data quality only.** No training, no model changes, no writes
outside `eval_run/lidc_full/`. `checkpoints/ct_best_model_phase35_v2.pt`,
`backend/main.py` and `data/ct_scan/` were not touched (verified at the end, §9).

Scales Phase A's 30-case / 107-crop feasibility check
([`phaseA_lidc_feasibility.md`](phaseA_lidc_feasibility.md)) to the whole collection,
and fixes the two defects Phase A found. Output: `eval_run/lidc_full/`.

## Verdict up front

**3,461 crops from 797 patients, patient-disjoint splits, zero overlap, ~35 minutes of
compute.** Phase A's cost model held: 21.1 min / 1.05 GB to download 1,999 slices
against a predicted "~25 min / 1.1 GB". Extraction was 81 s. The whole phase came in
*far* under Phase A's "~1 focused day" estimate — the time went into the two
defect-fixes, not the pipeline.

**Both Phase A defects were real, and one of them was worse than Phase A thought.**

- The label-mapping loss was *slightly better* at scale (38.4%, not 41.5%) and a
  cheaper alternative recovers most of it with **zero label flips** — a clean win, §2.
- The crop/lung-fraction defect was **worse**. Phase A's recommended ≥0.85 floor was
  calibrated on 30 cases; at full scale it rejects 30–51% of crops, and it rejects
  them **unevenly by class**. It had to be recalibrated against a control, not adopted
  as given. This is the main finding of the phase and is not a comfortable one — §3.

Coordinate handling re-verified at scale and reproduces Phase A almost exactly
(+391 HU, 386/400 nodules). 0 SOPInstanceUID mismatches in 1,997 slices.

---

## 1. What was built, and what was reused

Phase A's annotation download is *complete for all 1,010 cases* (the 9 MB XML zip), so
**nothing was re-downloaded except DICOM slices.** Phase A's 54 slices were seeded into
Phase B's cache rather than refetched.

| script | built on | what changed |
|---|---|---|
| `lidc_full/plan_full.py` | `parse_lidc_xml.py` + `plan_nodules.py` | all 1,018 series (no 30-patient sample, no ">=4 rated reads" prefilter); six label rules evaluated side by side; cluster distance measured to the running cluster centroid rather than to its first member |
| `lidc_full/download_slices_full.py` | `download_slices.py` | retries (3, backoff), atomic `.part` writes, Phase A cache seeding |
| `lidc_full/extract_crops_full.py` | `extract_crops.py` | per-slice streaming (2,000 slices will not fit in memory the way 54 did); crop size a swept parameter; hard lung-fraction floor on every class; contour-vs-annulus coordinate check |
| `lidc_full/lung_mask_diag.py` | new | diagnostic — is the low lung fraction a real chest-wall problem or a mask artifact? |
| `lidc_full/floor_calibration.py` | new | sets the floor against real distributions plus a chest-wall control |
| `lidc_full/make_splits.py` | new | patient-disjoint stratified splits + independent verification |
| `lidc_full/qc_contact_sheet.py` | `qc_contact_sheet.png` pattern | 30 crops (10/class) spread across the diameter range |

### Parsing coverage — Phase A's finding confirmed unchanged

1,318 annotation XMLs; **283 use the older `IdriReadMessage` root tag and are
skipped**; the remaining 1,035 still cover **1,018 / 1,018 CT series, 0 unmatched, 0
series without a usable annotation XML.** Identical to Phase A. The older-tag files are
redundant here, but a parser that does not handle or explicitly skip them will crash.

### Cross-reader clustering

No `noduleID` correspondence exists across reading sessions, so Phase A's greedy 3D
clustering at **≤10 mm centroid distance, at most one read per session per cluster** was
kept, with one documented improvement: distance is measured to the cluster's **running
centroid** instead of to whichever member was seen first, so a cluster's position is not
pinned by arrival order. In-plane distance still uses a nominal 0.7 mm/px — TCIA's
series metadata carries no `PixelSpacing`, so per-series spacing is not available before
the DICOMs are fetched.

**2,680 rated nodule clusters** across all 1,018 series. Reads per cluster:

| reads | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| clusters | 793 | 495 | 489 | 903 |

29.6% of clusters carry only a single radiologist's read. That matters for §2.

---

## 2. Defect 1 — label mapping. Six rules tried; the loss is partly recoverable

Phase A excluded 41.5% of nodules as ambiguous on 30 cases. At full scale the same rule
excludes **38.4%** — so Phase A's number extrapolated well, and the defect is real.

All six rules, on the same 2,680 clusters (`_work/label_mapping_study.json`):

| rule | Benign | Malignant | excluded | % excl | kept | malig % of kept | patients |
|---|---|---|---|---|---|---|---|
| **A** mean, dead band ≤2.5 / ≥3.5 *(Phase A's)* | 1131 | 521 | 1028 | **38.4** | 1652 | 31.5 | 725 |
| **B** mean split at 3 (<3 / >3) | 1374 | 665 | 641 | **23.9** | 2039 | 32.6 | 799 |
| **C** median, dead band ≤2.5 / ≥3.5 | 1159 | 565 | 956 | 35.7 | 1724 | 32.8 | 742 |
| **D** majority vote of discretized per-reader labels | 876 | 459 | 1345 | 50.2 | 1335 | 34.4 | 652 |
| **E** ≥2 readers agree, no reader dissents | 488 | 281 | 1911 | 71.3 | 769 | 36.5 | 469 |
| **F** as E, but a solo read labels its own nodule | 876 | 351 | 1453 | 54.2 | 1227 | 28.6 | 616 |

**The decisive result: zero disagreements.** On clusters that *both* rules label, every
alternative agrees with rule A on the class, in every case — 1652/1652 for B, 1609/1609
for C, 1222/1222 for D, 769/769 for E, 1227/1227 for F. The rules differ only in
**coverage**, never in what they call a nodule. So this is not a choice between
competing labellings; it is a pure strictness dial.

Notes on the alternatives that lost:

- **D (majority vote)** is the most principled-sounding reconciliation and is *stricter*
  than the mean, not looser — it excludes 50.2%, because a 4-reader panel splitting
  2 Benign / 1 Indeterminate / 1 Malignant is a tie. It buys nothing over A.
- **E / F (2-of-4 agreement)** collapse the dataset. E discards 71.3%, and structurally
  cannot label the 793 single-read clusters at all. F patches that by trusting a solo
  read — which is exactly the reader-discarding the panel exists to avoid, and it visibly
  skews the balance (malignant share drops to 28.6%).
- **C (median)** behaves almost identically to A. No reason to prefer it.

### Decision

**Extract the union of A and B (= B's 2,039 clusters), with rule A's label in
`label_strict` and an `is_borderline` flag on the 387 clusters only B commits to.**

Justification: B recovers 387 clusters (14.5 points of exclusion) at **zero risk of
relabelling anything A was confident about** — that is what the zero-disagreement result
buys. Those 387 are, by construction, the genuinely borderline ones (panel mean strictly
between 2.5 and 3.5), so they are exactly the label noise Phase A was right to be wary
of. Rather than choose for a downstream experiment that has not been run, both are in
the manifest: `is_borderline=0` reproduces Phase A's stricter dataset exactly, and
dropping the flag uses the larger one. Sensitivity to the choice is then a filter, not a
re-extraction.

**Still true, and not fixed by any of this:** 641 clusters (23.9%) remain unlabelled, so
even the inclusive set is biased toward the easier tail of the malignancy distribution;
and these are radiologists' *subjective impressions from the image*, **not
biopsy-confirmed pathology**. A model trained here learns what a radiologist panel
thinks looks malignant.

---

## 3. Defect 2 — crop size and the lung-fraction floor. Phase A's fix does not survive scale

Phase A measured 48 mm crops at lung fraction 0.79 / 0.73 / 0.81
(Malignant/Benign/Normal), 0.88 / 0.83 / **0.93** at 24 mm, and recommended tightening
to 24–32 mm plus **a hard floor at ≥0.85**.

### 3a. The sweep (600 slices, all sizes)

Lung fraction measured two ways: against the lung mask alone, and against
(lung ∪ the annotated lesion) — because the lesion is soft tissue and would otherwise
count as "non-lung" *in its own crop*, which would preferentially reject large nodules,
which are disproportionately malignant. That would be a class-correlated selection bias,
i.e. precisely the failure mode this whole investigation exists to remove.

| crop | Malignant lung / +lesion / %≥0.85 | Benign | Normal |
|---|---|---|---|
| 24 mm | 0.819 / 0.896 / **67.4%** | 0.817 / 0.827 / **53.7%** | 0.865 / 0.865 / **65.8%** |
| 28 mm | 0.810 / 0.871 / 62.7% | 0.796 / 0.804 / 49.9% | 0.836 / 0.836 / 58.6% |
| 32 mm | 0.798 / 0.847 / 57.0% | 0.777 / 0.782 / 45.7% | 0.816 / 0.816 / 53.9% |
| 40 mm | 0.770 / 0.804 / 43.5% | 0.737 / 0.741 / 37.9% | 0.795 / 0.795 / 48.6% |
| 48 mm | 0.739 / 0.764 / 37.0% | 0.700 / 0.702 / 28.9% | 0.746 / 0.746 / 39.6% |

**Phase A's 24 mm figures do not reproduce.** 0.88/0.83/0.93 on 30 cases becomes
0.82/0.82/0.87 on 600 slices. Applying ≥0.85 as given would discard **32.6% of
Malignant, 46.3% of Benign and 34.2% of Normal crops even at 24 mm** — and the 14-point
Malignant-vs-Benign gap is the class-correlated rejection the floor was supposed to help
avoid.

### 3b. Is the metric wrong, or the crops? (`lung_mask_diag.py`)

Before recalibrating I checked whether the mask was under-counting. Phase A's mask keeps
air-density (<−400 HU) components not touching the border and fills enclosed holes;
in-plane vessels connect back to the hilum, are never enclosed, and so read as
"not lung" even in a crop entirely inside the lung. Tested morphological closing at
radius 0/4/8/12/16 px, scoring crops centred on a lung pixel against control crops
centred on the chest wall:

| closing r | mask area | lung-site crop | %≥0.85 | **chest-wall crop** | %≥0.85 |
|---|---|---|---|---|---|
| 0 (Phase A's) | 0.2035 | 0.881 | 69.9 | **0.204** | **0.0** |
| 8 | 0.2070 | 0.888 | 71.9 | 0.242 | 3.6 |
| 16 | 0.2106 | 0.894 | 73.1 | 0.310 | 10.1 |

Closing recovers +1.3 points at best and starts eating the chest wall (control pass rate
0% → 10.1%). **The mask is not the problem and was left as Phase A had it.** The real
finding is the control column: the metric separates lung (0.881) from chest wall (0.204)
enormously well. The floor is a well-behaved instrument; 0.85 is simply the wrong
operating point on it.

### 3c. Calibration (`floor_calibration.py`, 697 slices)

Distributions are strongly bimodal — a median crop is ~0.99 (entirely intraparenchymal)
with a long lower tail of genuinely juxtapleural lesions. Chest-wall controls at 28 mm:
p25 = 0.010, **p75 = 0.373, max 0.753**.

| size | floor | keep M | keep B | keep N | M−B gap | **wall crops admitted** |
|---|---|---|---|---|---|---|
| 24 | 0.70 | 90.2 | 72.7 | 81.2 | 17.5 | 0.1% |
| **28** | **0.70** | **87.2** | **69.7** | **73.6** | **17.5** | **0.3%** |
| 28 | 0.80 | 73.5 | 55.1 | 63.6 | 18.4 | 0.0% |
| 28 | 0.85 | 69.7 | 49.3 | 58.5 | 20.4 | 0.0% |
| 32 | 0.70 | 83.3 | 63.0 | 70.5 | 20.3 | 0.3% |
| 32 | 0.85 | 65.0 | 43.2 | 55.5 | 21.8 | 0.0% |

### Decision: **28 mm crops, lung-fraction floor 0.70**

- **28 mm, not 24:** malignant nodules average **18.5 mm** diameter (p90 32.0, max 78.4);
  benign average 8.1 mm. A 24 mm box leaves the median malignant lesion essentially no
  parenchymal context — the crop becomes a picture of the inside of the mass. 28 mm is
  the midpoint of Phase A's recommended 24–32 mm band and retains more of the lesion's
  surroundings. 32 mm loses ~6 points of retention and widens the class gap.
- **0.70, not 0.85:** the wall control tops out at 0.753 with p75 = 0.373, so 0.70
  admits 0.3% of deliberately wall-centred crops — it does the job the floor exists for.
  Going from 0.70 to 0.85 buys that last 0.3% at a cost of ~20 points of retention on
  every class, and *widens* the M−B gap (17.5 → 20.4). Adopting 0.85 unchanged would
  have been strictly worse on both axes.
- **`lung_frac` is a manifest column**, so the strict-0.85 dataset is one filter away:
  **441 M / 673 B / 1,544 N = 2,658 crops**. 803 of the 3,461 crops (23%) sit in
  [0.70, 0.85).

### The part of this defect that is *not* fixed — stated plainly

**No lung-fraction floor is class-neutral.** At every size and every threshold tested,
Benign is rejected 12–22 points more often than Malignant. Two mechanisms, both real:
LIDC's benign nodules are smaller and more often subpleural, so their crops genuinely
touch the chest wall more; and the lesion-inclusion correction (§3a) mechanically
protects large lesions, which are disproportionately malignant — on the raw mask the
Malignant/Benign means are near-identical (0.810 vs 0.796) and the gap narrows to ~7
points, but the raw mask is wrong for the opposite reason. The two errors partly cancel
by coincidence, not by design.

The mitigation applied is to make the floor as loose as the wall control allows, since a
smaller rejection carries a smaller bias whatever its direction — and to record both
`lung_frac` and `lung_frac_excl_lesion` per row so the sensitivity is measurable rather
than assumed. **This remains an open, unresolved caveat on the dataset**, and it should
be re-examined before any Benign-vs-Malignant result from this data is believed.

---

## 4. Download and extraction at scale

**One representative slice per nodule cluster** — the ROI with the most contour points
across the cluster, i.e. the lesion's largest in-plane extent. The alternatives were
costed and rejected:

| granularity | slices | est. time | verdict |
|---|---|---|---|
| **representative slice per cluster** | **1,999** (2,614 for all rules) | **21 min / 1.05 GB** | **chosen** |
| every radiologist-contoured slice | 14,666 | ~2.7 h / ~7.7 GB | rejected |
| ±3 adjacent slices per nodule | ~14,700 | ~2.9 h / ~7.7 GB | rejected |

Reasons, and the first is specific to the crop-size decision above: at 28 mm, a slice ±3
away (LIDC slice thickness is commonly 2–2.5 mm, so up to ~7.5 mm off-plane) can miss a
5 mm nodule **entirely** — blind adjacent-slice sampling gets worse the tighter the crop,
so the two decisions are coupled. Second, all-contoured slices of one nodule are
near-duplicates of each other; `PROJECT_STATUS.md` §6 documents this project already
losing a result to near-duplicate inflation. Patient-level splits (§5) stop them leaking
*across* splits, but they would still inflate the apparent size of the dataset. Third,
2.7 h vs 21 min is real against the deadline in §7.

**Measured:** 1,999 requested, **1,997 ok, 2 failed**, 1,051 MB, **1,264 s = 21.1 min**,
1.54 slices/s at 6 workers. Phase A predicted ~25 min / 1.1 GB — accurate.

**Extraction:** 81 s for 1,999 slices → 3,461 crops, 128×128 PNG, lung window
WL −600 / WW 1500, physical 28 mm box sized per-slice from that slice's own
`PixelSpacing` (so crops are physically comparable across scanners and FOVs).

**Normal class:** unchanged from Phase A in design, because that design is the point —
one nodule-free crop per nodule slice, **from the same slice, same patient, same scanner,
same reconstruction, same crop size**, with a centre ≥40 mm from *every* annotated point
in the case (including unrated <3 mm nodules and non-nodule marks) and subject to the
same 0.70 floor. Normal and nodule crops therefore cannot differ by patient, scanner or
acquisition — the class↔acquisition correlation behind every failure in this
investigation is structurally unavailable, not merely controlled for.

---

## 5. Patient-disjoint splits

`PROJECT_STATUS.md` §6 records the previous instance of this bug in this project:
perceptual-hash/SSIM detection found **111 near-duplicate clusters, 47 of them split
across train/val/test** under a naive per-image split, with 35 of 109 test images having
a near-duplicate sibling in train — enough on its own to explain that model's 100% test
accuracy.

Split is by **LIDC-IDRI patient ID only**. Patients are placed whole, one at a time,
into whichever split is furthest below target for the classes that patient supplies
(rare classes weighted highest), so stratification never costs disjointness.

| split | patients | crops | Malignant | Benign | Normal | strict M / B / N |
|---|---|---|---|---|---|---|
| train | **410** | 2,450 | 412 | 648 | 1,390 | 332 / 539 / 1,390 |
| val | **195** | 506 | 75 | 133 | 298 | 63 / 107 / 298 |
| test | **192** | 505 | 75 | 132 | 298 | 59 / 108 / 298 |
| **total** | **797** | **3,461** | **562** | **913** | **1,986** | 454 / 754 / 1,986 |

**Zero patient overlap, verified twice and independently:** `make_splits.py` asserts it
from its own assignment map, and it was then re-derived from the written
`labels_manifest.csv` — `train∩val = 0`, `train∩test = 0`, `val∩test = 0`, and 0 patients
appearing in more than one split. Class proportions hold across splits (malignant share
16.8 / 14.8 / 14.9%).

One asymmetry worth naming: crops split 70.8 / 14.6 / 14.6 but patients split
51.4 / 24.5 / 24.1, because the greedy rare-class-first ordering sends multi-nodule
patients to train. No leakage results, but val and test are made of many
low-crop-count patients (~2.6 crops each vs train's 6.0).

Scanner spread across the whole set: GE 2,048 / Siemens 822 / Philips 328 / Toshiba 263.
Genuinely multi-vendor but **GE-dominated** — that is the collection's real composition;
Phase A's even 4-way spread was an artifact of its deliberate round-robin 30-case sample
and should not be expected at scale.

---

## 6. Quality control

### 6a. Coordinate sanity check, re-run on the new extraction

Phase A's **contour-vs-annulus** test (not the naive centre-vs-ring metric Phase A
flagged as an artifact — for a small nodule in a large crop the "ring" is dominated by
chest wall, and that metric reads as a false failure). A nodule is soft tissue inside
air-density parenchyma, so the correct axis reading must make the contour interior denser
than a thin annulus immediately around it. 400 nodules from the Phase B crops:

| interpretation | contour − annulus | median | positive | wins |
|---|---|---|---|---|
| **`xCoord` = column** (used) | **+391.4 HU** | +354.9 | 390/400 | **386/400** |
| `xCoord` = row | −0.6 HU | −1.7 | 191/400 | 14/400 |

Reproduces Phase A (+372 HU, 52/55) almost exactly at 7× the sample. Nothing broke at
scale. The extractor also hard-fails if this check ever chooses the row reading.

**Slice identity:** every file's `SOPInstanceUID` tag was re-read and compared to the UID
requested — **0 mismatches across 1,997 slices.** Filenames and slice ordering are used
nowhere in the pipeline.

### 6b. Visual inspection — 30 crops read directly

[`lidc_full/qc_contact_sheet.png`](lidc_full/qc_contact_sheet.png), 10 per class, nodule
classes sampled across the diameter range so a size-dependent failure would show. Red
circle = crop centre, where the nodule should be. These were opened and looked at, not
asserted.

**Honest read: 30/30 anatomically plausible, none malformed — no black crops, no
wrong-region crops, no coordinate-error signatures.** Specifics:

- **Malignant (10/10).** `0160 n0` (12.5 mm, 2|4|4|4) is a textbook bright round nodule
  dead centre with a dark halo; `0194 n0` (21.9 mm, 4|4|5|5) a large irregular mass;
  `0303 n0` (32.1 mm) fills essentially the whole crop — the visible cost of a 28 mm box
  on the largest lesions, correct but with no surrounding context left.
  **Two marginal:** `0756 n0` (26.2 mm, in-contour HU −589) and `0686 n16` (4.9 mm, HU
  −799) show only faint ground-glass density at the circle rather than a solid mass.
- **Benign (10/10).** Small, well-circumscribed bright nodules centred as expected —
  `0951 n4` (3.0 mm) is a tiny speck exactly on the circle, `0390 n0` (8.2 mm, 1|1|1) a
  clean round nodule. **One marginal:** `0246 n0` (11.7 mm, borderline, lung 0.743) has
  its brightest structure at the right edge rather than at the circle, consistent with a
  juxtapleural lesion sitting against the pleural band.
- **Normal (10/10).** Lung parenchyma with normal vessel branching, no mass, including
  one visibly noisy thin-slice/low-dose acquisition. **Three (`0148` 0.734, `0256`
  0.711, `0671` 0.799) show a visible pleural/chest-wall band** occupying maybe a fifth
  of the crop. That is the honest, visible cost of the 0.70 floor, and it is what a
  0.85 filter removes.

### 6c. In-contour HU

Mean in-contour HU **−336 (Malignant)** vs **−374 (Benign)**, crop-level **−750
(Normal)** — the density ordering real pathology gives, but a much weaker separation
than Phase A's −204 / −411 on 55 nodules. Phase A's sample was the easy tail of a small
draw; this is the population figure and should be preferred.

**86 of 1,475 nodule crops (5.8%) have in-contour HU below −700 HU**, i.e. air density.
All are small (3.6–6.3 mm). This is partial-volume / small-contour rasterization plus
genuine ground-glass lesions, not a coordinate bug — the §6a check is decisive against
that, and the rate is not class-correlated (5.7% of Malignant, 5.9% of Benign). It
affects a manifest statistic, not the crops themselves, and no filter keys on it.

---

## 7. Issues that did not show up in Phase A's 30 cases

1. **Phase A's ≥0.85 floor does not survive scale** (§3). The single largest finding
   here, and it cost most of the phase's time. Resolved by recalibrating against a
   chest-wall control. The class-correlated component of the rejection is **not**
   resolved — see the end of §3.
2. **Two slices are permanently un-downloadable.** SOP UIDs `...448378396789516014605561762604`
   and `...628619440608225619886544814747` return **HTTP 500** from `getSingleImage`,
   3 attempts each with backoff. Server-side, not rate limiting — every other request
   in the run succeeded and 0 needed a retry. **Unresolved**; the cost is 2 nodule crops
   and 2 Normal crops out of 3,461 (0.1%), so it was accepted rather than worked around.
3. **Filename collisions.** 8 patients contribute two CT series each, so Phase A's
   `<patient>_n<cluster>.png` naming is not unique at scale and would have silently
   overwritten crops *and* left duplicate manifest rows pointing at one file. Caught by
   inspection before the extraction run, fixed by adding a series suffix. Verified after
   the fact: 3,461 rows, 3,461 unique image paths.
4. **Memory.** Phase A loaded all 54 DICOMs into a dict; at 2,000 slices that is ~1 GB of
   resident pixel data. Restructured to stream per slice. Not a bug in Phase A, but its
   code does not scale as written.
5. **11 slices yielded no valid Normal site** (no lung location simultaneously ≥40 mm
   from every annotation and ≥0.70 lung fraction). Expected; those slices contribute
   nodule crops only.
6. **Scanner balance is not what Phase A's sample suggested** (§5). GE is 59% of crops.
   Phase A's even spread was a property of its sampling, not of LIDC-IDRI.

### Budget

The task flagged a ~10-day window before an exam and Phase A's "~1 focused day"
estimate. **This phase ran well under that** — ~35 minutes of compute (21 min download,
81 s extraction, the rest sweeps and calibration) inside a single session. Nothing took
dramatically longer than Phase A predicted; the download estimate was accurate to within
20%. The time that *was* spent went to §3, which was the right place to spend it.

**No training was performed and none should follow from this report without a separate
decision.** Phase A's split recommendation stands unchanged: this dataset answers an
investigation question, and it **cannot be served by `POST /predict/ct`**, which takes a
whole uploaded slice — deploying a nodule-crop classifier needs a nodule detector or a
user-drawn ROI that does not exist. Keep `ct_best_model_phase35_v2.pt` live.

---

## 8. Artifacts

All under `eval_run/lidc_full/`. Nothing referenced by `backend/`.

| path | what |
|---|---|
| `plan_full.py` | full parse, clustering, 6-rule label study, download plan |
| `download_slices_full.py` | targeted `getSingleImage` fetch, 6 threads, retries |
| `extract_crops_full.py` | crop-size sweep, extraction, coordinate check |
| `lung_mask_diag.py`, `floor_calibration.py` | the §3 investigation |
| `make_splits.py` | patient-disjoint splits + verification |
| `qc_contact_sheet.py`, `qc_contact_sheet.png` | the 30 crops read in §6b |
| `images/{Malignant,Benign,Normal}/` | **3,461 crops**, 128×128 PNG (29 MB) |
| `labels_manifest.csv` / `.json` | per-crop manifest, **Phase A's 19 columns unchanged and in order**, plus 6: `label_strict`, `is_borderline`, `median_malignancy`, `lung_frac`, `lung_frac_excl_lesion`, `split` |
| `_work/label_mapping_study.json` | §2 table |
| `_work/crop_size_sweep.json`, `lung_mask_diag.json`, `floor_calibration.json` | §3 |
| `_work/coord_check_full.json` | §6a |
| `_work/download_log_full.json`, `extract_log.json`, `split_summary.json` | timings, skips, splits |
| `_work/dicom/` | 1,997 DICOM slices, 1.05 GB |

Disk added: **~1.05 GB**, essentially all DICOM. Phase A's `_work/xml/` is reused in
place and was not duplicated.

## 9. Live system integrity

Re-verified after extraction:
`checkpoints/ct_best_model_phase35_v2.pt` mtime **Aug 24 12:45** (unchanged),
`backend/main.py` mtime **Sep 6 14:18** (unchanged), `data/ct_scan/` **1,036 files**
(unchanged). All writes confined to `eval_run/lidc_full/`.

Stopping here as instructed — dataset built and QC'd, **no training.**
