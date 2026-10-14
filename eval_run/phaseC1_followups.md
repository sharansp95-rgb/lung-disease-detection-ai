# Phase C.1: the two follow-ups Phase C flagged — translation augmentation, and a looser Benign floor

**Scope: exactly the two bounded follow-ups named in Phase C §9, and nothing else.**
Two new experimental checkpoints, `checkpoints/ct_lidc_nodulecrop_v2_jitter.pt` and
`checkpoints/ct_lidc_nodulecrop_v3_jitter_loosebenign.pt` — **neither deployed**.
Phase C's `ct_lidc_nodulecrop_v1.pt`, the live `ct_best_model_phase35_v2.pt`,
`backend/main.py`, `data/ct_scan/` and Phase B's `eval_run/lidc_full/labels_manifest.csv`
are all untouched (re-hashed in §7). **No Phase D work was started.**

---

## Verdict up front

**Follow-up 1 works, decisively, and it changes one specific load-bearing claim in
Phase C's Phase D assessment. Follow-up 2 confirms Phase C's Benign caveat in
direction but shows it is smaller than feared, and it does not improve Benign
overall — it redistributes where Benign works.**

1. **Translation jitter closes almost the whole position-lock gap.** The identical
   9 mm off-centre re-extraction stress test goes from **0.816 → 0.034** (Phase C's
   `v1`) to **0.792 → 0.725** (`v2`). Grad-CAM inverts completely: enrichment at the
   displaced lesion **0.57× → 4.21×**, at the now-empty frame centre **3.03× → 0.50×**,
   and the heatmap peak is nearer the lesion than the frame centre in **2.4% → 87.0%**
   of crops. The model now searches.

2. **The cost is real and modest: ~3 points of centred accuracy.** Held-out test
   accuracy **0.9030 → 0.8693**, balanced **0.8591 → 0.8292**. "Is there a nodule in
   this patch" **0.970 → 0.931**.

3. **A new measurement Phase C did not have: the tolerance CURVE, not one point.**
   Phase C's no-go rested on "the detector has to centre it to within a few
   millimetres". Measured properly, `v1`'s tolerance is *even tighter* than Phase C
   stated — accuracy is already down to **0.338 at 4.2 mm** of displacement. `v2`
   holds **0.797 / 0.783 / 0.725** at 4.2 / 8.5 / 12.7 mm. The required detector
   tolerance moved from **under ~4 mm to ~13 mm**, and it is a *training parameter*,
   not an architectural limit.

4. **The looser Benign floor at 0.50 fails Phase B's own chest-wall control** — it
   admits **7.49%** of deliberately wall-centred crops against **0.29%** at 0.70, a
   26× rise. Reported rather than waved past; §4 explains the narrow way it was
   nonetheless used defensibly, and what that does and does not license.

5. **Phase C's Benign caveat: confirmed, and now quantified as small.** On the 414
   newly-admitted Benign crops, a model that never saw them scores **0.765** recall
   against **0.863** on the crops the strict floor kept — the bias direction Phase C
   found does persist. But the effect on the headline is **Benign recall 0.841 →
   0.815** when the excluded population is added back: about **2.6 points**, not a
   collapse.

---

## 1. What was run, and the design choice between "separate" and "combined"

Three of the four cells of a 2×2 were run. The brief left this call open and asked
for it to be justified briefly rather than agonised over.

| checkpoint | translation jitter | Benign floor | isolates |
|---|---|---|---|
| `ct_lidc_nodulecrop_v1.pt` (Phase C, unmodified) | no | 0.70 | the baseline |
| **`ct_lidc_nodulecrop_v2_jitter.pt`** | **±10 mm/axis** | 0.70 | **v1→v2 = the augmentation, alone** |
| **`ct_lidc_nodulecrop_v3_jitter_loosebenign.pt`** | ±10 mm/axis | **0.50 (Benign only)** | **v2→v3 = the data change, alone** |
| *(not run)* | no | 0.50 | — |

**Why the missing cell was skipped.** `v2 → v3` already isolates the data change with
the augmentation held constant, which is what the Benign question needs. A
loose-floor-without-jitter run would only answer "does loose Benign interact with
having no jitter", which nothing in Phase C asks. Two runs at ~10 min each rather
than three, and the isolation is not weakened.

**Both new models were trained from scratch (ImageNet init), not fine-tuned from
`v1`.** Fine-tuning an already position-locked checkpoint would confound "did jitter
teach translation tolerance" with "how much of the lock survived", and the run is
only ~10 minutes, so there was no reason to accept the confound.

Everything not named above is Phase C's setup unchanged: `ml/model.py`'s
`build_model("densenet121", 3)`, `ml/dataset.py`'s `build_transforms` verbatim, seed
42, 15 epochs (2 frozen warmup @ 1e-3, 13 fine-tune @ 1e-4), Adam, batch 8 + AMP,
inverse-frequency weighted `CrossEntropyLoss`, best-on-val-balanced-accuracy
checkpointing, and the manifest's `split` column used as-is. The class weights
printed by the `v2` run are **1.2603 / 1.9822 / 0.5875** on **648 / 412 / 1,390**
training crops — byte-identical to Phase C's, which is a free confirmation that
nothing else moved.

---

## 2. Follow-up 1: translation augmentation

### 2a. How the jitter was implemented, and why not `RandomAffine(translate=...)`

This is the one place where the obvious implementation would have quietly invalidated
the result. The stored Phase B crops are **128 px covering exactly 28 mm**. Translating
one of those by 9 mm means shifting it ~41 px out of 128 and filling 41 px with
padding — a synthetic artifact. The stress test this augmentation exists to fix was
deliberately built to *avoid* synthetic translation (it re-extracts from the DICOM).
Training on padded translations and then testing on real re-extracted ones would be a
train/test mismatch of precisely the kind this investigation exists to catch.

So `prep_jitter_source.py` re-opens the original DICOM for every train-split row and
cuts an **oversized native-resolution window** — the 28 mm box plus a 10 mm margin on
every side — through the identical windowing code (WL −600 / WW 1500, uint8). At train
time a 28 mm sub-window is taken at a random offset and resized to 128 px, which is
bit-identical to what `extract_crops_full.py` would have written had it been asked for
a crop centred that many mm away. Real pixels, no padding, no wrapping.

Three checks, measured not assumed:

* **Fidelity.** The zero-offset sub-window reproduces the stored Phase B PNG
  **bit-exactly on 98/98 spot-checked rows (max absolute gray difference 0)**, and
  110/110 on the extended manifest.
* **Coverage.** **2,450 of 2,450** train rows got the full ±10 mm margin — no row
  needed a reduced jitter budget (3 of 2,726 did on the extended manifest).
* **Physical scale.** The offset is drawn in **millimetres**, uniform in ±10 mm per
  axis, then converted to pixels with **that row's own `px_mm`** from the manifest —
  the same conversion Phase B/C use everywhere. A fixed *pixel* jitter would be a
  different physical displacement on a 0.5 mm/px scan than on a 0.9 mm/px one.
  Realised displacements: mean **7.68 mm** Euclidean, median 8.15, max 13.43.

**Applied to all three classes, including Normal.** Jittering only the nodule classes
would make "was this image jittered" predict the label — a new shortcut, in an
investigation about shortcuts.

Training: **9.6 min, peak 0.62 GB of 4.00 GB**, 36,750 jittered crops served. Best val
balanced accuracy **0.8179 at epoch 15** (Phase C's `v1`: 0.8528 at epoch 7). Note the
best epoch is the *last* one — `v2` was still improving when the schedule ran out. The
schedule was kept identical anyway, because comparability is worth more here than
squeezing the last fraction of a point.

### 2b. The 9 mm stress test, before and after

Same script, same method, same 207 test nodule crops re-extracted from the same DICOMs
with the 28 mm box shifted 9 mm per axis. **The Phase C.1 copy of the probe was first
run against `v1` and reproduced Phase C §4 exactly** — 0.816 → 0.034, enrichment
0.57× / 3.03×, peak-follows 2.4%, 195/207 called Normal at 0.942 mean confidence — so
the before/after is a genuine comparison and not two different measurements.

| | **v1 (Phase C)** | | **v2 (jitter)** | |
|---|---|---|---|---|
| | centred | **shifted 9 mm** | centred | **shifted 9 mm** |
| nodule accuracy (n=207) | 0.816 | **0.034** | 0.792 | **0.725** |
| Benign (n=132) | 0.841 | **0.008** | 0.818 | **0.788** |
| Malignant (n=75) | 0.773 | **0.080** | 0.747 | **0.613** |
| **Normal control (n=298)** | 0.963 | **0.936** | 0.923 | **0.913** |

The Normal control is still load-bearing and still behaves: Normal crops go through the
identical shift and barely move for both models, so the shift does not damage the input
in either case. Re-extraction reproduces the stored PNGs' predictions on **100.0%** of
crops for `v2` as well, so the probe's code path is faithful for the new checkpoint too.

**Grad-CAM on the shifted crops — the mechanism, not just the score:**

| | v1 | **v2** |
|---|---|---|
| enrichment at the lesion's actual (off-centre) position | 0.57× | **4.21×** |
| enrichment at the now-empty frame centre | 3.03× | **0.50×** |
| peak closer to lesion than to frame centre (50% = no preference) | 2.4% | **87.0%** |
| predictions on the 207 shifted crops | **195 Normal** @ 0.942 | **20 Normal** @ 0.688 |

That is a complete inversion. `v1` looked at the middle and correctly reported the
middle was empty; `v2` finds the lesion where it actually is.

**And the centred behaviour Phase C cared about is preserved.** Centre-disk enrichment
on the *unshifted* crops: **v1 3.07× → v2 3.30×**, against Phase C's measured untrained
architectural baseline of **1.05×**. The augmentation added translation tolerance
without costing lesion attention.

### 2c. The tolerance curve — the number a Phase D cost estimate actually needs

Phase C measured the position lock at exactly one displacement. A curve is what a
detector spec needs, and it also settles the obvious objection to the result above:
the training jitter is ±10 mm per axis, so **the 9 mm probe point sits inside the
augmentation range**. Points beyond 10 mm/axis are outside it. Same re-extraction
method, Grad-CAM dropped (accuracy is the decision-relevant quantity).

Nodule accuracy vs. displacement — 207 nodule crops, 298 Normal controls:

| shift (mm/axis) | Euclidean (mm) | **v1 nodule** | **v2 nodule** | v1 Normal ctrl | v2 Normal ctrl |
|---|---|---|---|---|---|
| 0 | 0.0 | 0.816 | 0.792 | 0.963 | 0.923 |
| 3 | 4.2 | **0.338** | **0.797** | 0.970 | 0.916 |
| 6 | 8.5 | **0.121** | **0.783** | 0.956 | 0.913 |
| 9 | 12.7 | **0.034** | **0.725** | 0.936 | 0.913 |
| 11 | 15.6 | 0.043 | **0.396** | 0.913 | 0.876 |
| 13 | 18.4 | 0.068 | **0.126** | 0.906 | 0.866 |

Two readings, both important and one of them unflattering:

* **`v1` is worse than Phase C reported.** Phase C said "a few millimetres". It is
  under four: at **4.2 mm** of displacement, `v1` is already at 0.338, and Benign
  specifically at **0.129**. The tolerance Phase D would have had to hit was tighter
  than the report stated.
* **`v2`'s tolerance is bought, not free.** It is flat to ~8.5 mm, still usable at
  12.7 mm, and then falls off a cliff at 15.6 mm — **right where the ±10 mm/axis
  training range ends**. The model did not learn general translation invariance; it
  learned the range it was shown. This is a genuine and useful result (the tolerance
  is now a dial), but it is not evidence of extrapolation, and it should not be
  reported as such. Widening the dial much further would need a bigger crop, because
  ±10 mm/axis in a 28 mm box already puts the lesion within 4 mm of the frame edge.

### 2d. Regression on the standard held-out test set

Same `split == "test"` crops as Phase C (192 patients / 505 crops, patient-disjoint).
The Phase C.1 eval script reproduces Phase C §2 exactly on `v1` (0.9030 / 0.8591, same
confusion matrix), so these are comparable.

| | **v1 (Phase C)** | **v2 (jitter)** | Δ |
|---|---|---|---|
| accuracy | **0.9030** | **0.8693** | −3.4 pts |
| balanced accuracy | **0.8591** | **0.8292** | −3.0 pts |
| "is there a nodule in this patch" | 0.970 / 0.972 bal | 0.931 / 0.932 bal | −3.9 pts |
| Benign / Malignant grading (n=207) | 0.816 | 0.792 | −2.4 pts |
| F1 Benign | 0.838 | 0.785 | −5.3 pts |
| F1 Malignant | 0.744 | **0.747** | +0.3 pts |
| F1 Normal | 0.975 | 0.940 | −3.5 pts |

Confusion matrix, `v2` (rows = true, cols = predicted):

| | → Benign | → Malignant | → Normal |
|---|---|---|---|
| **Benign** | **108** | 15 | 9 |
| **Malignant** | 16 | **56** | 3 |
| **Normal** | 19 | 4 | **275** |

**There is a regression and it should not be minimised: about three points across the
board.** The error structure is unchanged in kind — nodule/no-nodule crossings rise
from 15 to 31 of 505, and Benign↔Malignant confusion stays the dominant error. Vendor
spread stays flat (0.841–0.931 balanced across four manufacturers), so nothing about
the augmentation reopened the acquisition-style channel.

Three points of centred accuracy for a 3× wider centring tolerance is the trade this
follow-up buys. Whether that is worth it depends entirely on whether anything is ever
going to front-end the classifier — which is the Phase D question, and §6.

---

## 3. Follow-up 2, step 1: re-calibrating the floor at 0.50

Phase B's `floor_calibration.py` re-run with the floor grid extended downward — same
sampler, same `Random(23)` seed, same 700 slices, same chest-wall control construction,
same metric. **The deterministic parts reproduce Phase B exactly**: 28 mm Malignant
p5/p25/p50 = 0.622/0.789/0.975, Benign = 0.522/0.648/0.843, and the chest-wall control
p75 = **0.373**, max = **0.753** — identical to `_work/floor_calibration.json`. (The
Normal row differs slightly because Phase B swept three crop sizes and therefore drew
different random Normal sites from the same seeded stream; the classes that matter here
are deterministic.)

| floor | keep Malignant % | keep Benign % | keep Normal % | **M−B gap** | **wall admitted** |
|---|---|---|---|---|---|
| **0.70** (Phase B's choice) | 87.18 | 69.73 | 75.57 | **17.5** | **0.29%** (2/694) |
| 0.65 | 90.60 | 74.53 | 82.33 | 16.1 | 0.58% (4/694) |
| 0.60 | 96.58 | 81.63 | 86.93 | 15.0 | 1.73% (12/694) |
| 0.55 | 98.72 | 90.40 | 90.52 | 8.3 | 3.46% (24/694) |
| **0.50** (Phase C's target) | 99.15 | 96.45 | 95.69 | **2.7** | **7.49%** (52/694) |
| 0.45 | 99.57 | 97.29 | 97.56 | 2.3 | 13.54% (94/694) |
| 0.40 | 99.57 | 97.49 | 97.99 | 2.1 | 20.89% (145/694) |

**The honest answer to "does 0.50 still keep wall-centred junk out": no, not to Phase B's
standard.** Phase B picked 0.70 partly *because* it admitted only 0.3% of deliberately
wall-centred crops. At 0.50 that is **7.49%** — a 26× rise. Lowering the floor blindly,
which the brief explicitly warned against, would have imported exactly the junk the
control was built to detect.

**But the payoff is the specific defect Phase B flagged as unresolved.** Phase B: "*no*
lung-fraction floor is class-neutral — Benign is rejected 12–22 points more often than
Malignant at every size and threshold tested." At 0.70 that gap is **17.5 points**. At
0.50 it is **2.7 points**. The class-correlated selection bias, which Phase C named as
the largest un-quantified uncertainty in its numbers, is essentially gone at 0.50.

### How that tension was resolved

**The loose floor was applied to the Benign nodule class only. The Normal class was
left at 0.70 and was not re-sampled.**

The chest-wall control exists to stop the *blind Normal sampler* from landing on the
chest wall — a Normal crop has no anchor, so a low lung fraction there can mean the crop
is of nothing in particular. A Benign crop is centred on a **radiologist-annotated
nodule**, so a low lung fraction there means a real juxtapleural lesion with chest wall
in frame. The 7.49% figure measures a risk that the anchored class does not carry.

That is a real argument, not a rationalisation, but it is also the limit of what it
licenses: it justifies loosening the floor for the *annotated* class only, and it does
not justify a 0.50 floor for the dataset as a whole. Nothing below was built on a looser
Normal class.

---

## 4. Follow-up 2, step 2: the newly-admitted Benign crops

`extract_loose_benign.py` re-runs `extract_crops_full.py`'s extraction logic
(imported, not reimplemented — same `lung_mask`, `box_for`, lesion-inclusion correction,
windowing, `save_png`, naming) restricted to Benign nodules with
`lung_frac ∈ [0.50, 0.70)` — the band that was never written to disk, which is why
Phase B's column supports the stricter direction as a filter but not the looser one.

**414 new Benign crops from 290 patients.** Two checks fell out of the run for free:

* The re-run counted exactly **913** Benign crops as "already written at 0.70" —
  matching Phase B's recorded Benign count exactly. The extraction reproduces Phase B's
  admitted set.
* **All 290 contributing patients were already in Phase B's manifest.** Zero genuinely
  new patients, as the brief predicted. Every new crop simply inherits the split its
  patient already had; nothing was reshuffled. `build_manifest_v2.py` re-derives the
  patient→split map from the written CSV afterwards and asserts zero patients in more
  than one split, the same way `make_splits.py` does. It also raises rather than guesses
  if a new patient ever appears.

Phase B's `labels_manifest.csv` is **not modified**. A new
`eval_run/phaseC1/labels_manifest_v2.csv` carries all 3,461 original rows (tagged
`source_phase=B`) plus the 414 new ones (`source_phase=C1_loose_benign`).

| split | patients | crops | Benign (of which new) | Malignant | Normal |
|---|---|---|---|---|---|
| train | 410 | 2,726 | 924 (276) | 412 | 1,390 |
| val | 195 | 576 | 203 (70) | 75 | 298 |
| test | 192 | 573 | 200 (**68**) | 75 | 298 |

**Benign grew 913 → 1,327, +45.3%.** The dataset is 3,461 → 3,875 crops.

### QC — are these real nodule crops or wall crops?

Phase B read 30 crops directly and ran the contour-vs-annulus HU test. Both were redone
for the new band, against an equal-size random sample of the Benign crops already
admitted:

| | n | contour − annulus HU (mean / median) | positive | wall fraction (mean / p90) |
|---|---|---|---|---|
| **NEW Benign, `lung_frac` [0.50, 0.70)** | 414 | **+235.3 / +206.3** | **371/414** | **0.380 / 0.452** |
| existing Benign (≥ 0.70), random sample | 414 | +422.4 / +378.2 | 413/414 | 0.071 / 0.231 |

The annotated lesion is genuinely present and genuinely denser than its surroundings in
the new crops — but the contrast is roughly half, and the crops are on average **38%
chest wall** rather than 7%. Both are exactly what "juxtapleural benign nodule" predicts.

**24 crops read directly** (`qc_loose_benign_sheet.png`, sorted by lung fraction so the
worst are visible): all 24 are anatomically plausible and centred on a pleura-adjacent
structure; in the 0.50–0.52 tail, roughly half are visibly wall-dominated and the nodule
is hard to pick out by eye. That is the honest, visible cost of the looser floor, and it
is the same kind of cost Phase B recorded for 0.70 ("three Normal crops show a visible
pleural band").

Population characteristics: mean diameter **7.51 mm**, mean in-contour HU **−324**
(Phase B's overall Benign: −374 — the new lesions read denser, consistent with being
pressed against the pleura).

---

## 5. Did the looser floor change anything? Three questions, three answers

### (a) Did it admit meaningfully more / different Benign crops? **Yes, on both counts.**

+414 crops, +45.3% of the Benign class, from 290 patients already in the dataset. And
they are a measurably different population, not more of the same: 38% chest wall vs 7%,
contour-vs-annulus contrast +235 HU vs +422 HU, in-contour density −324 HU vs −374 HU.

### (b) Did it change Benign-class performance? **It redistributed it rather than improving it.**

Benign recall on the extended test set (573 crops), by lung-fraction band. `v1` and `v2`
**never saw** the newly-admitted band in training; `v3` did.

| Benign recall by `lung_frac` | ≥ 0.85 (n=95) | [0.70, 0.85) (n=37) | **[0.50, 0.70) — NEW (n=68)** | **all Benign (n=200)** |
|---|---|---|---|---|
| `v1` (no jitter, never trained on the band) | **0.863** | 0.784 | **0.765** | **0.815** |
| `v2` (jitter, never trained on the band) | 0.800 | 0.865 | **0.662** | **0.765** |
| `v3` (jitter + trained on the band) | 0.716 | 0.703 | **0.868** | **0.765** |

Holding jitter constant (`v2` → `v3`), adding the loose crops moves recall on the
newly-admitted band **0.662 → 0.868 (+20.6 pts)** and moves it *down* on the stricter
bands (0.800 → 0.716 and 0.865 → 0.703). **Overall Benign recall is identical at
0.765.** Benign F1 on the extended test does improve slightly, 0.789 → 0.805, entirely
through precision.

So the extra data did not add Benign capability; it moved where the capability sits. A
model with the same total budget now spends more of it on juxtapleural lesions and less
on easy central ones. That is arguably the *right* redistribution for a clinical task,
but it is not an improvement, and calling it one would overstate the numbers.

### (c) Does Phase C's bias direction persist in the newly-admitted crops? **Yes for a model that never saw them — and it inverts for one that did.**

This is the cleanest answer of the three, and it is the measurement Phase C said it
could not make because "those crops were never written."

* **For `v1` — Phase C's own checkpoint, evaluated on crops it never trained on —
  Benign recall falls monotonically with lung fraction: 0.863 → 0.784 → 0.765.** The
  direction Phase C found from the [0.70, 0.85) slice extends into the band below it.
  Phase C's reasoning was correct.
* **The magnitude is smaller than the caveat allowed for.** Phase C wrote "I cannot say
  the 0.838 Benign F1 would survive on an unfiltered benign nodule population, and I
  have concrete reason to think it would drop." It does drop — **Benign recall 0.841 →
  0.815, F1 0.838 → 0.847** (recall down, precision up, F1 marginally up) when the
  excluded population is added back to the test set. On overall test accuracy the
  effect is 0.9030 → 0.8866. **This is a ~2–3 point effect, not a collapse.**
* **And it is a training-distribution artifact, not a property of the crops.** Once the
  band is in training (`v3`), it becomes the band the model does *best* on (0.868 vs
  0.703/0.716). The near-wall Benign crops are not intrinsically harder; they were
  simply absent.

**So Phase C's "treat Benign as an upper bound" should be updated to: it is an upper
bound by about 2–3 points of recall, measured — not an unquantified risk.** That is the
follow-up doing exactly what it was flagged to do.

### `v3` in full, for the record

| | on Phase B's original 505 test rows | on the extended 573 |
|---|---|---|
| accuracy | 0.8515 | 0.8534 |
| balanced accuracy | 0.8327 | **0.8503** |
| F1 Benign / Malignant / Normal | 0.743 / 0.742 / 0.933 | 0.805 / 0.721 / 0.926 |
| "is there a nodule here" | 0.923 | 0.925 |
| Benign vs Malignant (n=275) | 0.773 | 0.796 |

`v3`'s 9 mm stress test and tolerance curve match `v2`'s shape closely — nodule accuracy
**0.796 → 0.705** at 9 mm/axis, enrichment at the displaced lesion **3.35×** vs **0.55×**
at the empty centre, peak-follows **78.9%**, and the same cliff past the training range
(0.393 at 15.6 mm, 0.095 at 18.4 mm). The translation tolerance comes from the jitter and
survives the data change; it is not an artifact of either dataset.

---

## 6. Updated verdict on Phase D

**Phase C's go/no-go recommendation stands — do not start Phase D before the exam — but
one of its two stated reasons is now falsified, and that is a material change to the
cost estimate rather than a rhetorical one.**

### What changed

Phase C §9: *"The model has **zero** localization ability — accuracy 0.816 → 0.034 the
moment the lesion moves 9 mm. Every bit of the burden of finding the lesion falls on the
detector, and the detector has to centre it to within a few millimetres, because that is
the tolerance the classifier actually has. […] §4 makes that estimate more pessimistic,
not less, because it converts an unknown tolerance into a demonstrated hard
requirement."*

**That requirement is not hard.** It was a property of the training distribution, and it
cost a one-line augmentation concept, ~40 lines of real-pixel plumbing, and a 9.6-minute
run to move:

| | v1 (Phase C) | v2 (jitter) |
|---|---|---|
| usable centring tolerance | **< 4.2 mm** (0.338 at 4.2 mm) | **~12.7 mm** (0.725 at 12.7 mm) |
| accuracy at 12.7 mm displacement | 0.034 | 0.725 |
| CAM enrichment at the displaced lesion | 0.57× | 4.21× |
| CAM peak follows the lesion | 2.4% | 87.0% |
| centred test accuracy | 0.9030 | 0.8693 |

A detector that must place a box within ~13 mm of a lesion centre is a categorically
easier component than one that must place it within 4 mm. Phase C's §9 said this
follow-up "might change the Phase D cost estimate materially" — it does, in the
direction Phase C guessed, and by more than "partially closing" the gap: the 9 mm gap
is 96% closed (0.034 → 0.725 against a centred ceiling of 0.792).

### What did not change

* **The cliff sits at the edge of the augmentation range.** `v2` holds to 12.7 mm and
  breaks at 15.6 mm — exactly where ±10 mm/axis training stops. The model learned the
  range it was shown, not general translation invariance. The tolerance is a dial, and
  widening it much further needs a larger crop, since ±10 mm/axis in a 28 mm box already
  puts the lesion 4 mm from the frame edge. **No claim of extrapolation is supported.**
* **It costs ~3 points of centred accuracy** (0.9030 → 0.8693, balanced 0.8591 →
  0.8292). Real, and paid on every crop whether or not anything is off-centre.
* **The schedule argument is untouched.** Phase A costed a nodule detector as "a larger
  project than everything in Phase A combined, and not a ~10-day-with-an-exam
  undertaking." Relaxing the tolerance from 4 mm to 13 mm makes the detector *easier*;
  it does not make it *small*. There is still no detector, no detection training data
  prepared, and no evaluation harness for one.
* **Every deployment fact is unchanged.** `POST /predict/ct` still takes a whole
  uploaded slice; a crop classifier still cannot be served through it. "Normal" here
  still means *this 28 mm patch contains no nodule*, not *this patient is healthy*.
  Phase A's deployment no-go holds.
* **Nothing here touches the live model.** `ct_best_model_phase35_v2.pt` stays
  deployed; the standing recommendation to stop citing its external-Malignant 12/12 as
  generalization evidence is unaffected (Phase C §7(d) remains the reason).

### The concrete difference, stated precisely

Phase C's Phase D assessment was **"GO on the evidence, NO-GO on the schedule, and the
schedule call is not close because the required detector tolerance is a few
millimetres."**

The updated assessment is **"GO on the evidence, NO-GO on the schedule, and the
schedule call is still not close — but for the scope reason alone, not the tolerance
reason."** The tolerance objection has been measured, attacked, and largely removed.
What remains is that building, training and evaluating a nodule detector is a project,
and there are ~10 days and an exam.

**Recommendation, unchanged from Phase C: write it up rather than extend it.** Phase C.1
strengthens that write-up rather than complicating it — the CT story now reads: the
shortcut was real and structural, removing it structurally worked, the resulting model
learns real lesion features, its one severe limitation (position lock) was diagnosed and
then *fixed* with a measured before/after, and the largest remaining uncertainty (the
Benign selection bias) was bounded at 2–3 points rather than left open. That is a
complete arc.

**Do not start Phase D.** Nothing in this phase is an argument for starting it now; it is
an argument that it would be cheaper than Phase C thought, whenever it happens.

---

## 7. Live-system integrity

Re-hashed after all training and evaluation, not asserted:

```
checkpoints/ct_best_model_phase35_v2.pt  sha=93b3a979884edbc4  mtime=1787555730.4296012
backend/main.py                          sha=780120553178c0a5  mtime=1788684518.831581
data/ct_scan/                            n=1036  newest mtime 2026-08-23 20:53:00
eval_run/lidc_full/labels_manifest.csv   3461 rows, unmodified (mtime Sep 8 20:57)
checkpoints/ct_lidc_nodulecrop_v1.pt     unmodified (mtime Sep 9 00:25, written by Phase C)
```

The checkpoint and `backend/main.py` hashes match the values recorded in
`phaseC_lidc_training.md` §8 exactly. `data/ct_scan/` still holds 1,036 files whose
newest modification time is **2026-08-23**, i.e. before this investigation began — I do
not have Phase C's exact aggregate-hash formula, so the file count and mtime evidence is
given in place of reproducing its digest rather than presenting a differently-computed
number as a match.

Two new checkpoints were written (`ct_lidc_nodulecrop_v2_jitter.pt`,
`ct_lidc_nodulecrop_v3_jitter_loosebenign.pt`, 27 MB each). Every other write is confined
to `eval_run/phaseC1/`. **Disk added: 76 MB in `eval_run/phaseC1/` + 55 MB of
checkpoints = ~131 MB.**

**One process is running and it is not mine:** `python -m uvicorn main:app --reload
--port 8000`, the project's own backend dev server, which was already running before this
phase started and was not touched. No training or evaluation process was left running.

---

## 8. Artifacts

All under `eval_run/phaseC1/` unless noted. Nothing is referenced by `backend/`.

| path | what |
|---|---|
| `checkpoints/ct_lidc_nodulecrop_v2_jitter.pt` | **jitter-augmented checkpoint** (epoch 15, val bal 0.8179) — not deployed |
| `checkpoints/ct_lidc_nodulecrop_v3_jitter_loosebenign.pt` | **jitter + loose-Benign checkpoint** (epoch 13, val bal 0.8471) — not deployed |
| `prep_jitter_source.py`, `jitter_index.json`, `jitter_index_v2.json`, `jitter_src/` | §2a real-pixel jitter sources + the bit-exactness check |
| `lidc_data_jitter.py` | Phase C's Dataset + on-the-fly mm-denominated translation jitter |
| `train_lidc_crops_jitter.py`, `train_v2_jitter.log`, `train_v3.log` | training, identical to Phase C's except the jitter |
| `training_history_v2_jitter.json`, `training_curves_v2_jitter.png` | per-epoch metrics, `v2` |
| `training_history_v3_jitter_loosebenign.json`, `training_curves_v3_jitter_loosebenign.png` | per-epoch metrics, `v3` |
| `offcentre_probe_c1.py`, `offcentre_results_{v1_repro,v2,v3}.json` | §2b, the 9 mm stress test (v1 run = the reproduction check) |
| `offcentre_control_c1.py`, `offcentre_control_results_{v2,v3}.json` | §2b Normal control + re-extraction fidelity |
| `tolerance_sweep.py`, `tolerance_sweep.json`, `tolerance_sweep_v3.json` | §2c the displacement curve — the new measurement |
| `eval_test_c1.py`, `eval_test_v1_repro.json`, `eval_test_v2.json`, `eval_test_v3_orig505.json` | §2d, on Phase B's original 505 test rows |
| `eval_extended_{v1,v2_jitter,v3}.json` | §5, on the 573-crop extended test set, with per-band Benign slices |
| `confusion_matrix_test_{v2_jitter,v3}.png` | §2d / §5 figures |
| `floor_recalibration_050.py`, `floor_recalibration_050.json`, `.log` | §3, Phase B's calibration re-run with the grid extended to 0.40 |
| `extract_loose_benign.py`, `loose_benign_rows.json`, `images_loose_benign/` (414) | §4, the newly-admitted Benign crops |
| `build_manifest_v2.py`, `labels_manifest_v2.csv`, `manifest_v2_summary.json` | §4, the extended manifest with inherited patient-disjoint splits |
| `qc_loose_benign.py`, `qc_loose_benign.json`, `qc_loose_benign_sheet.png` | §4 QC — contour-vs-annulus, wall fraction, 24 crops read directly |

Stopping here as instructed — **Phase D not started.**
