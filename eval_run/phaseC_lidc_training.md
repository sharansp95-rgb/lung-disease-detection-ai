# Phase C: training on the LIDC nodule crops — is there real lesion-level signal?

**Scope: training + evaluation on `eval_run/lidc_full/` only.** New experimental
checkpoint `checkpoints/ct_lidc_nodulecrop_v1.pt`. `ct_best_model_phase35_v2.pt`,
`backend/main.py` and `data/ct_scan/` untouched (re-hashed in §9). **No Phase D work
was started**, per the brief.

This answers the question the whole investigation has been building toward. Every prior
phase established that the whole-slice CT classifier keys on acquisition framing rather
than pathology — and that removing the framing confound (phase37) did *not* produce
lesion learning. Phase A/B built the structural fix: 28 mm crops centred on
radiologist-annotated nodules, so there is no background, table or framing left in the
input to shortcut on. This trains on that data and asks whether anything real is there.

---

## Verdict up front

**Yes — there is genuine lesion-level signal, and this is the first phase of the
investigation where Grad-CAM shows the model looking at pathology rather than at style.
But the model has no localization ability whatsoever, and that limitation is
quantified, not hedged.**

Three findings, in order of how much they matter:

1. **The attention is real and it is on the lesion.** Grad-CAM centre enrichment is
   **2.96×** against an architectural baseline of **1.05×** (measured, not assumed —
   an ImageNet densenet121 with an untrained head run through the identical pipeline).
   Border enrichment is **0.17×**: the model actively *avoids* the crop edge, so the
   crop-boundary shortcut that would have been the obvious new failure mode did not
   appear. Every previous phase measured ~1.0× — no preference at all. This is a
   categorical change, not an incremental one.

2. **But it is position-locked, and the test that proves it is brutal.** Re-extracting
   the *same lesions* from the *same DICOM slices* with the 28 mm box shifted so the
   nodule sits 9 mm off centre — real pixels, no synthetic translation — collapses
   nodule accuracy from **0.816 to 0.034**. Grad-CAM enrichment at the lesion's new
   position is **0.57×** (below uniform) while the now-empty frame centre still draws
   **3.03×**. The heatmap peak is nearer the lesion than the frame centre in **2.4%**
   of cases. The model reads the centre of the frame and reports what is there; it
   does not search.

3. **That is a real feature, not a shortcut — and the distinction is evidenced.**
   195 of the 207 shifted crops are called **Normal at 0.942 mean confidence**. That is
   the correct answer *about the centre of the frame*: the lesion left, so there is no
   longer a lesion in the middle. The control confirms the probe is sound — Normal
   crops put through the identical 9 mm shift are essentially unaffected
   (**0.963 → 0.936**), so the collapse is caused by moving a lesion, not by
   re-cropping. What the model learned is a genuine, local, ~centimetre-scale
   lesion-presence-and-character feature with a hard-wired spatial prior on where to
   apply it.

**Headline numbers, LIDC test split (192 patients / 505 crops, patient-disjoint):**
**accuracy 0.903, balanced accuracy 0.859.** "Is there a nodule in this patch at all"
is **0.970** (balanced 0.972). "Benign or malignant, given there is one" is **0.816**.

---

## 1. Training

`eval_run/phaseC/train_lidc_crops.py`, adapted from `ml/train.py` rather than written
fresh (same warmup→fine-tune schedule, same Adam/lr/seed, same inverse-frequency
weighted `CrossEntropyLoss`, same best-on-val checkpointing). `ml/model.py`'s
`build_model("densenet121", 3)` reused unchanged; `ml/dataset.py`'s `build_transforms`
reused verbatim so augmentation matches every other run in this project.

Three deliberate differences, all forced and all reported:

| # | change | why |
|---|---|---|
| 1 | manifest-backed `Dataset` (`phaseC/lidc_data.py`) instead of `ImageFolder` | `lidc_full/images/` is laid out by class, not by split. The manifest's `split` column is used **as-is**; nothing re-split. |
| 2 | batch 8 + `torch.autocast` fp16 | the brief's starting point for a 4 GB card |
| 3 | best epoch selected on val **balanced** accuracy | 57% of this dataset is Normal, so plain accuracy is a weak selector. Both checkpoints were written and both epochs logged, so the choice is auditable. |

**Hardware, as actually used.** RTX 3050 Laptop, 4.00 GB. Batch 8 + AMP peaked at
**0.62 GB allocated** — no OOM, no back-off needed, and roughly 6× headroom. I kept
batch 8 anyway rather than raising it: the run is 9.4 minutes end to end, so a larger
batch buys nothing worth the loss of comparability with the brief's stated setting.
15 epochs (2 frozen warmup @ 1e-3, 13 fine-tune @ 1e-4), ~20 s/epoch frozen, ~40 s
fine-tune.

Class weights from the 648 / 412 / 1,390 training counts: Benign 1.2603,
Malignant 1.9822, Normal 0.5875.

| | best val balanced acc | best val acc |
|---|---|---|
| epoch | **7** | 12 |
| val acc | 0.8992 | 0.9071 |
| val balanced acc | **0.8528** | 0.8468 |

Epoch 7 is `checkpoints/ct_lidc_nodulecrop_v1.pt`. Val balanced accuracy plateaus at
0.84–0.85 from epoch 7 onward while train accuracy climbs to 0.95 — mild overfitting
after epoch ~8, no class collapse, nothing pathological. Curves:
`phaseC/training_curves.png`.

**Note on reproducibility.** This run was interrupted at epoch 8 by a pause request and
restarted from scratch rather than resumed, because a partial run's checkpoint should
not be silently treated as a finished model. The restart reproduced the interrupted
run's epoch-7 checkpoint exactly (val_acc 0.8992, val_bal 0.8528), which is a free
seed-determinism check.

**Budget.** Well inside the ~10-day exam window: **9.4 min training, ~25 min for all
evaluation and probes combined.** Nothing here took dramatically longer than the dataset
size implied, and nothing needed grinding through.

---

## 2. Held-out LIDC test set

`split == "test"`, **192 patients / 505 crops**, patient-disjoint from train and val.
`phaseC/eval_lidc_test.py`, raw output `phaseC/eval_test_results.json`,
figure `phaseC/confusion_matrix_test.png`.

**Accuracy 0.9030 · balanced accuracy 0.8591** (chance balanced = 0.333).

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Benign | 0.835 | 0.841 | 0.838 | 132 |
| Malignant | 0.716 | 0.773 | 0.744 | 75 |
| Normal | 0.986 | 0.963 | 0.975 | 298 |
| **macro avg** | **0.846** | **0.859** | **0.852** | **505** |

Confusion matrix (rows = true, cols = predicted):

| | → Benign | → Malignant | → Normal |
|---|---|---|---|
| **Benign** | **111** | 17 | 4 |
| **Malignant** | 17 | **58** | 0 |
| **Normal** | 5 | 6 | **287** |

**The structure of the errors is the informative part.** Almost nothing crosses the
nodule/no-nodule boundary: only 4 of 207 nodule crops are called Normal, and 11 of 298
Normal crops are called a nodule. Collapsed to that binary question the model is at
**0.970 accuracy / 0.972 balanced**. Essentially all the error is **Benign ↔ Malignant
confusion** — 34 of the 49 total errors. Restricted to the 207 nodule crops, malignancy
discrimination is **0.816** (0.831 using only the two nodule logits).

That is the honest shape of the task: *finding* the nodule in a centred patch is close
to solved; *grading* it is hard, which is what one should expect from labels that are
four radiologists' subjective impressions rather than biopsy.

### Slices that interrogate the dataset's known defects

| subset | n | acc | balanced | Benign recall | Malignant recall |
|---|---|---|---|---|---|
| non-borderline (Phase A strict labels) | 465 | 0.918 | **0.876** | 0.852 | 0.814 |
| borderline only | 40 | 0.725 | **0.708** | 0.792 | 0.625 |
| `lung_frac ≥ 0.85` (Phase A's floor) | 390 | 0.921 | **0.894** | 0.863 | 0.860 |
| `lung_frac ∈ [0.70, 0.85)` | 115 | 0.843 | **0.756** | 0.784 | **0.500** |
| nodules ≥ 8 mm | 138 | 0.804 | 0.804 | 0.779 | 0.829 |
| nodules < 8 mm | 69 | 0.841 | 0.453 | 0.906 | **0.000** (n=5) |
| GE (n=306) / Siemens (138) / Philips (29) / Toshiba (32) | | | 0.839 / 0.895 / 0.867 / 0.928 | | |

Three things fall out. Borderline labels cost ~17 points of balanced accuracy, exactly
as Phase B predicted — the labels are the noise, not the model. The loose 0.70
lung-fraction floor costs ~14 points, concentrated almost entirely in Malignant recall
(0.860 → 0.500) on the juxtapleural crops. And **there is no meaningful vendor spread**
— 0.839 to 0.928 across four manufacturers with very different sample sizes, which is
what a model that is *not* keying on scanner style looks like.

---

## 3. Grad-CAM — the check this whole investigation was built around

`phaseC/gradcam_lidc.py`, run on **all 505 test crops** (not a sample), raw output
`phaseC/gradcam_results.json`, 505 overlays in `phaseC/gradcam_pngs/`, 12-panel summary
`phaseC/gradcam_grid.png`.

Metric is the same enrichment measure used in `ct_shortcut_inversion_analysis.md` and
`ct_framing_normalization_experiment.md`: **(share of Grad-CAM mass in a zone) ÷ (that
zone's share of image area)**. 1.00× = no preference. Ground truth is available here in
a way it never was for whole slices: the nodule is at the crop centre by construction
(`center_row`/`center_col`), and `nodule_diam_mm` gives its size. Zones on the 224×224
input (28 mm box → 8 px/mm): **lesion** = centre disk at the nodule's own radius,
floored at 16 px because Grad-CAM is read off a 7×7 feature map (32 px/cell) and a
smaller disk is not resolvable; **centre** = fixed 56 px disk (area share 0.196);
**border** = within 16 px of the frame edge (area share 0.265).

| group | n | lesion enr | centre enr | **border enr** | median peak dist |
|---|---|---|---|---|---|
| all test crops | 505 | **3.69×** | **2.96×** | **0.17×** | 0.32 mm |
| Benign correct | 111 | 3.77× | 3.33× | 0.15× | 5.57 mm |
| Benign misclassified | 21 | 1.85× | 1.88× | 0.52× | 4.06 mm |
| Malignant correct | 58 | 2.02× | 2.36× | 0.35× | 3.94 mm |
| Malignant misclassified | 17 | 3.51× | 3.35× | 0.16× | 5.57 mm |
| Normal correct | 287 | 4.18× | 3.02× | 0.11× | 0.09 mm |
| Normal misclassified | 11 | 2.50× | 2.33× | 0.41× | 5.66 mm |
| **CONTROL: ImageNet weights, untrained head** | 169 | **1.06×** | **1.05×** | **0.88×** | 8.97 mm |

**Reading this honestly:**

* **The signal is real and large.** 2.96× centre against a measured 1.05× architectural
  baseline. Compare `ct_framing_normalization_experiment.md` §6, where the whole-slice
  model sat at **1.00–1.07×** lung-field enrichment on images it classified *correctly*.
  The trained model has moved somewhere the untrained architecture does not go on its
  own.
* **No crop-border shortcut.** 0.17× vs the control's 0.88×. Training pushed attention
  *away* from the frame edge. The most plausible new artifact did not materialise.
* **Misclassified crops show consistently weaker, more diffuse attention** across
  Benign (3.77× → 1.85×) and Normal (4.18× → 2.50×), with border enrichment rising
  (0.15 → 0.52, 0.11 → 0.41). When the model is wrong it is also not looking properly.
  That coupling is what you want to see; it is evidence the metric is measuring
  something causally connected to the decision.
* **Visually** (`gradcam_grid.png`): on large malignant masses the hot region visibly
  *conforms to the lesion's outline* — the spiculated mass and the large round mass in
  row 2 both show attention tracing the actual boundary, not a generic blob. On small
  nodules and on Normal crops it is a rounded blob at the centre, which is what a 7×7
  CAM floor looks like.
* **The ambiguity I could not resolve from this table alone, stated plainly:** Normal
  crops — which contain no lesion — show the **highest** centre enrichment of any class
  (4.18×, median peak 0.09 mm from dead centre). That is not consistent with "found the
  lesion". It is consistent with "always looks in the middle". Since every lesion in
  this dataset is in the middle by construction, this table cannot separate lesion
  attention from a positional prior. §4 was written to settle that, and it does.

---

## 4. The off-centre probe — separating "found the lesion" from "looks at the middle"

`phaseC/offcentre_probe.py` + `phaseC/offcentre_control.py`, raw output
`phaseC/offcentre_results.json`, `phaseC/offcentre_control_results.json`, overlays
`phaseC/offcentre_pngs/`.

**Method.** For each of the 207 test nodule crops, the same lesion is re-extracted from
the same original DICOM slice with the 28 mm box shifted so the nodule sits ~9 mm off
centre — still well inside the frame. These are real pixels through the identical
windowing/resampling code (WL −600 / WW 1500, 128 px, bilinear); no padding, no
wrapping, no synthetic translation. The only thing that changed is where the lesion is.
Re-extracting *unshifted* through the same code path reproduces the stored training
PNGs' predictions on **100.0%** of crops in all three classes, so the code path is
faithful.

| | centred | **shifted 9 mm** |
|---|---|---|
| nodule accuracy (n=207) | 0.816 | **0.034** |
| Benign (n=132) | 0.841 | **0.008** |
| Malignant (n=75) | 0.773 | **0.080** |
| **Normal control (n=298)** | 0.963 | **0.936** |

| Grad-CAM on the shifted crops | value |
|---|---|
| enrichment at the lesion's actual (off-centre) position | **0.57×** |
| enrichment at the now-empty frame centre | **3.03×** |
| peak closer to lesion than to frame centre | **2.4%** (50% = no preference) |
| median peak distance to frame centre | **0.09 mm** |

**What this establishes, and what it does not.**

The Normal control is the load-bearing row. Normal crops go through exactly the same
9 mm shift and barely move (0.963 → 0.936). So the shift does not damage the input, and
the collapse on nodule crops is caused specifically by *moving a lesion out of the
centre* — not by re-cropping, resampling, or anything about the probe.

And the failure is not random: **195 of 207 shifted crops are called Normal at 0.942
mean confidence.** Read as a statement about the whole frame that is wrong. Read as a
statement about the centre of the frame it is *correct* — the lesion is no longer there.
The model has learned a genuine local feature ("is there a lesion here, and what kind")
and applies it at a fixed location.

So the answer to the investigation's central question is **not** "another shortcut". A
shortcut would be a cue correlated with the label but causally irrelevant to it —
background, table, vendor style, crop border. This is the opposite: the model is using
the lesion's own appearance, and it fails when you take the lesion away from where it
looks. That is a *scope* limitation, not a validity one.

It is still a severe limitation. A classifier with zero localization ability is only as
good as whatever centres its input — which is precisely the Phase D question.

---

## 5. Class-from-style shortcut probe, at 32× the data

`phaseC/style_probe.py`, raw output `phaseC/style_probe_results.json`. Same probe as
`ct_framing_normalization_experiment.md` §2C and Phase A: can a simple classifier
recover the **class** from global, non-lesion statistics alone? 26 features — the 20
global image statistics from the ancestor probe, plus manufacturer one-hot, `px_mm`, and
`mean_HU_crop`. Deliberately excluded as lesion-derived: `mean_HU_in_contour`,
`nodule_diam_*`, the ratings, and `lung_frac_excl_lesion` (its correction is computed
from the annotated contour). 5-fold **grouped by patient**, as Phase A did.

| data | balanced accuracy (RF / LR) | chance |
|---|---|---|
| whole-slice training data (phase37 §2C) | **0.944** | 0.333 |
| Phase A nodule crops (107 crops, 26 patients) | **0.433** / 0.464 | 0.333 |
| **Phase B nodule crops (3,461 crops, 797 patients)** | **0.523 / 0.535** | 0.333 |
| — image statistics only, no vendor/px_mm/HU | 0.519 / 0.523 | 0.333 |
| — **manufacturer one-hot ONLY (control)** | **0.363 / 0.363** | 0.333 |
| — + `lung_frac` (ablation) | 0.562 / 0.570 | 0.333 |
| Benign vs Malignant (Phase A: 0.687 / 0.653) | **0.715 / 0.709** | 0.500 |
| Benign vs Malignant, non-borderline only | 0.726 / 0.737 | 0.500 |

**Did something creep back at scale? Partly — but not the thing we were afraid of.**

The 3-class figure rose 0.433 → 0.523. Two reasons to not read that as a returning
shortcut. First, Phase A's 0.433 came from 107 crops across 26 patients; that estimate
carries enormous variance and the Phase B number should simply be preferred, the same
way Phase B's HU separation superseded Phase A's. Second and more decisively, **look at
which features do the work**: the top random-forest importances are `std`, `mean`,
`frac_gt200`, `p75`, `p95`, `mean_HU_crop` — every one of them a measure of *how much
dense tissue is in this 28 mm box*. That is not acquisition style. A malignant nodule is
larger and denser than a benign one, and both are denser than empty parenchyma, so a
crop containing one genuinely has a higher mean and p75. This is crude pathology leaking
into global statistics, which is signal, not shortcut.

**The control that makes this conclusive: manufacturer alone gives 0.363 against a
chance of 0.333.** Scanner vendor — the acquisition-style channel that drove every
previous failure in this investigation, and the one thing at 32× scale that could
plausibly have become learnable — carries essentially nothing. That matches the test-set
vendor slices in §2 (0.839–0.928 balanced accuracy across four vendors). **The
shortcut *surface* stayed collapsed.** What rose is a density statistic that is
genuinely confounded between real pathology and the selection bias discussed next.

---

## 6. Is the Benign result trustworthy? Largely not, and here is the specific reason

Phase B flagged that **no lung-fraction floor is class-neutral: Benign crops were
rejected 12–22 points more often than Malignant at every size and threshold tested**,
for two real mechanisms — LIDC's benign nodules are genuinely smaller and more often
subpleural, and the lesion-inclusion correction mechanically protects large,
disproportionately malignant lesions. The Benign class in this dataset has therefore
been filtered by a process correlated with its own label. That caveat has to be applied
to every Benign number above, and it is not a formality.

**What I can conclude about Benign:**

* Benign is the class the model does *best* on among the two nodule classes on raw
  recall (0.841 vs Malignant's 0.773), and its confusions are overwhelmingly with
  Malignant (17 of 21), not with Normal (4). So the model is locating benign nodules
  reliably and grading them imperfectly.
* The 21 Benign errors are accompanied by measurably worse attention (lesion enrichment
  1.85× vs 3.77× for correct ones), so they look like genuine hard cases rather than a
  systematic artifact.
* The Benign-vs-Malignant style probe at **0.715** is well above the 0.500 chance line
  and *rose* slightly from Phase A's 0.687 at 32× the data. Phase A flagged this exact
  number as un-disentanglable and it remains so — with the added, sharper worry that
  the selection bias is one of its plausible sources, since "which benign nodules
  survived the floor" is precisely a size/position filter and size/density is what the
  probe's top features measure.

**What I cannot conclude, and will not claim:**

I cannot say the 0.838 Benign F1 would survive on an unfiltered benign nodule
population, and I have concrete reason to think it would drop. The surviving Benign
crops are biased toward the larger, more central, more nodule-like end of the benign
distribution — exactly the ones easiest to see and hardest to confuse with parenchyma.
The `lung_frac ∈ [0.70, 0.85)` slice is a partial, imperfect window on this: the crops
nearest the chest wall are the ones the model does worst on (balanced 0.756 vs 0.894),
and *those are disproportionately the benign ones the floor was already discarding.*
The direction of the bias and the direction of the performance gradient agree, which is
what you would expect if the filtering is flattering the Benign score.

**A specific, cheap follow-up that would settle it** (not run here — out of scope, and
it needs re-extraction, not just a filter): re-extract at a floor of ~0.50 with the
chest-wall control re-calibrated, and compare Benign recall on the crops that only the
looser floor admits. Phase B's `lung_frac` column supports the *stricter* direction as a
filter, but not the looser one, because those crops were never written.

**Bottom line: treat the Normal and Malignant numbers as trustworthy within this
dataset, and treat the Benign numbers as an upper bound.**

---

## 7. External images — comparison attempted, and declared invalid

`phaseC/external_scale_check.py` (→ `external_scale_check.json`),
`phaseC/external_eval.py` (→ `external_eval_results.json`).

I attempted this rather than declining on principle, and it fails on four independent
grounds. Each is measured.

**(a) Physical scale is unknown and must be guessed.** LIDC crops are a 28 mm box sized
from each slice's own DICOM `PixelSpacing`. The external images are 8-bit PNG exports
with no DICOM metadata, so mm/px has to be inferred from anatomy. Assuming a 320 mm
adult chest width gives 0.40–0.98 mm/px, and a ±15% error on that assumption is a ±15%
error on the physical crop size.

**(b) Resolution is 2–4× short.** Under that estimate, 28 mm works out to **28–37 px** in
the external Malignant images and **31–70 px** in the Normals. LIDC crops are **128 px**
for the same 28 mm. An external crop upsampled to 224 carries a fraction of the real
spatial detail of anything the model was trained on — and nodule texture is the entire
thing being classified.

**(c) The intensity window does not match, and does not even match within the external
set.** LIDC crops are rendered from HU at a fixed lung window (WL −600 / WW 1500), so
parenchyma sits at a known constant level — measured p35 gray ≈ **82**. The external
images were exported with unknown per-image windows. Their lung band measures:

| set | lung-band gray | LIDC reference |
|---|---|---|
| external Malignant (12) | 29–65 | ~82 |
| external Normal (9) | **0–112** | ~82 |

`normal_21` reads **0.0** — a soft-tissue/mediastinal window in which lung is pure black
with no parenchymal texture at all. `normal_6` reads 15.6 and additionally carries a
tiled *SCIENCEPHOTOLIBRARY* watermark across the whole frame. The Malignant and Normal
external sets are windowed differently **from each other**, so any comparison between
them is confounded by rendering, not pathology.

**(d) Decisively: there is nothing to centre a Malignant crop on.** A nodule crop must be
centred on a nodule. I opened all 12 external Malignant slices and attempted to localize
a lesion in each. **Only one — `adenocarcinoma_000113` — contains a discrete,
confidently localizable focal lung lesion.** Two more (`adenocarcinoma_000112`,
`squamous_000110`) show non-nodular consolidation I would not stake a crop centre on.
Four (`large_cell_000108`, `large_cell_000111`, `large_cell_000113`, `squamous_000108`)
are slices at the thoracic inlet / apices where essentially no lung lesion is present at
that level at all. The remaining five show grossly clear lungs at the slice shown.
These images are labelled by **carcinoma subtype at the case level**; the individual
slice is not nodule-annotated and frequently does not contain a visible lesion.

**So no external-Malignant number is produced.** Inventing crop centres for 11 of 12
slices would manufacture exactly the kind of number this investigation exists to
discredit. This inability is itself a finding, and a retrospectively illuminating one:
**the external-Malignant 12/12 that three prior checkpoints scored was never a lesion
detection result, because most of those slices do not visibly contain a lesion.** That
independently corroborates the standing recommendation to stop citing it.

### The one external number that can be produced, with its own caveat

The external **Normal** side needs no localization — "a nodule-free patch of lung
parenchyma" is exactly how Phase B built its own Normal class, and it reproduces
mechanically: lung-mask the slice, apply the same ≥0.70 lung-fraction floor, cut the
same 28 mm box. 6 crops per slice, 54 total.

| | Benign | Malignant | **Normal** | Normal rate |
|---|---|---|---|---|
| all 54 crops | 0 | 3 | **51** | **0.944** |
| excluding the 2 flagged slices (42 crops) | 0 | 3 | 39 | 0.929 |

For context, on external Normal data the whole-slice checkpoints scored 0/9 and 0/264
(`phase35_v2`), 7/9 and 250/264 (`phase36`), 2/9 and 47/264 (`phase37`).

**This is not a generalization score and must not be reported as one.** Subject to
caveats (a)–(c), and more importantly: **a degenerate always-Normal model would score
54/54 on this test.** There is no external Malignant counterpart to balance it — that is
the point of §7(d) — and §4 showed this model's dominant failure mode *is* to answer
Normal whenever a lesion is not perfectly centred. The 51/54 is consistent with the
model working as designed, and it is also consistent with the least interesting possible
explanation. It should be read as "the out-of-distribution behaviour is not obviously
broken", nothing more.

---

## 8. Comparison with the live model, and what the numbers actually mean

| | `ct_best_model_phase35_v2` (live) | `ct_lidc_nodulecrop_v1` (this) |
|---|---|---|
| input | whole CT slice | 28 mm nodule-centred crop |
| in-dataset test accuracy | **~0.981** (103 slices) | **0.903** (505 crops) |
| test set size | 103 images, 1 source | 505 crops, 192 patients, 4 vendors |
| splits | image-level, leakage-repaired | **patient-disjoint** |
| labels | folder class, single source | 4-radiologist panel, 23.9% excluded as ambiguous |
| Grad-CAM lesion/lung enrichment | **1.00×** (= no preference) | **2.96×** vs 1.05× baseline |
| what drives the decision | acquisition framing/style (3 phases of evidence) | the lesion at the frame centre (§3, §4) |
| class-from-style probe | **0.944** | **0.523**, vendor-only **0.363** |

**The lower number is the better result, and it should be reported that way.**
98.1% and 90.3% are not measuring the same thing. The 98.1% comes from a single-source
whole-slice test set on a model that three separate experiments showed is reading
acquisition style, with Grad-CAM attention statistically indistinguishable from uniform
*on images it gets right*. The 90.3% comes from patient-disjoint splits across four
scanner vendors, on labels that are four radiologists' independent malignancy ratings
with the ambiguous quarter thrown out, evaluated on a model whose attention is
demonstrably on the lesion. It is a harder, more legitimate task measured more honestly.

Nothing here changes what should be said about the live model. **Keep
`ct_best_model_phase35_v2.pt` deployed; keep not citing its external-Malignant 12/12 as
generalization evidence** — §7(d) adds a new, independent reason for that, namely that
most of those slices do not contain a visible lesion to detect. And Phase A's deployment
no-go still holds unchanged: a nodule-crop classifier cannot be served by
`POST /predict/ct`, which takes a whole uploaded slice.

---

## 9. Go / no-go on Phase D

**Recommendation: GO on the evidence, NO-GO on the schedule. Do not start Phase D before
the exam. The nodule-crop line is a positive result and should be written up as the
investigation's conclusion, not extended.**

**Why the evidence says go.** Phase D's premise — that a detector could front-end this
classifier — requires the classifier to be worth front-ending. It is. That question was
genuinely open at the end of Phase B and is now closed with three independent pieces of
evidence: attention at 2.96× against a measured 1.05× baseline (§3); attention that
follows the lesion's *presence* rather than any frame artifact, proven by an off-centre
re-extraction with a clean Normal control (§4); and a style probe whose vendor channel
is dead at 0.363 (§5). "Is there a nodule in this patch" runs at 0.970. This is not the
more-sophisticated-negative-result outcome the brief allowed for. The earlier phases
kept producing models that were right for the wrong reasons; this one is right for the
right reason, within a narrow scope.

**Why the schedule says no-go, and this is not a close call.** §4 is the reason. The
model has **zero** localization ability — accuracy 0.816 → 0.034 the moment the lesion
moves 9 mm. Every bit of the burden of finding the lesion falls on the detector, and the
detector has to centre it to within a few millimetres, because that is the tolerance the
classifier actually has. That is not "add a detection stage"; that is building the harder
half of a two-stage system to a tolerance this phase has now measured and found tight.
Phase A costed a detector as "a larger project than everything in Phase A combined, and
not a ~10-day-with-an-exam undertaking" — §4 makes that estimate *more* pessimistic, not
less, because it converts an unknown tolerance into a demonstrated hard requirement.

There is also the deployment fact that has not moved since Phase A: `POST /predict/ct`
takes a whole uploaded slice, and the class semantics differ — "Normal" here means *this
28 mm patch has no nodule*, not *this patient is healthy*. Shipping this would change
what the application claims, not just how well it claims it.

**Concretely, what to do instead:** write this up. Phase C converts the project's CT
story from "our model reports 98% and we cannot explain why it fails externally" into a
documented, evidence-backed finding — the shortcut was real, it was structural, removing
it structurally *worked*, and the resulting model learns real lesion features but cannot
localize. That is a substantially stronger thing to present than either a fragile 98% or
an unfinished two-stage system.

**If Phase D is ever picked up after the exam**, the two things this phase says to do
first, cheaply, before touching a detector: (1) re-extract at a looser lung-fraction
floor to bound the Benign selection bias (§6) — currently the largest un-quantified
uncertainty in these numbers; (2) train with random translation augmentation of the crop
centre, which directly attacks the §4 position-lock and would relax the tolerance the
detector has to hit. (2) in particular is a one-line augmentation change and a 9-minute
run, and it might change the Phase D cost estimate materially.

---

## 10. Live-system integrity

Re-hashed after all training and evaluation, not asserted:

```
checkpoints/ct_best_model_phase35_v2.pt  sha=93b3a979884edbc4  mtime=1787555730.4296012
backend/main.py                          sha=780120553178c0a5  mtime=1788684518.831581
data/ct_scan/                            n=1036  aggregate=f6f64d2a35e2e525
```

All three match the values recorded in `ct_framing_normalization_experiment.md` §8 and
the snapshot taken before this phase began — unchanged. No existing checkpoint was
modified; the only checkpoint written is the new
`checkpoints/ct_lidc_nodulecrop_v1.pt`. All other writes are confined to
`eval_run/phaseC/`. Disk added: **51 MB**. No process left running.

---

## 11. Artifacts

All under `eval_run/phaseC/` unless noted. Nothing is referenced by `backend/`.

| path | what |
|---|---|
| `checkpoints/ct_lidc_nodulecrop_v1.pt` | **the new experimental checkpoint** (epoch 7, val bal 0.8528) — not deployed |
| `lidc_data.py` | manifest-backed Dataset; reuses `ml/dataset.py` transforms |
| `train_lidc_crops.py`, `train_run.log` | training, adapted from `ml/train.py` |
| `training_history.json`, `training_curves.png` | per-epoch metrics incl. per-class recalls |
| `ct_lidc_nodulecrop_v1_valacc.pt` | the best-val-*accuracy* epoch (12), kept so the selection choice is auditable |
| `eval_lidc_test.py`, `eval_test_results.json` | §2, incl. per-image predictions and all subset slices |
| `confusion_matrix_test.png` | §2 figure |
| `gradcam_lidc.py`, `gradcam_results.json` | §3, all 505 crops + untrained-head control |
| `gradcam_pngs/` (505), `gradcam_grid.png` | §3 overlays |
| `offcentre_probe.py`, `offcentre_results.json`, `offcentre_pngs/` | §4, the decisive probe |
| `offcentre_control.py`, `offcentre_control_results.json` | §4 Normal control + re-extraction fidelity check |
| `style_probe.py`, `style_probe_results.json` | §5 |
| `external_scale_check.py`, `external_scale_check.json` | §7 (a)–(c), the measurements |
| `external_eval.py`, `external_eval_results.json`, `external_crops/` | §7 external Normal probe |

Stopping here as instructed — **Phase D not started.**
