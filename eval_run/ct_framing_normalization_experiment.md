# Framing-normalization experiment: does removing the acquisition-framing confound produce genuine lesion learning?

**Status: experimental, nothing deployed.** `checkpoints/ct_best_model_phase35_v2.pt`,
`backend/main.py` and `data/ct_scan/` are untouched — verified by checksum in §7, not
merely asserted.

This continues `ct_normal_root_cause.md` → `ct_retrain_experiment.md` →
`ct_shortcut_inversion_analysis.md`. Those established that the CT classifier never
detected lesions across domains: both the incumbent `phase35_v2` and the retrained
`phase36_normal_fix` key on **acquisition framing style** (scanner-table visibility,
full-FOV vs thorax-crop, background composition), which is confounded with class label
in opposite directions in the two training sets. This experiment removes that cue from
the input and retrains, to answer the question the previous writeup left open: *is there
any lesion-level signal in this architecture and dataset size once the framing confound
is controlled for?*

## Verdict up front

**Outcome (c): neither.** Removing the framing confound did **not** produce a model that
generalizes on held-out external data, and did **not** produce lesion-relevant attention.

The confound removal itself worked and is measurable: a random forest could predict the
class from global style statistics alone at **94.4% balanced accuracy** on the original
training data and is at **exactly chance (33.3%)** after normalization; scanner-table
detection inside the training set went from **99.8% of images to 0%**. So this is not a
"the fix wasn't applied properly" failure.

What the retrained model does instead: it calls **224 of 285 external images Malignant
(78.6%)** when only 12 of them (4.2%) are, with median confidence 0.93. That is the same
"not-IQ-OTH-like ⇒ Malignant" behaviour as the incumbent `phase35_v2`, just arrived at
through a different cue. Its headline external-Malignant score of 12/12 is therefore
**still not evidence of malignancy detection** — the external Normal images from the same
source are called Malignant at nearly the same rate.

Grad-CAM confirms it, and adds a finding the earlier writeups did not have: even
**in-domain**, where the model is 97.1% accurate, attention is diffuse — lung-field
Grad-CAM enrichment is **1.00×** on test Malignant (1.0 = no preference at all over what
area share alone would give) and ~23% of the heatmap mass falls on background pixels that
are *constant across every image in the dataset*.

Useful negative result: this rules out framing/style as the *sole* cause. Something else —
most plausibly that 829 single-source training images cannot support lesion-level features
— is doing the work. See §5 and §6.



---

## 1. The framing-normalization pipeline

Code: [`eval_run/framing/ct_framing_normalize.py`](framing/ct_framing_normalize.py).
Applied identically to every image; **no step branches on which dataset an image came
from**. There is no CT-specific segmentation model in this project
(`backend/seg_utils.py`'s U-Net is X-ray-only), so this is a thresholding/contour
approach, which the task brief accepts.

| # | step | why |
|---|---|---|
| 1 | grayscale | — |
| 2 | background level `bg` = median of the 1-px outer frame | IQ-OTH/NCCD renders background at ~37–38, external sources at 0. Estimating it per image is what makes everything downstream source-agnostic. |
| 3 | upper level `hi` = 99.5th percentile | per-image dynamic range |
| 4 | adaptive threshold `thr = bg + max(6, 0.18·(hi−bg))` | `scripts/ct_table_shortcut_audit.py` hardcodes 150; that does not transfer — some maftouni slices have a 99th percentile of **29**, some external slices of **166**. A fixed absolute threshold detects nothing on half the corpus. |
| 5 | `image > thr`, 3×3 opening | kills speckle and the thin diagonal annotation leader lines present in ~1/3 of the mohamedhanyyy images |
| 6 | largest connected component | **this is what drops the scanner couch** — it is separated from the patient's back by an air gap, so it is a different component. Also drops the circular gantry-FOV vignette ring in the external images. |
| 7 | `binary_fill_holes` | re-includes lungs / airways / mediastinal air, which step 5 excluded for being dark |
| 8 | closing with disk(9), fill again | smooths the silhouette so mask-edge serration does not become its own texture cue |
| 9 | crop to mask bbox | full-FOV and pre-cropped images end up framed on the body identically |
| 10 | resample the body box to a **canonical aspect** (1.45, the corpus median) then pad to square | see below |
| 11 | **fixed elliptical support** inscribed in the canonical box, identical pixels for every image | see below |
| 12 | **exact histogram specification**: pixels in that support rank-transformed onto a uniform distribution over [16, 255], ties broken by deterministic sub-level dither | see below |
| 13 | outside the ellipse → 0; output 256×256 PNG | — |

### Why steps 10–12 exist (they were not in the first design)

The first version stopped at "mask, crop to bbox, pad to square, resize". Step 2 of this
experiment — the pre-retrain leak check — showed that version **still leaked the origin at
~99% accuracy**, so it was iterated on three times before any training was run. Each
iteration was driven by a measured leak:

* **v1 → v2.** Origin still recoverable at 98.7%. Cause: black-padding fraction (IQ-OTH
  body boxes average aspect **1.56 ± 0.20**, external **1.25 ± 0.17** — these renders are
  anisotropic in different ways) plus the in-body intensity histogram (IQ-OTH std 93.3 vs
  external 80.8; fraction of pixels >200: 0.184 vs 0.094). Fix: canonical aspect (step 10)
  + percentile→uniform histogram equalization.
* **v2 → v3.** Origin still recoverable at 98.2%, now through a single scalar: **body
  silhouette fill** (mask area / box area, 0.78 IQ-OTH vs 0.85 external), because the
  external sources' circular reconstruction FOV truncates the chest wall and makes their
  silhouettes boxier. Every surviving intensity statistic was an algebraic proxy for that
  one number. Fix: intersect with a common ellipse.
* **v3 → v4.** Origin *still* recoverable at 98.2%, top feature `max` at AUC 0.968:
  IQ-OTH mean max 253.8 ± 0.5, external 254.9 ± 0.2. This was pure quantization
  bookkeeping — averaging ranks over tied intensities leaked the sources' **saturation
  structure** (IQ-OTH renders clip large areas to 255). Fix: make the support fixed rather
  than per-image, and break ties with dither so the output histogram is exactly uniform.

After v4 every global pixel statistic is **byte-identical for every image in the corpus**
(mean 72.551, std 84.196, zero-fraction 0.46259, min 0, max 255 — verified directly on
both origins). Visual confirmation: [`eval_run/framing/qc_pairs.png`](framing/qc_pairs.png)
(4 before/after pairs at full size) and
[`eval_run/framing/qc_contact.png`](framing/qc_contact.png) (42 images, 7 source groups,
original row above normalized row).

### Failure modes and limitations of this pipeline

Stated because they bound how far the result can be trusted:

* If the couch touches the patient's back with no air gap, it joins the torso component
  and survives. (Not observed in this corpus — table detection is 0/1036 after — but the
  mechanism is a threshold heuristic, not anatomy.)
* Arms inside the FOV join the torso component and widen the bbox.
* **Steps 10–12 deliberately destroy real information.** Body proportions are gone
  (canonical aspect), absolute density is gone (rank equalization preserves ordering, not
  values), and a thin rim of chest wall is clipped from images whose body over-fills the
  ellipse. A model trained on this cannot use lesion *size in cm* or *density in HU* as
  evidence — only relative density and spatial pattern. HU meaning was already
  unrecoverable: these are pre-windowed 8-bit exports with different, unrecorded window
  settings per source.
* Rank equalization amplifies noise in large flat soft-tissue regions (visible grain in
  the QC images). The grain is statistically identical across sources by construction, so
  it is not itself a source cue, but it is added texture the model sees.
* It segments "bright blob", not "thorax". 1,321/1,321 images passed without a single
  degenerate mask (`n_failed = 0`), but that is an empirical result on this corpus, not a
  guarantee.

---

## 2. Verifying the confound is gone — *before* retraining

Code: [`eval_run/framing/ct_framing_leakcheck.py`](framing/ct_framing_leakcheck.py).
Raw: [`eval_run/framing/leakcheck.json`](framing/leakcheck.json),
log [`eval_run/framing/leakcheck_v4.log`](framing/leakcheck_v4.log).
Four probes, each run on the original images and again on the normalized ones.
1,036 IQ-OTH + 285 external images; 5-fold stratified CV.

### A. Table/couch detector

`scripts/ct_table_shortcut_audit.py`'s method (threshold → largest body component → look
for a bright structure below the body's lower edge), with its hardcoded 150 replaced by
the same per-image adaptive threshold, so it is meaningful on external images too.

| | BEFORE | AFTER |
|---|---|---|
| IQ-OTH Benign | 120/120 (100.0%) | **0/120 (0.0%)** |
| IQ-OTH Malignant | 499/500 (99.8%) | **0/500 (0.0%)** |
| IQ-OTH Normal | 415/416 (99.8%) | **0/416 (0.0%)** |
| IQ-OTH all | 1034/1036 (99.8%) | **0/1036 (0.0%)** |
| external all | 135/285 (47.4%) | 6/285 (2.1%) |

The 6 residual external detections are detector artifacts, not couches — after
normalization there is nothing outside the ellipse but a constant, so a "bright structure
below the body" can only be mis-segmentation within the body.

### B. Trivial origin classifier — global style statistics only

20 cheap global features (dimensions, aspect, background pixel counts at three
thresholds, mean/std/min/max, seven percentiles, table fraction). Logistic regression and
random forest.

| subset | BEFORE (bal. acc / AUC) | AFTER (bal. acc / AUC) | chance |
|---|---|---|---|
| all IQ-OTH vs all external | 0.9995 / 1.000 | **0.522 / 0.522** | 0.50 |
| Normal only | 0.9988 / 1.000 | **0.522 / 0.522** | 0.50 |
| Malignant only | 1.000 / 1.000 | **0.500 / 0.500** | 0.50 |

Honest caveat: after step 12 these features are constant by construction, so this probe
being at chance is *guaranteed*, not evidence. That is why probes C and D exist.

### C. Class-from-style **inside the training data** — the probe that actually matters

The learnable shortcut is not "can something tell the sources apart"; the model only
ever sees IQ-OTH data during training. The shortcut is "can global style statistics
predict the **class** within that training data" — the thing that was true before (89.3%
table-visible for Malignant vs ~100% for Normal/Benign, per PROJECT_STATUS.md §6).

| | BEFORE acc / bal. acc | AFTER acc / bal. acc | chance bal. acc |
|---|---|---|---|
| logistic regression | 0.700 / 0.686 | 0.345 / **0.3333** | 0.3333 |
| random forest | 0.962 / **0.944** | 0.345 / **0.3333** | 0.3333 |

**This is the decisive result of step 2.** A random forest could recover the class from
nothing but global style statistics at 94.4% balanced accuracy before normalization. After
normalization it is at exactly chance. The style→label shortcut is not reduced; it is
arithmetically unavailable.

### D. Spatial origin probe — the residual that remains

32×32 downsampled pixels, i.e. actual image content rather than global statistics.

| subset | BEFORE (bal. acc) | AFTER (bal. acc) |
|---|---|---|
| all IQ-OTH vs all external | 1.000 | **0.968** |
| Normal only | 1.000 | **0.990** |

**The origin is still recoverable from spatial content, and this experiment does not fix
that.** It is also not obviously fixable and not obviously the same problem: different
sources genuinely scan different patients, at different slice levels, with different
reconstruction kernels. The measurable residual traceable to framing is that external
bodies still fill slightly more of the ellipse rim (body fraction of support 0.988 vs
0.951) — masking cannot restore chest wall that was never in the source frame. So:
domain shift remains; what has been removed is the *class-correlated framing shortcut*
inside the training distribution.

---

## 3. The rebuilt dataset

Built by [`eval_run/framing/build_framing_dataset.py`](framing/build_framing_dataset.py)
into `data/ct_scan_framing_normalized/` (new directory; `data/ct_scan/` untouched).
**Splits are carried over verbatim** — nothing is re-split, so the leakage-aware split and
the 61-image protocol-mismatch exclusion already baked into `data/ct_scan/` are preserved,
and the test split is the same images as every previous experiment's.

| split | Benign | Malignant | Normal | total |
|---|---|---|---|---|
| train | 96 | 400 | 333 | 829 |
| val | 12 | 50 | 42 | 104 |
| test | 12 | 50 | 41 | 103 |

External sets, normalized through the identical pipeline into
`eval_run/framing/external_normalized/`:

| set | n | note |
|---|---|---|
| `ext_malignant` | 12 | the documented external Malignant set |
| `ext_normal` | 9 | the documented external Normal failure set |
| `heldoutB_mohamedhanyyy` | 99 | Group B from `ct_retrain_experiment.md`, located at `eval_run/ct_normal_heldout/` |
| `heldoutB_maftouni` | 165 | ditto |

1,321 images processed, **0 failures**, manifest with per-image mask diagnostics in
[`eval_run/framing/build_manifest.json`](framing/build_manifest.json).

Note on what changed vs. the phase36 experiment: this run adds **no new training data at
all**. Training is the same 829 IQ-OTH images as the incumbent — only the preprocessing
differs. That is deliberate: it isolates the framing variable.

---

## 4. Training

`ml/train.py` **unchanged**. Identical methodology to both previous runs: densenet121,
15 epochs, 2 frozen-backbone warmup epochs at lr 1e-3 then fine-tune at 1e-4, batch 32,
seed 42, inverse-frequency weighted CrossEntropyLoss. Only `--data_dir` differs.
Driver: [`eval_run/framing/run_phase37.sh`](framing/run_phase37.sh),
log [`eval_run/framing/phase37_run.log`](framing/phase37_run.log).

Class weights: Benign 2.879 / Malignant 0.691 / Normal 0.830 (identical class counts to
the incumbent, since no data was added).

| run | data dir | checkpoint | best val_acc | best epoch |
|---|---|---|---|---|
| phase37 | `data/ct_scan_framing_normalized/` | `checkpoints/ct_best_model_phase37_framing_normalized.pt` | 1.0000 | 8 |

Curves `ml_logs/ct_phase37_framing_normalized_{curves.png,history.json}`. Convergence was
noisier than the previous runs (val_acc oscillates 0.85–1.00 between epochs 5–15) —
consistent with a harder input representation after density magnitude and body size were
removed — but nothing pathological, no class collapse.

**Reproduction check** (same logic as `backend/models.py::load_model_bundle`, 0.001
tolerance): claimed 1.0000, reproduced 1.0000, diff 0.0000 → **PASS**.

Caveat on that 1.0000: the val split is 104 images and best-epoch selection is on val, so
val accuracy is optimistically biased and is not an independent number. The test-split
figures below are the ones to read.

---

## 5. Results — three checkpoints, full counts

All from `eval_run/ct_ckpt_eval.py`. Each checkpoint is evaluated through **its own**
preprocessing — `phase35_v2` and `phase36` on the original images, `phase37` on the
framing-normalized ones — which is the correct comparison, since preprocessing is the
intervention under test. The test split is the same 103 images in all three cases.

Raw per-image predictions: [`results_ct_retrain_before.json`](results_ct_retrain_before.json),
[`results_ct_retrain_after.json`](results_ct_retrain_after.json),
[`eval_run/framing/results_ct_phase37.json`](framing/results_ct_phase37.json).

### (a) In-dataset test set (`data/ct_scan/test`, n=103)

| | `phase35_v2` (live) | `phase36_normal_fix` | `phase37_framing_normalized` |
|---|---|---|---|
| Benign | 10/12 (83.3%) | **12/12 (100%)** | 10/12 (83.3%) |
| Malignant | **50/50 (100%)** | **50/50 (100%)** | 49/50 (98.0%) |
| Normal | **41/41 (100%)** | 40/41 (97.6%) | **41/41 (100%)** |
| **overall** | 101/103 (98.1%) | **102/103 (99.0%)** | 100/103 (97.1%) |

`phase37` is the weakest of the three in-dataset, by 1 image vs the incumbent. Expected:
the pipeline deliberately discards lesion size and density magnitude.

### (b) External Malignant (n=12, `mohamedhanyyy` adenocarcinoma / large-cell / squamous)

| | `phase35_v2` | `phase36_normal_fix` | `phase37_framing_normalized` |
|---|---|---|---|
| correct | **12/12 (100%)** | 0/12 (0%) | **12/12 (100%)** |
| predictions | Malignant ×12 | Normal ×12 | Malignant ×12 |
| median confidence | 0.98 | 0.90 | **1.00** |

### (c) External Normal

| | `phase35_v2` | `phase36_normal_fix` | `phase37_framing_normalized` |
|---|---|---|---|
| original failure set (n=9) | 0/9 | **7/9** | 2/9 |
| — predictions | Mal ×7, Ben ×2 | Nor ×7, Mal ×1, Ben ×1 | Mal ×6, Nor ×2, Ben ×1 |
| Group B mohamedhanyyy (n=99, 9 patients) | 0/99 | **85/99** | 27/99 |
| — predictions | Mal ×63, Ben ×36 | Nor ×85, Ben ×13, Mal ×1 | Mal ×62, Nor ×27, Ben ×10 |
| Group B maftouni (n=165, 44 patients) | 0/165 | **165/165** | 20/165 |
| — predictions | Mal ×164, Ben ×1 | Nor ×165 | Mal ×144, Nor ×20, Ben ×1 |
| **Group B total (n=264, 53 patients)** | **0/264 (0%)** | **250/264 (94.7%)** | **47/264 (17.8%)** |

### The number that decides it

Across all 285 external images (12 truly Malignant, 273 truly Normal):

| | images predicted Malignant | true prevalence |
|---|---|---|
| `phase35_v2` | 246/285 (86.3%) | 4.2% |
| `phase36_normal_fix` | 2/285 (0.7%) | 4.2% |
| `phase37_framing_normalized` | **224/285 (78.6%)** | 4.2% |

`phase37`'s 12/12 on external Malignant is produced by a model that calls 4 out of every 5
external images malignant. It is the incumbent's artifact, reproduced. `phase36` is the
mirror image — it has a believable base rate but catches 0 of the 12 real cancers.

Compared to the incumbent it was meant to fix, `phase37` is a real but small improvement on
external Normal (0/264 → 47/264) at the cost of 1 in-dataset image — nowhere near enough to
matter, and not attributable to lesion learning, since the improvement is spread across
images the model is still confidently wrong about elsewhere (median confidence on external
predictions 0.93).

---

## 6. Grad-CAM — does it attend to plausible lesion regions?

Code: [`eval_run/framing/gradcam_phase37.py`](framing/gradcam_phase37.py).
Overlays: `eval_run/framing/gradcam_phase37_pngs/` (21 PNGs, same pattern as
`eval_run/ct_shortcut_gradcam_pngs/`), 6-panel summary
[`eval_run/framing/gradcam_phase37_grid.png`](framing/gradcam_phase37_grid.png).
Raw: [`gradcam_phase37_results.json`](framing/gradcam_phase37_results.json).

Rather than only describing heatmaps in prose (which is how the previous writeup reached
its conclusion), attention is quantified as **enrichment**: the fraction of Grad-CAM mass
falling in a zone, divided by that zone's share of body area. 1.00× means the heatmap is
no more concentrated there than sprinkling it uniformly would be. The lung field is
approximated as the darkest 30% of in-body pixels (air density), the rim as body pixels
within ~8px of the mask edge.

| set | n | correct | lung-field enrichment | rim enrichment | CAM mass outside body |
|---|---|---|---|---|---|
| external Malignant | 12 | 12 | **1.07×** | 0.69× | 19% |
| external Normal | 9 | 2 | **0.90×** | 0.73× | 24% |
| **in-dataset test Malignant** | 50 | 49 | **1.00×** | — | 23% |
| **in-dataset test Normal** | 41 | 41 | **1.07×** | — | 24% |
| **in-dataset test Benign** | 12 | 10 | **1.04×** | — | 20% |

Reading this honestly:

* **No lesion localization on external Malignant.** 1.07× is indistinguishable from no
  preference. Visually the heatmaps are broad washes over mediastinum + heart + both
  lungs; no discrete focus anywhere a peripheral nodule would be. This is the same
  qualitative picture as `ct_shortcut_inversion_analysis.md` found for both earlier
  checkpoints, so on this measure nothing improved.
* **Good news, narrowly:** the *specific* residual this experiment could have introduced —
  a mask-edge artifact from the new masking — did **not** appear. Rim enrichment is 0.69–0.73×,
  i.e. the model attends to the mask boundary *less* than area share. The pipeline did not
  swap one edge cue for another.
* **The in-dataset control is the important row.** The model gets 97.1% in-dataset while
  its attention is at 1.00–1.07× enrichment and ~23% of the heatmap mass sits on
  *constant* background pixels outside the elliptical support — pixels that are byte-identical
  in all 1,321 images and carry zero information. Whatever supports the 97.1%, it is not
  a spatially localized lesion feature.
* **Caveat on the method:** Grad-CAM here is read off densenet121's final feature map,
  which is 7×7 for a 224×224 input — each cell covers 32×32 px. That resolution genuinely
  cannot resolve a small nodule, and mass "outside the body" is partly bleed from that
  coarseness plus upsampling. So "no localization" is somewhat overstated by the metric.
  It is not overstated enough to rescue the conclusion: a model keying on a real lesion
  would still show enrichment well above 1.0 at 7×7, and it does not, on any set,
  including ones it classifies correctly.

---

## 7. Assessment and recommendation

### Did it work?

**No, on both counts asked.**

(a) *Generalizes better on held-out external data?* Marginally and misleadingly. External
Normal 0/264 → 47/264 vs the incumbent, external Malignant unchanged at 12/12,
in-dataset 101/103 → 100/103. But 78.6% of all external images are still called Malignant
at median confidence 0.93 against a true prevalence of 4.2%. That is not generalization;
it is the incumbent's out-of-distribution reflex with a slightly weaker bias.

(b) *Attends to plausible lesion regions?* No. 1.07× lung enrichment on external
Malignant, and — the finding that generalizes past this experiment — 1.00× in-domain on
cases it classifies correctly.

### What was nonetheless established

1. **The framing confound is removable, and removing it is not sufficient.** The previous
   writeup identified table visibility / FOV framing as "specific and potentially
   addressable". It was addressable — class-from-style went 94.4% → 33.3% balanced
   accuracy, table detection 99.8% → 0% — and addressing it did not produce lesion
   learning. The hypothesis that framing was the blocking cause is **falsified**.
2. **`phase36`'s inversion is now explained more sharply.** With framing gone, the model
   *still* reverts to "external ⇒ Malignant". So the direction of the shortcut is set by
   whichever cue survives, and *any* residual domain difference will be recruited. Probe D
   shows a large one remains: origin is recoverable from 32×32 downsampled pixels at 96.8%
   balanced accuracy even after normalization. Framing was one channel among several, not
   the channel.
3. **The in-domain Grad-CAM control is new evidence about the incumbent.** A model can hold
   97–98% on this test split with attention that is statistically indistinguishable from
   uniform. This is a reason to distrust the in-dataset CT numbers themselves, not only the
   external ones.

### What remains uncertain

* **Dataset size is now the leading unexplained factor, but is not proven.** 829 training
  images from a single source, 96 of them Benign. This experiment removed the one
  alternative explanation that had been concretely demonstrated; it did not positively
  demonstrate the size explanation. That would need, at minimum, a learning-curve study or
  a multi-source Malignant training set.
* **The masking pipeline's own trade-offs are real and unquantified.** Removing density
  magnitude and body size removes genuine diagnostic information. Part of the 1-image
  in-dataset regression, and possibly more, is the cost of the intervention rather than a
  property of the problem. A gentler variant (mask + table removal only, keeping density)
  was not tried and would separate these.
* **External validation is still thin.** 12 Malignant images from one Kaggle source, and
  Normals from two. No external Benign at all. Group B is 53 patients.
* **Grad-CAM at 7×7 is a coarse instrument** (see §6 caveat).
* **Single seed, single run.** Val accuracy oscillated 0.85–1.00 across late epochs, so
  run-to-run variance on a 104-image val split is likely non-trivial and was not measured.

### Recommendation

**Do not adopt `phase37`. Do not adopt `phase36`. Keep `phase35_v2` deployed as-is — while
noting that this experiment further undermines the case for the CT model's reported
accuracy, rather than supporting it.**

Concretely, for the project's own reporting: the recommendation from
`ct_retrain_experiment.md` §6 stands and is strengthened — **stop citing external Malignant
12/12 as evidence of CT generalization.** Three checkpoints now produce that number or its
inverse for reasons unrelated to malignancy. The in-dataset CT figures should carry an
explicit caveat too, given §6's in-domain attention result.

If there is appetite for another attempt, the options in rough order of expected value:

1. **Multi-source Malignant training data** (breaking source↔label correlation on *both*
   sides, as `ct_retrain_experiment.md` already proposed). This remains untried and is the
   only intervention that addresses probe D's residual. Cost: `mohamedhanyyy`'s carcinoma
   subtypes are the obvious candidate and using them consumes the external Malignant test
   set, so a replacement external set is needed first.
2. **A learning-curve study** on the existing data to test the dataset-size hypothesis
   cheaply before investing in (1).
3. **A gentler normalization** (mask + table removal, keeping density and body size) to
   measure how much of the in-dataset regression this pipeline caused.

None of these are started. Nothing further will be done without a decision.

---

## 8. Confirmation that the live system is untouched

Verified by re-hashing, not asserted. Snapshot taken before any work
([`eval_run/framing/protected_before.json`](framing/protected_before.json)) and re-checked
after training and evaluation, via
[`eval_run/framing/protected_snapshot.py`](framing/protected_snapshot.py):

```
OK   checkpoints/ct_best_model_phase35_v2.pt  sha=93b3a979884edbc4
     mtime_before=1787555730.4296012  mtime_now=1787555730.4296012
OK   backend/main.py                          sha=780120553178c0a5
     mtime_before=1788684518.831581   mtime_now=1788684518.831581
OK   data/ct_scan   n=1036->1036
     aggregate_sha256=97e1a0506a74782c->97e1a0506a74782c
     max_mtime=1787498580.7406175->1787498580.7406175
RESULT: ALL UNTOUCHED
```

`data/ct_scan`'s aggregate hash is over the SHA-256 of all 1,036 files plus their paths and
sizes; the file count, the aggregate, and the newest mtime in the tree are all unchanged,
so no file was added, removed, modified or rewritten.

The phase36 checkpoints were also not touched (`ct_best_model_phase36_normal_fix.pt`
sha `c1cb8ab061e71dc3`, `..._mfonly.pt` sha `1f542951ae27e618`, both retaining their
pre-existing mtimes from the earlier experiment).

`backend/main.py`'s `CHECKPOINTS` dict still points at `ct_best_model_phase35_v2.pt`; the
new checkpoint is not referenced anywhere in `backend/`, and the live backend was never
started against it.

---

## 9. Artifacts

Experimental only — nothing below is referenced by `backend/main.py`.

| path | what |
|---|---|
| `checkpoints/ct_best_model_phase37_framing_normalized.pt` | new experimental checkpoint |
| `checkpoints/ct_phase37_framing_normalized_best.pt` | `ml/train.py`'s own output (same weights) |
| `data/ct_scan_framing_normalized/` | normalized dataset, splits carried over verbatim |
| `eval_run/framing/external_normalized/` | normalized external + Group B sets |
| `eval_run/framing/ct_framing_normalize.py` | the pipeline |
| `eval_run/framing/build_framing_dataset.py` | dataset build |
| `eval_run/framing/ct_framing_leakcheck.py` | step-2 verification, probes A–D |
| `eval_run/framing/diagnose_leak.py` | per-feature leak attribution (drove v1→v4) |
| `eval_run/framing/gradcam_phase37.py` | Grad-CAM + attention-enrichment metric |
| `eval_run/framing/protected_snapshot.py` | integrity snapshot/check for §8 |
| `eval_run/framing/run_phase37.sh` | train + eval driver |
| `eval_run/framing/leakcheck.json`, `leakcheck_v4.log` | verification results |
| `eval_run/framing/build_manifest.json` | per-image mask diagnostics, 1,321 records |
| `eval_run/framing/results_ct_phase37.json` | per-image predictions |
| `eval_run/framing/gradcam_phase37_results.json` | per-image attention metrics |
| `eval_run/framing/gradcam_phase37_pngs/` | 21 overlays |
| `eval_run/framing/gradcam_phase37_grid.png` | 6-panel summary |
| `eval_run/framing/qc_contact.png`, `qc_pairs.png` | before/after visual QC |
| `eval_run/framing/phase37_run.log` | training + eval log |
| `ml_logs/ct_phase37_framing_normalized_{curves.png,history.json}` | training curves |

Stopping here as instructed — no integration, no deployment, awaiting a decision.

