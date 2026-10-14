# Phase F: multi-slice ensembling at inference time — does it fix the through-plane fragility?

**Scope: one bounded check, no training, no new model.** Phase E.1 root-caused the two-stage
pipeline's 60.7% end-to-end accuracy to a specific defect: `v2_jitter` is stable in-plane and
collapses through-plane (40.5% of predictions flip when the crop moves to an adjacent slice). This
phase tests the cheapest possible fix for that defect — **run the same unmodified classifier on N
adjacent slices and combine the answers** — against the **exact same 84 nodules / 45 patients** Phase
E.1 measured. Everything written is confined to `eval_run/phaseF/` (1.1 MB, 4 Python files + JSON).
**`eval_run/phaseE/pipeline.py`, `backend/main.py` and `checkpoints/ct_best_model_phase35_v2.pt` are
untouched — re-hashed in §6, all five Phase E §7 digests match and both mtimes are unchanged.** No
checkpoint written, no dependency added, no DICOM downloaded, nothing deployed, no process left
running.

---

## Verdict up front

**It does not work, and the mechanism check says it cannot work in this form.** This is answer **(c)**
from the brief, and the evidence for it is stronger than the aggregate numbers alone.

1. **No strategy beats the baseline. The best result is a tie.** `N=3 majority vote` scores
   **51/84 (60.7%), Wilson [0.500, 0.705]** — *identical* to Phase E.1's single-slice 51/84. Every
   other variant is worse: N=3 mean **49/84 (58.3%)**, N=5 mean **36/84 (42.9%)**, N=5 vote 37/84,
   and max-confidence is catastrophic at 31/84 and 28/84. **Nothing approaches ~70%.** §1.
2. **The tie is not "no change" — it is 5 predictions swapped, 2 each way.** Paired against the
   baseline, N=3 vote changes 5 of 71 matched predictions: 2 wrong→right, 2 right→wrong, 1 neutral.
   The point estimate is unmoved because the flips cancel, not because the ensemble reproduces the
   baseline. §1.2.
3. **The mechanism check is the finding, and it is a clean refutation.** Re-running Phase E.1's
   through-plane sweep with ensembling: at **2 mm** the single displaced slice scores 0.488 and the
   N=3 ensemble scores **0.482**; at 5 mm 0.226 → 0.226; at 8 mm 0.101 → 0.083. **It does not help at
   the smallest displacement either** — so this is not even the "patches small errors only" case the
   brief asked about. §3.
4. **The conditional test kills the hypothesis outright.** The hypothesis was "if the true centre
   slice is among the N classified, its answer should be recoverable". At 2 mm the true slice IS in
   the N=3 window 48.8% of the time — and on exactly those crops the ensemble scores **0.549, the
   same 0.549 the single displaced slice already scored**. Widen to N=5, where the true slice is
   present 73.8% of the time, and accuracy *falls* from 0.492 to **0.395**. **The true slice's vote is
   present and gets outvoted.** §3.2.
5. **Why: the neighbouring slices are not noisy-but-unbiased, they are systematically worse and
   systematically biased toward `Normal`.** On perfectly centred windows, per-slice accuracy is
   **0.738 at the true slice, 0.601 at ±1, 0.375 at ±2**, and the `Normal` rate climbs **10.7% →
   24.4% → 52.4%**. Averaging a 0.74 classifier with two 0.60 classifiers and two 0.38 classifiers
   does what arithmetic says it does. §4.1.
6. **New failure introduced, exactly as the brief anticipated: averaging washes out confident correct
   answers.** On GT-centred crops, N=5 mean **breaks 19 nodules that the single true-centre slice got
   right — 10 of them at ≥0.90 confidence — and fixes 2**. The classifier's own ceiling drops from
   **62/84 (0.738) to 45/84 (0.536)** under N=5. §4.2.
7. **It also damages the axis that was working.** In-plane accuracy at 2 mm, which single-slice held
   flat at 0.744, becomes 0.720 at N=3 and **0.509 at N=5**. Ensembling trades a working axis for no
   gain on the broken one. §3.3.
8. **Scan-level moves by one scan, inside noise, and for the wrong reason.** 23/45 → **24/45** for
   both N=3 variants ([0.391, 0.671] against the baseline's [0.370, 0.650]); Benign scans over-called
   Malignant 19 → **17**. That comes from candidates drifting toward `Normal`, the same drift that
   costs nodules. §2.
9. **Cost is genuinely free, which is the one clean result.** N=5 stage 2 is **0.129 s mean per scan**
   (10–20 crops), total latency **33.1 s** against the baseline's 33.2 s, **peak VRAM 0.751 GB of
   4.00 — bit-identical to Phase E/E.1** — and host RSS is unchanged to within 0.01 GB per scan. §5.

**Recommendation (§7): (c), it doesn't help. Do not pursue inference-time ensembling.** The
hypothesis was well-formed and cheap and it has been properly falsified rather than merely
underpowered.

---

## 0. What was run, what was reused, and how the reuse was verified

| | Phase E.1 | **Phase F** |
|---|---|---|
| nodules / patients | 84 / 45 | **the same 84 / 45** |
| detector | run on all 45 volumes | **not re-run — its stored candidates are reused** |
| classifier crops | 1,176 (sweep) | **9,123** |
| runtime | ~2 h download + ~25 min inference | **4.4 min** (crops + inference) + 3.5 min latency probe |

**Reused, not rewritten** — the standing rule since Phase A's coordinate bug:

* `phaseE/pipeline.py` — `crop_at_ras`, `crop_to_pil`, `ras_to_series_rc`, `load_classifier`,
  `aggregate`, `TwoStagePipeline`, `CLASS_NAMES`, `DEFAULT_CKPT`: **imported, and the file is not
  edited** (its digest is recorded in §6 for the first time so future phases can check it too).
* `phaseD/volume_io.py` — `read_series`, `annotation_to_ras`, `LPS_TO_RAS`, verbatim.
* Phase E.1's stability-sweep methodology (`stability_sweep.py`) — same 2 / 5 / 8 mm magnitudes, same
  ±S through-plane and ±R/±A in-plane direction split, same flip-vs-0 mm-prediction definition.
* Phase E/E.1's per-nodule outcomes and matching (8 mm nearest candidate, per-nodule, independent) —
  read from `e2e_results.json` / `e2e_results_e1.json`. **Detection is byte-for-byte unchanged: 71/84
  detected, the same 13 misses.**

**The one new operation** is `ensemble.shift_to_slice(ras, meta, dk)`: re-project a RAS point onto the
DICOM slice `dk` slices away, keeping x/y fixed. It is deliberately implemented as a **translation
along the slice normal**, so the shifted point goes back through Phase E's own `crop_at_ras`
unchanged — **no coordinate maths is reimplemented**. Because the shift is purely normal-direction
and the in-plane basis vectors are orthogonal to the normal, `dk=0` must reproduce `crop_at_ras`
exactly, and it does: **84/84 GT windows have a dk=0 crop with the identical slice index and identical
integer pixel box**. The step is always the volume's **own** slice spacing, never an assumed mm value
— these 45 series run **0.5 to 3.0 mm** (median 1.25), so an N=5 window spans **2.0 mm on the thinnest
series and 12.0 mm on the thickest**. That variation is itself part of the result (§3.4).

### Reusing the detector's stored output, verified rather than assumed

Re-running the detector would have cost ~25 min and changed nothing (it is deterministic and
untouched), so its 229 stored candidates were reused. That is only legitimate if the crops reproduce:

| check, 229 candidates | result |
|---|---|
| dk=0 crop lands on the stored slice index | **229/229** |
| dk=0 crop's integer pixel box equals the stored box | **227/229** |
| dk=0 predicted **class** equals the stored class | **229/229** |
| max abs difference in any stored probability | 0.0014 |

The 2 box mismatches come from the stored crop centre being rounded to 0.01 px; neither changes the
predicted class. **The residual probability difference is not a crop difference — it is this GPU.**
Feeding *the identical tensor* through the classifier at batch size 1, 5 and 64 returns
`[0.5948, 0.5956, 0.5956]` for P(Benign) on the same crop: cuDNN picks different kernels per batch
shape. **That is why the N=1 baseline in every table below is recomputed from this run's own dk=0
forward pass rather than read out of Phase E.1's JSON** — it makes the before/after strictly
like-for-like. It lands on **51/84, with zero per-nodule prediction differences from Phase E.1's
published result**, and on **23/45 scan-level with 19 Benign over-calls**, both exactly as published.
**The harness is validated against the number it is being compared to.**

---

## 1. End-to-end accuracy, the same 84 nodules

### 1.1 The table

Score ≥ 0.5, Phase E/E.1's operating point. Detection is identical for every row (71/84); only the
classifier's answer on the matched candidate changes.

| variant | end-to-end | accuracy | **Wilson 95%** | called `Normal` | GT-centred ceiling | scan-level |
|---|---|---|---|---|---|---|
| **N=1 single slice (baseline)** | **51/84** | **0.607** | **[0.500, 0.705]** | 11 | **62/84 (0.738)** | 23/45 |
| N=3, **mean probability** | 49/84 | 0.583 | [0.477, 0.683] | 14 | 61/84 (0.726) | **24/45** |
| N=3, **majority vote** | **51/84** | **0.607** | **[0.500, 0.705]** | 11 | 60/84 (0.714) | **24/45** |
| N=3, max confidence | 31/84 | 0.369 | [0.274, 0.476] | 33 | 50/84 (0.595) | 22/45 |
| N=5, mean probability | 36/84 | 0.429 | [0.328, 0.535] | 27 | 45/84 (0.536) | 23/45 |
| N=5, majority vote | 37/84 | 0.440 | [0.339, 0.547] | 26 | 45/84 (0.536) | 22/45 |
| N=5, max confidence | 28/84 | 0.333 | [0.242, 0.440] | 38 | 31/84 (0.369) | 20/45 |

Tie-break for majority vote, stated because with 3 classes and N=3 a 1/1/1 tie is common: **the class
with the higher summed probability across the N slices wins.** The vote's reported probability vector
is the mean (a vote produces no calibrated vector, and the scan-level driver needs one to rank).

By label, at the two variants that are not obviously broken:

| | n | N=1 | N=3 mean | N=3 vote |
|---|---|---|---|---|
| Benign | 68 | 42 (0.618) | 40 | 42 |
| Malignant | 16 | 9 (0.563) | 9 | 9 |

**The best number any strategy reaches is 60.7%, [0.500, 0.705] — the number it started at.** The
~70% target is not approached by anything here; the closest upper confidence bound (0.705) is the
baseline's own.

### 1.2 The tie is a coincidence of cancelling flips, not stability

Paired, per matched nodule (n=71), against the recomputed N=1 baseline:

| variant | predictions changed | wrong→right | right→wrong | **net** |
|---|---|---|---|---|
| N=3 mean | 9 / 71 | 3 | 5 | **−2** |
| **N=3 vote** | **5 / 71** | **2** | **2** | **0** |
| N=3 maxconf | 25 / 71 | 1 | 21 | −20 |
| N=5 mean | 23 / 71 | 3 | 18 | **−15** |
| N=5 vote | 20 / 71 | 2 | 16 | −14 |
| N=5 maxconf | 31 / 71 | 2 | 25 | −23 |

**Even the best variant is churning 7% of its predictions to arrive at the same score.** Nothing here
resembles "the ensemble stabilises the classifier"; the flips simply happen to balance at N=3 with a
vote, and stop balancing as soon as the window widens or the rule changes.

---

## 2. Scan-level — Phase E.1's specific failure mode, checked directly

Phase E.1 flagged that scan-level accuracy *worsened* at scale: 23/45 (51.1%), with **19 Benign scans
over-called Malignant**, driven by unmatched candidates graded Malignant. Same aggregation rule
(untouched: any candidate graded Malignant flags the scan; the highest P(Malignant) is named driver),
same 45 scans, 29 Benign / 16 Malignant truth.

| variant | scan accuracy | Wilson 95% | **Benign over-called Malignant** | Malignant scans missed | "no nodule confirmed" |
|---|---|---|---|---|---|
| **N=1 (baseline)** | **23/45 (0.511)** | [0.370, 0.650] | **19** | 3 | 1 |
| N=3 mean | **24/45 (0.533)** | [0.391, 0.671] | **17** | 3 | 2 |
| N=3 vote | **24/45 (0.533)** | [0.391, 0.671] | 18 | 3 | 1 |
| N=5 mean | 23/45 (0.511) | [0.370, 0.650] | 17 | 4 | 3 |
| N=5 vote | 22/45 (0.489) | [0.350, 0.630] | 17 | 4 | 4 |
| N=5 maxconf | 20/45 (0.444) | [0.309, 0.588] | 15 | 5 | 8 |

**Read honestly: this is a one-scan improvement with a 28-point-wide interval, and it is bought with
the same mechanism that loses nodules.** Over-calls fall 19 → 17 because more candidates get graded
`Normal` when neighbouring slices are averaged in — and that same drift is what turns N=3 mean's
per-nodule 51 into 49, and what produces 3–8 scans reporting "no nodule confirmed" where the baseline
produced 1. **Ensembling does not fix the false-positive aggregation problem; it slightly mutes it by
making the classifier report less of everything.** The rule Phase E called "visibly wrong as
specified" is still visibly wrong.

---

## 3. The mechanism check — the part that matters

Phase E.1's sweep, re-run with ensembling. For each of the 84 nodules the crop is displaced by 2 / 5 /
8 mm along the **2 pure through-plane directions** (±S) and, as a control that must not degrade, the
**4 in-plane directions** (±R, ±A). At each displaced position the N-slice window is taken **around
the displaced slice**, which is exactly how a real detector error presents itself. 168 through-plane
crops and 336 in-plane crops per offset per variant.

### 3.1 Through-plane: no recovery at any displacement

| offset | **N=1 (single displaced slice)** | N=3 mean | N=3 vote | N=5 mean | N=5 vote | N=3 maxconf | N=5 maxconf |
|---|---|---|---|---|---|---|---|
| **2 mm** | **0.488** | 0.482 | 0.482 | 0.411 | 0.429 | 0.321 | 0.280 |
| **5 mm** | **0.226** | 0.226 | 0.220 | 0.208 | 0.196 | 0.179 | 0.184 |
| **8 mm** | **0.101** | 0.083 | 0.083 | 0.071 | 0.071 | 0.089 | 0.077 |

(0 mm reference accuracy is 0.738, unchanged from Phase E.1.)

**The brief asked specifically whether this helps at 2 mm but not at 5–8 mm — the pattern that would
mean it only patches the smallest errors. It does not even do that.** At 2 mm, the displacement a real
detector actually produces (Phase D measured a 2.75 mm mean 3D error), the ensemble is **flat to
slightly negative**. Counting individual rescues: of the 86 through-plane crops at 2 mm where the
single displaced slice was wrong, N=3 mean recovers **5** and leaves 81 wrong; N=5 mean recovers 7 and
leaves 79. Those handfuls are more than offset by newly-broken crops, which is why the aggregate
accuracy does not move.

### 3.2 The conditional test — the hypothesis, isolated

The hypothesis has a precondition: **the true centre slice must be inside the window.** It often is
not, because 2 mm on a thin series is more than one slice:

| through-plane offset | slices actually moved |
|---|---|
| 2 mm | 1 slice 49%, 2 slices 25%, 3 slices 23%, 4 slices 4% |
| 5 mm | ≤2 slices 31%, up to 10 slices |
| 8 mm | ≥3 slices always, up to 16 |

so the true slice sits inside the window **48.8% of the time at 2 mm/N=3, 73.8% at 2 mm/N=5, 31.0% at
5 mm/N=5, and never at 5 mm/N=3 or at 8 mm.** Restricting to the crops where it *is* present:

| case | n | single displaced slice | **ensemble** |
|---|---|---|---|
| 2 mm, N=3, mean **(true slice in window)** | 82 | 0.549 | **0.549** |
| 2 mm, N=3, vote (true slice in window) | 82 | 0.549 | **0.549** |
| 2 mm, N=5, mean (true slice in window) | 124 | 0.492 | **0.395** |
| 2 mm, N=5, vote (true slice in window) | 124 | 0.492 | **0.403** |
| 5 mm, N=5, mean (true slice in window) | 52 | 0.269 | 0.308 |
| 2 mm, N=3, mean (true slice **outside**) | 86 | — | 0.419 |

**This is the direct answer to the phase's question. On exactly the crops where the correct slice is
among the N being classified, N=3 ensembling produces the identical accuracy to not ensembling at all,
and N=5 produces a 10-point loss.** The true slice's vote is there. It is outvoted, or averaged
under. The only cell that improves is 5 mm/N=5 (0.269 → 0.308 on n=52, a 2-crop difference, and the
same variant scores 36/84 end to end) — noise, not a finding.

### 3.3 In-plane control — ensembling damages the axis that worked

| offset | **N=1** | N=3 mean | N=3 vote | N=5 mean | N=5 vote |
|---|---|---|---|---|---|
| 2 mm | **0.744** | 0.720 | 0.735 | **0.509** | 0.539 |
| 5 mm | **0.735** | 0.717 | 0.732 | **0.509** | 0.533 |
| 8 mm | **0.723** | 0.708 | 0.723 | **0.506** | 0.527 |

Phase C.1's in-plane tolerance — the one real, reproduced success in this arc — is **flat under N=3
and cut by ~22 points under N=5**. Any deployment of this would be paying a certain loss on the good
axis for a measured zero on the bad one.

### 3.4 Slice thickness, checked because it is the obvious confound

An N=5 window is 2 mm of tissue on a 0.5 mm series and 12 mm on a 3.0 mm series, so the thin- and
thick-slice halves are not the same experiment.

| | thin (≤1.25 mm, 43 nodules) | thick (>1.25 mm, 41 nodules) |
|---|---|---|
| N=1 end-to-end | 20/43 (0.465) | 31/41 (0.756) |
| N=3 mean | **21/43 (0.488)** | 28/41 (0.683) |
| N=5 mean | **21/43 (0.488)** | 15/41 (0.366) |
| 2 mm through-plane sweep, N=1 | 0.430 | 0.549 |
| 2 mm through-plane sweep, N=3 mean | 0.419 | 0.549 |
| 2 mm through-plane sweep, N=5 vote | 0.477 | 0.378 |

**The thin-slice half is where ensembling should have worked, and the effect there is +1 nodule out of
43.** It is not significant and it is dwarfed by the thick-slice damage. (The 0.465-vs-0.756 baseline
gap between the two halves is a real and unexplained property of this test set — it is not something
this phase set out to measure, and it is recorded here rather than interpreted.)

---

## 4. New failure patterns introduced by ensembling

### 4.1 Neighbour slices are systematically biased, not noisy — the root cause of the null result

Measured on the 84 perfectly-centred GT windows, one row per slice offset:

| slice | n | **accuracy** | agrees with the true slice | **predicted `Normal`** |
|---|---|---|---|---|
| the true slice (dk=0) | 84 | **0.738** | 100% | 10.7% |
| dk = ±1 | 168 | **0.601** | 72.6% | **24.4%** |
| dk = ±2 | 168 | **0.375** | 46.4% | **52.4%** |

**Ensembling assumes the extra voters are noisy around the truth. They are not — they are worse
classifiers with a growing `Normal` bias.** This is the same asymmetry Phase E.1 found in its flip
destinations (429 of 618 flips at 8 mm were `Benign → Normal`), now measured as a property of the
voters themselves. Averaging five predictions whose accuracies are 0.38 / 0.60 / **0.74** / 0.60 /
0.38 cannot outperform the 0.74 term, and the `called_Normal` column in §1.1 shows exactly where the
loss goes: **11 → 27 nodules called `Normal` at N=5 mean.**

### 4.2 Wash-out: it destroys confidently-correct answers

The brief asked whether averaging washes out cases that were confidently correct on the true centre
slice. **Yes, and it is the dominant cost at N=5.** On GT-centred windows:

| variant | correct answers **broken** | mean single-slice confidence of those | of which ≥0.90 confident | fixed | net |
|---|---|---|---|---|---|
| N=3 mean | 3 | 0.603 | 0 | 2 | −1 |
| N=3 vote | 3 | 0.603 | 0 | 1 | −2 |
| N=3 maxconf | 13 | 0.801 | 6 | 1 | −12 |
| **N=5 mean** | **19** | **0.838** | **10** | 2 | **−17** |
| N=5 vote | 19 | 0.838 | 10 | 2 | −17 |
| N=5 maxconf | 33 | 0.871 | 20 | 2 | −31 |

**N=3 is comparatively gentle — it breaks only low-confidence cases (mean 0.603, none above 0.90).
N=5 breaks high-confidence ones (mean 0.838, ten at ≥0.90).** That is the difference between the two
window sizes in one line, and it is why the classifier's own ceiling falls from 0.738 to 0.536 under
N=5.

### 4.3 Majority vote dominated by wrong neighbours — also yes

The brief's second predicted failure. At N=5 the true slice is 1 vote of 5 and, per §4.1, the other
four are right 60% / 38% of the time with a `Normal` lean, so the vote goes to the neighbours whenever
two of them agree wrongly: **N=5 vote changes 20 of 71 matched predictions and 16 of those go
right→wrong.** At N=3 the true slice is 1 of 3 and the tie-break (summed probability) rescues it often
enough to hold the score — which is precisely why N=3 vote ties instead of losing. **The tie-break
rule, not the ensembling, is what holds the best variant at baseline.**

### 4.4 Max-confidence is the worst rule and the reason is instructive

`maxconf` loses 20–23 nodules net. Its failure is systematic: the most *confident* slice is very often
an off-centre one that confidently sees nothing and returns `Normal` at >0.95 — the `called_Normal`
count goes 11 → 33 at N=3. **Confidence on this classifier is not a proxy for being on the right
slice.** Worth recording, because "pick the most confident view" is the intuitive rule and it is the
worst of the three by a wide margin.

---

## 5. Resources and latency — the one clean positive

Measured with the **real** two-stage pipeline (Phase E's `TwoStagePipeline`, imported unmodified),
both models resident, on 5 volumes, timing single-slice and N=3/N=5 stage 2 back to back in the same
process on the same candidates.

| | Phase E (10 scans) | Phase E.1 (35 scans) | **Phase F, N=5 (5 scans)** |
|---|---|---|---|
| **peak VRAM allocated** | 0.751 GB of 4.00 | 0.751 GB | **0.751 GB — identical** |
| VRAM during stage 2 alone | — | — | 0.237–0.335 GB |
| **stage 2 time / scan** | 0.2 s | 0.0 s | **0.129 s** (N=3: 0.117 s) |
| crops classified / scan | 2–4 | — | **10–20** |
| **total latency / scan** | 33.4 s | 35.7 s | **33.1 s** (19.2–43.3) |
| peak host working set | 8.23 GB / 15.26 | 10.23 GB / 15.26 | **7.00 GB / 15.26** (5 scans) |

**The 5× forward passes cost nothing measurable.** Crop extraction for a whole N=5 window is 2–4 ms;
the forward pass is 0.05–0.19 s; aggregation is under 1 ms. Total latency is inside the ~35 s Phase D/E
budget with room to spare — the N=5 figure is *lower* than the baseline's 0.18 s only because the
baseline's number includes first-call warm-up, which is the honest reading of two numbers this small.

**Host RAM, which Phase E identified as the tight resource, does not move.** Per-scan RSS with the
N=5 window in flight against the same scan's single-slice RSS: 2.17→1.98, 4.92→4.91, 3.47→3.49,
4.79→4.80, 5.05→4.88 GB — **differences of ≤0.01 GB in either direction, i.e. noise.** The reason is
structural: a 28 mm crop is a 40×40 array, and even 20 of them batched to 224×224×3 is ~12 MB against
a float32 chest volume of several GB. The 7.00 GB peak here is **not** comparable to Phase E's 8.23 or
Phase E.1's 10.23 — those are peaks over 10 and 35 scans including much larger volumes, and this probe
ran 5. **The comparable quantity is the N=1-vs-N=5 delta within one process, and it is zero.**
Ensembling does not move this any closer to the ceiling; `MAX_CONCURRENT = 1` remains the binding
protection for the reasons Phase E gave, unchanged.

---

## 6. Live-system integrity

Re-hashed **after** all inference. sha256, first 16 hex characters — the same formula Phase C.1/D/E/E.1
used, which is confirmed by all five reproducing exactly.

```
checkpoints/ct_best_model_phase35_v2.pt      sha=93b3a979884edbc4  mtime=1787555730.4296012   MATCHES Phase E §7 / E.1 §6
backend/main.py                              sha=780120553178c0a5  mtime=1788684518.831581    MATCHES Phase E §7 / E.1 §6
checkpoints/ct_lidc_nodulecrop_v1.pt         sha=bdcbd6b2e5312435                             MATCHES Phase E §7 / E.1 §6
checkpoints/ct_lidc_nodulecrop_v2_jitter.pt  sha=5d60055e02974e91                             MATCHES Phase E §7 / E.1 §6
eval_run/lidc_full/labels_manifest.csv       sha=95d734f64123a46f  3461 rows                  MATCHES Phase E §7 / E.1 §6

eval_run/phaseE/pipeline.py                  sha=936427fc20d389a0  mtime=1788946798.4063764   recorded for the first time
eval_run/phaseE/run_eval.py                  sha=cc487ad5344fc643  mtime=1788946651.7800155   recorded for the first time
```

**All five match and both recorded mtimes are unchanged**, so the live checkpoint and `backend/main.py`
were neither rewritten nor rewritten-to-identical-content. `phaseE/pipeline.py`'s mtime is
**2026-09-09 15:09**, i.e. Phase E's own build time — it was imported, never edited; its digest is now
on record so later phases can check it the same way. Both classifier checkpoints were opened
**read-only**; no checkpoint was written. Every write this phase is inside `eval_run/phaseF/`
(**1.1 MB**). No new dependency. Nothing deployed, no `uvicorn` started, **no process left running**
(verified: zero `python.exe`).

**One transparency note.** At the start of this phase `eval_run/phaseE1/integrity.py` was executed to
establish the baseline digests, and it writes to `phaseE1/_work/integrity.json` — so that one Phase
E.1 artifact has a new mtime. Its content is a deterministic function of the five files it hashes,
all of which are unchanged, so the content is identical; no Phase E.1 measurement is affected. Phase
F has its own `integrity.py` writing to `phaseF/_work/`, which is what produced the block above.

---

## 7. Verdict: (c) — it doesn't help, and the reason generalises

**Do not pursue inference-time multi-slice ensembling.** Not "it helped a little", not "it needs
tuning" — the hypothesis was specific, it was tested on its own terms, and it failed on its own terms.

### Why this is a real answer and not an underpowered one

A null result is only worth acting on if it could have come out otherwise. This one could have:

1. **The precondition was satisfied often enough to detect an effect.** The true slice was inside the
   window on 82 (N=3) and 124 (N=5) of 168 through-plane crops at 2 mm. **On exactly those crops the
   N=3 ensemble scored 0.549 against the single slice's 0.549 — not a wash, an identity.**
2. **The failure has a measured cause, not a shrug.** Neighbour slices are 0.601 (±1) and 0.375 (±2)
   accurate with `Normal` rates of 24% and 52%. The ensemble is not combining independent views of the
   same lesion; it is diluting one adequate view with progressively blanker ones. Any post-hoc
   combination rule inherits that, which is why all three rules fail and the two that fail worst are
   the two that give neighbours more weight.
3. **It was checked in both directions.** The in-plane control degrades (0.744 → 0.509 at N=5), the
   perfect-crop ceiling degrades (0.738 → 0.536), the wash-out is quantified (19 broken, 10 of them at
   ≥0.90 confidence), and the slice-thickness confound is stratified out. The one favourable-looking
   cell (scan over-calls 19 → 17) is traced to the same `Normal` drift that costs nodules.
4. **The comparison is exact.** Same 84 nodules, same 45 patients, same detection, same matching, same
   Wilson interval on the same n — and the recomputed baseline reproduces Phase E.1's published 51/84
   and 23/45 with **zero** per-nodule differences.

### On a hypothetical Phase G

The brief's option (b) — "helps but falls short, so a genuinely 3D-aware classifier is next" — is
**not** what the evidence says, and it would be wrong to report it that way. But this phase does
produce one piece of evidence relevant to that question, and it points both ways:

* **For:** the ±1 slice is not blind. It is 0.601 accurate and agrees with the true slice 72.6% of the
  time, so there is real signal in the neighbourhood that a model taking a small stack as its **actual
  input** could learn to weight — something a post-hoc average, which must treat all N slices as
  equally authoritative, structurally cannot do.
* **Against:** everything Phase E.1 §7 already said, unchanged and unchallenged by this phase. Such a
  model must be **retrained**, re-validated on all three axes, and would plausibly cost centred
  accuracy off a **74% ceiling that is already the binding constraint** — and the aggregation redesign
  and guardrail chain still do not exist. **~10 days and an exam.**

**This phase does not make that call and was explicitly told not to.** What it does establish is that
the cheap version of the idea is now closed: **the through-plane defect cannot be patched at inference
time, and anything that addresses it has to change the model.**

### What this phase does NOT show

* **It does not re-open or re-litigate Phase E.1's conclusions** — it reproduces them exactly and adds
  one negative result.
* **Still LIDC, still the detector's own lineage.** Nothing here is a generalization claim.
* **N ∈ {3, 5} with three fixed rules.** A weighted average with a learned or distance-based weighting
  was not tested — though §4.1 predicts what it would find, since the optimal weighting of a
  0.38/0.60/**0.74**/0.60/0.38 stack is "put all the weight on the centre", which is the baseline.
* **The detector was not re-run**; its stored candidates were reused, with the verification in §0.
* **The latency probe is 5 scans**, sufficient for a cost delta that is ~0 and not offered as a new
  latency distribution.
* **Phase A's deployment no-go and the standing advice not to cite the live model's external-Malignant
  12/12 as generalization evidence are untouched.**

**Phase F closes as: the best ensemble ties the baseline at 51/84 (60.7%, [0.500, 0.705]); the
mechanism check shows the true slice is present and outvoted; ensembling breaks 19 confidently-correct
answers at N=5 and damages the in-plane axis; it is free to run and it buys nothing. Answer (c).**

---

## 8. Artifacts

All under `eval_run/phaseF/` (1.1 MB). Nothing is referenced by `backend/`; nothing under
`eval_run/phaseE/` or `eval_run/phaseE1/` was modified (one exception noted in §6).

| path | what |
|---|---|
| **`ensemble.py`** | `shift_to_slice` / `slice_window` / `window_crops` (the only new geometry, built on Phase E's `crop_at_ras`), the three aggregation rules with the vote tie-break, the Wilson helper |
| **`run_ensemble.py`** | §0 — one pass over the 45 volumes producing candidate windows, GT-centred windows and the ensembled stability sweep; 9,123 crops in 4.4 min; the reuse verification |
| **`analyze_ensemble.py`** | §1–§4 — end-to-end and scan-level scoring, the GT-centred ceiling and wash-out, the sweep decomposition, the true-slice-in-window conditional, slice-spacing strata |
| **`latency_probe.py`** | §5 — the real two-stage pipeline on 5 scans, single-slice vs N=3 vs N=5 stage 2 timed back to back, VRAM/RSS probed |
| `integrity.py` | §6 |
| `_work/ensemble_raw.json` | every per-slice probability vector this phase computed (candidates, GT windows, sweep) |
| `_work/ensemble_analysis.json` | every table in §1–§4 |
| `_work/latency.json` | §5, per-scan |
| `_work/{run_ensemble,analyze,latency}.log` | run transcripts |
| `_work/integrity.json` | §6 |

Reused read-only and unmodified: `eval_run/phaseE/{pipeline,run_eval}.py`,
`eval_run/phaseE/_work/{e2e_results,scan_reports}.json`,
`eval_run/phaseE1/_work/{cases_e1,e2e_results_e1,scan_reports_e1}.json`,
`eval_run/phaseD/{volume_io}.py`, `eval_run/phaseD/_work/{cases.json,volumes/}`,
`eval_run/phaseE1/_work/volumes/`, `ml/{model,dataset}.py`,
`checkpoints/ct_lidc_nodulecrop_v2_jitter.pt`.

**Nothing was integrated. No model was trained. Phase G was not started — that decision is the
user's.**
