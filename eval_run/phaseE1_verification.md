# Phase E.1: the two measurements Phase E could not carry — bigger n, and per-crop stability

**Scope: measurement only. Nothing was built.** Phase E's pipeline (`eval_run/phaseE/pipeline.py`,
`run_eval.py`) was run **unmodified** — imported, its two path constants repointed, its `main()`
called. No pipeline feature, no FP-aggregation redesign, no guardrail work. Everything written this
phase is confined to `eval_run/phaseE1/` (8 Python files + JSON results + 4.9 GB of cached DICOM).
**`backend/main.py`, `/predict/ct` and `checkpoints/ct_best_model_phase35_v2.pt` are untouched —
re-hashed in §6, all five digests match Phase E §7 exactly.** No checkpoint was written, no new
dependency installed, nothing deployed.

---

## Verdict up front

**Both questions are answered, and the answers point the same way.**

1. **The end-to-end number held, and the interval is now half as wide. 51/84 nodules (60.7%)**
   across **45 patients**, Wilson 95% **[0.500, 0.705]** — against Phase E's 13/22 (59.1%),
   [0.387, 0.767]. **The 59.1% was not bad luck. It reproduces at 3.8× the sample.** §1.
2. **The 20% churn is REAL and GENERAL — it was not one bad patient.** Pooled, **15 of 71 matched
   predictions (21.1%, Wilson [0.132, 0.320]) change class** between the annotation-centred and
   detector-centred crop, at a mean displacement of 2.88 mm. Phase E's 4/20 came from 2 patients;
   the pooled 15 come from **13 different patients**. LIDC-0321 is the worst single scan but sits
   at the **91st percentile, z=1.68** of a 45-patient distribution — high, not anomalous. §2.3.
3. **The mechanism is now identified, and it is the finding of this phase. `v2_jitter`'s
   translation tolerance is IN-PLANE ONLY.** Displace a crop within the slice and accuracy is flat
   (0.738 → 0.723 at 8 mm, 3.9–8.3% flip). Displace it **through-plane** — to an adjacent slice, which
   is what a 3D detector actually does — and it collapses (0.738 → **0.226 at 5 mm**, **40.5% flip at
   2 mm**). §2.2.
4. **Phase C.1's tolerance curve never tested that axis, and neither did the augmentation that
   produced it.** Both shifted the 28 mm box on the two *image* axes only (its displacements are
   `SHIFT_MM × √2`). So the "~13 mm tolerance" Phase C.1 reported, which is what made Phase D look
   affordable, is a **2-of-3-axis measurement**. On the third axis `v2_jitter` is **barely better than
   the un-augmented `v1`** (through-plane accuracy at 5 mm: v1 0.137, v2 0.226, against in-plane
   v1 0.292, v2 **0.735**). §2.2.
5. **The classifier remains the ceiling, and the ceiling did not move: 62/84 (73.8%)** on perfect
   annotation-centred crops, Wilson [0.635, 0.820] — Phase E measured 0.727 at n=22. **A perfect
   detector still yields ~74%.** §1.2.
6. **Malignant is no longer n=4.** Twelve Malignant-carrying series were added specifically to fix
   Phase E's stated non-claim: **16 Malignant nodules, 13 detected, 9 graded correctly (56.3%,
   Wilson [0.332, 0.769])**. Still wide, but it is now a measurement rather than a refusal. §1.3.
7. **Scan-level accuracy got worse with more scans, not better: 23/45 (51.1%)**, with **19 Benign
   scans over-called Malignant**, driven by **2.83 unmatched candidates/scan of which 27 are graded
   Malignant**. Phase E called the max-driven aggregation rule "visibly wrong as specified" at
   5/10; at 45 scans it is worse. §4.

**Recommendation (§7): close the investigation here.** Not because the pipeline is broken — it works,
and the measurements are clean — but because the numbers now say the thing a Phase F would be built
on does not hold. Details in §7, including the one honest argument on the other side.

---

## 0. What was measured, and what was reused

| | Phase E | **Phase E.1** |
|---|---|---|
| volumes | 10 | **45** (10 cached + **35 newly downloaded**) |
| nodules | 22 | **84** (68 Benign / **16 Malignant**) |
| patients | 10 | **45** |
| stability sweep | — | **1,176 re-extracted crops/checkpoint**, 4 offsets × 14 directions × 84 nodules, run for **both** `v2_jitter` and `v1` |

**Reused rather than rewritten**, per the standing rule:

* `phaseE/run_eval.py` — **imported and called**, not copied. `run_eval_e1.py` sets
  `run_eval.PHASE_D_WORK` and `run_eval.OUT` and calls `main()`. Matching logic (8 mm nearest-candidate,
  per-nodule, independent), the 15 mm unmatched radius, both controls, the error decomposition and the
  resource probes are **the same code that produced 13/22**. That is what makes the two halves poolable.
* `phaseE/pipeline.py` — `crop_at_ras`, `crop_to_pil`, `load_classifier`, `CLASS_NAMES`, `DEFAULT_CKPT`,
  imported unchanged. The stability sweep does **not** load the detector (it measures the classifier's
  own positional stability), but the crop path is byte-for-byte Phase E's.
* `phaseD/volume_io.py` — `read_series`, `annotation_to_ras`, verbatim.
* `phaseD/download_volumes_perslice.py` — `sops` / `fetch_slice` **imported**. Phase D's warning was
  honoured: `download_volumes.py`'s whole-series-zip route was never invoked.
* `phaseD/select_cases.py` — its selection pattern (deterministic, no RNG, prefer more nodules per
  series) copied into `select_cases_more.py` with the already-downloaded series excluded.

**The crop path was re-verified on every new volume, not assumed:** the 0 mm crop is compared against
Phase B's stored PNG for all 84 nodules. **84/84 bit-exact, max absolute gray difference 0** — including
all 62 nodules from series Phase B/C/D/E never opened as volumes. The extraction is provably still the
classifier's training distribution.

### On the download cost, plainly

The brief asked for this to be reported honestly rather than ground through silently. **35 volumes,
9,791 slices, 4.9 GB, ~2 h 5 min of wall-clock download** at a sustained 0.6–1.4 MB/s. That is *faster*
per slice than the one rate Phase D recorded (468 s for 179 slices); there is simply 35× more of it.
One deliberate experiment: a second parallel downloader **reduced** combined throughput (45 → 25
slices/min), so TCIA is throttling per-client, and it was killed after one measurement. Inference was
as Phase D/E predicted — **35.7 s mean/scan (18.0–71.8)**, so the 45-scan evaluation cost ~25 min, not
the bottleneck. Nothing here ran dramatically slower than predicted.

---

## 1. The bigger end-to-end test set

### 1.1 Headline, pooled

Same pipeline, same harness, same match radii. `score ≥ 0.5` is the default, as in Phase E.

| | Phase E (n=22) | **Phase E.1 pooled (n=84)** |
|---|---|---|
| patients | 10 | **45** |
| detected (≤ 8 mm) | 20/22 (0.909) | **71/84 (0.845)**, Wilson [0.753, 0.907] |
| **END-TO-END CORRECT** | **13/22 (0.591)** | **51/84 (0.607)** |
| **95% CI (Wilson)** | [0.387, 0.767] — width **0.38** | **[0.500, 0.705]** — width **0.20** |
| 95% CI (Clopper–Pearson) | [0.364, 0.793] | [0.495, 0.712] |
| 95% CI (patient-clustered bootstrap) | [0.333, 0.833] | [0.506, 0.707] |
| — Benign↔Malignant crossed | 3 | 9 |
| — detected but called Normal | 4 | 11 |
| — missed by the detector | 2 | 13 |
| **control: same crops, GT-centred** | 16/22 (0.727) | **62/84 (0.738)**, [0.635, 0.820] |
| at `score ≥ 0.3` | 13/22 (0.591) | **52/84 (0.619)**, [0.512, 0.716] |

**The Phase E half reproduces exactly** when re-pooled through this phase's code — 13/22, Wilson
[0.3873, 0.7674], matching Phase E §8's "roughly 0.39–0.78". The CI methodology is Phase E's; only n
changed. The patient-clustered bootstrap is added because nodules from one scan are not independent —
it is reported alongside Wilson, not instead of it, and at n=84 the three intervals agree closely
(they did **not** at n=22, where the clustered interval was markedly wider — itself a demonstration of
how little 22 nodules constrained anything).

**Reading it plainly: 59.1% was not an unlucky draw.** The point estimate moved by 1.6 points and the
interval shrank by half. **The pipeline's end-to-end accuracy is ~61%, and that is now a measurement
rather than a guess.** The naive product of the two published stage numbers (0.909 × 0.792 = 0.72)
remains materially wrong, exactly as Phase E found.

### 1.2 Error decomposition, pooled

| source | nodules lost (of 84) | Phase E's n=22 equivalent |
|---|---|---|
| **classifier error on a perfect, annotation-centred crop** | **22** | 6 |
| detector miss that cost a correct answer | 7 | 1 |
| net cost of the detector's crop centring | 4 (9 wrong-ward, 5 right-ward) | 2 |
| **total wrong** | **33** → 51/84 correct | 9 → 13/22 |

**The classifier is still the bottleneck by ~3×, and its ceiling is stable across 4× the data:
0.727 at n=22, 0.738 at n=84.** That figure is consistent with `v2_jitter`'s own 0.792 nodule accuracy
on Phase C.1's 207-crop test set (these 84 are a subset of that set; 0.738 sits inside binomial noise
of 0.792 at n=84). **Nothing anomalous. A pipeline cannot be more accurate than its second stage, and
its second stage is ~74% on crops it is already perfectly centred on.**

Detection fell from 0.909 (n=22) to 0.845 (n=84). The intervals overlap heavily, and the earlier
figure came from 10 series chosen for carrying the most annotated nodules; **0.845 over 45 patients is
the better estimate of the detector's recall on this population**, and it still comfortably clears
Phase D's headline.

### 1.3 Malignant — Phase E's explicit non-claim, now closed

Phase E recorded "18 B / 4 M, so n=4 is not a Malignant accuracy measurement." Step 1's
nodules-per-download ranking made this *worse* rather than better — every multi-nodule series in the
LIDC test split happens to be all-Benign, so the first 23 new volumes added **50 Benign and zero
Malignant**. That is a selection artifact of optimising the download, and it was paid to fix:
`add_malignant_cases.py` selected 12 Malignant-carrying test series (every one carries exactly one
malignant nodule, so these are one-nodule-per-volume downloads), ordered by series UID — **not** by
slice count, which would have been faster to fetch and would have biased the sample by acquisition
protocol.

| | n | detected | end-to-end correct | accuracy | Wilson 95% |
|---|---|---|---|---|---|
| **Malignant** | **16** | 13 | **9** | **0.563** | [0.332, 0.769] |
| **Benign** | **68** | 58 | **42** | **0.618** | [0.499, 0.724] |

**Malignant and Benign end-to-end accuracy are statistically indistinguishable here, both ~0.6.** n=16
is still small and the interval is still wide — this does not license a clinical claim — but the
honest statement is no longer "we cannot say", it is **"56%, ±22 points"**.

---

## 2. The per-crop stability sweep

### 2.1 Method

For each of the 84 nodules: take its annotation position in RAS mm, cut the 28 mm crop (the 0 mm
reference), then re-cut it at **2, 3, 5 and 8 mm Euclidean** displacement in **14 fixed directions** —
the 6 signed RAS axes plus the 8 cube-corner diagonals, unit-normalised. The direction set is identical
for every nodule. Every crop is re-extracted from the original DICOM (real pixels, no padding, no
synthetic translation), classified by `v2_jitter`, and scored on **whether the predicted class differs
from that nodule's own 0 mm prediction** — per-crop flipping, not aggregate accuracy.

**1,176 crops per checkpoint.** The whole sweep was run twice: on `v2_jitter` and, as a control, on
Phase C's un-augmented `v1`.

**Why the direction split matters.** Phase C.1's `tolerance_sweep.py` shifted the box by `SHIFT_MM` on
**both image axes and nothing else** — its Euclidean displacements are `SHIFT_MM × √2` (4.2 / 8.5 /
12.7 / 15.6 / 18.4 mm). The augmentation it was validating is the same shape: `prep_jitter_source.py`
cuts an oversized window from **the same DICOM slice**, so the jitter is in-plane by construction (its
recorded mean realised displacement, 7.68 mm with max 13.43, is what uniform ±10 mm on *two* axes
gives). **A detector mis-centres in 3D and lands on a different slice.** Splitting the 14 directions
into 4 in-plane, 2 pure through-plane and 8 mixed is what makes this sweep comparable to Phase C.1 and
isolates the axis nobody had measured.

### 2.2 The result — the tolerance is in-plane only

`v2_jitter`, 84 nodules, 0 mm accuracy **0.738**:

| directions | 2 mm | 3 mm | 5 mm | 8 mm |
|---|---|---|---|---|
| **in-plane only** (4 dirs) | 3.9% flip / acc **0.744** | 4.8% / **0.738** | 6.6% / **0.735** | 8.3% / **0.723** |
| **through-plane only** (2 dirs) | **40.5%** / **0.488** | **51.2%** / **0.399** | **69.6%** / **0.226** | **82.1%** / **0.101** |
| mixed diagonal (8 dirs) | 25.7% / 0.606 | 35.0% / 0.527 | 52.1% / 0.372 | 67.3% / 0.231 |
| **all 14 directions** | **21.6%** / 0.628 | **28.7%** / 0.569 | **41.6%** / 0.455 | **52.5%** / 0.353 |

The same sweep on `v1` (Phase C's un-augmented checkpoint), 0 mm accuracy 0.798:

| directions | 2 mm | 3 mm | 5 mm | 8 mm |
|---|---|---|---|---|
| in-plane only | 15.8% / 0.684 | 28.0% / 0.580 | 60.1% / **0.292** | 78.6% / 0.146 |
| through-plane only | 39.9% / 0.524 | 55.4% / 0.387 | 81.0% / **0.137** | 91.7% / 0.042 |

**Three things follow, and the third is the point of this phase.**

* **The sweep reproduces Phase C.1 where the two overlap, so the method is validated against a known
  result.** `v1` in-plane reads 0.292 at 5 mm; Phase C.1's curve reads 0.338 at 4.2 mm and 0.121 at
  8.5 mm. `v2` in-plane is flat (0.744 → 0.723 out to 8 mm); Phase C.1 reported 0.792 → 0.797 → 0.783
  out to 8.5 mm. Both match.
* **The jitter did exactly what Phase C.1 said, on the axes Phase C.1 measured.** In-plane, `v1` loses
  50 points by 5 mm and `v2` loses **0.3 of a point**. That result stands and is not being retracted.
* **On the third axis the augmentation bought almost nothing.** Through-plane at 5 mm: `v1` 0.137,
  `v2` **0.226**. Against in-plane at 5 mm: `v1` 0.292, `v2` **0.735**. **`v2_jitter` is a
  translation-tolerant model in two dimensions and a position-locked model in the third** — and the
  third is the one a 3D detector moves along. Phase C.1's conclusion that "the required detector
  tolerance moved from under ~4 mm to ~13 mm" is true in-plane and **false through-plane, where it is
  still ~2 mm**.

**Is this just "the nodule left the slice"?** Partly at large offsets, and demonstrably **not** at small
ones. A 2 mm through-plane offset always moves at least one slice (minimum observed |Δslice| = 1, and
these are 0.5–2.5 mm series), and for nodules **≥ 9 mm — where 2 mm of z is well inside the lesion and
the adjacent slice still shows it — the flip rate is still 21%, and 31% at 3 mm**. Small nodules do
behave as physics predicts (< 6 mm: 61% flip at 2 mm, 94% at 8 mm), and at 5–8 mm the lesion genuinely
is leaving the frame. **The instability is real at the displacements a detector actually produces; it
is compounded, not caused, by lesions leaving thin slices.** The dominant flip is `Benign → Normal`
(429 of 618 flips at 8 mm), i.e. the model reporting an empty patch.

### 2.3 Does Phase E's 20% churn hold up, and was it one patient? — the direct answer

**Yes it holds up, and no it was not one patient.**

| | Phase E (n=22) | **pooled (n=84)** |
|---|---|---|
| matched predictions | 20 | **71** |
| **changed class vs the GT-centred crop** | **4 (20.0%)** | **15 (21.1%)**, Wilson [0.132, 0.320] |
| correct → wrong / wrong → correct | 3 / 1 | **9 / 5** |
| mean / max displacement | 2.75 / 5.39 mm | **2.88 / 6.60 mm** |
| **patients contributing a flip** | **2** (3 of 4 in LIDC-0321) | **13 of 45** |

**21.1% against Phase E's 20.0% — the rate replicates almost exactly, at 3.5× the matched sample, and
it is now spread across 13 patients.** Phase E's inability to distinguish "real 20% churn" from "one
odd scan" is resolved in favour of *real*.

**On LIDC-0321 specifically:** it is still the worst single scan (3 of 3 matched predictions flipped),
and in the stability sweep it is high but not anomalous — flip rate 0.500 at 2 mm against a
45-patient population mean of 0.188 (**z = 1.68, 91st percentile, 4 of 45 patients at or above it**),
and by 5 mm it is merely **60th percentile**. **It is a bad scan drawn from a distribution that
contains plenty of others like it, not a unique artifact.** Its Phase E flips are explained by the same
mechanism as everyone else's: all three sat at |Δslice| = 1.

---

## 3. What this means for the tolerance story

Phase C.1 §6 closed by saying the detector-tolerance objection had been "measured, attacked, and
largely removed", leaving only the schedule argument. **That conclusion needs one correction, and it is
material.** The tolerance was removed on the two axes that were tested and augmented. On the third it
was never tested, and it is still there. Phase D then measured detector displacements averaging
2.75 mm *in 3D* — which read as comfortably inside a 13 mm budget, and are in fact **outside a ~2 mm
through-plane budget**. That is the whole of Phase E's compounding surprise, and it is now explained
rather than merely observed.

**This is a correction to Phase C.1's scope, not a refutation of its result.** The jitter experiment
was correct, its before/after was real, and a reader of Phase C.1 would have had no way to know the
curve was 2-of-3-axis — the report does not claim otherwise, it simply never says which axes. What
changes is the cost estimate that was built on it.

---

## 4. False-positive handling at 45 scans — worse, not better

Reported because the pooled run produces it, not because this phase attempted a fix (it did not — the
aggregation rule is untouched).

| | Phase E (10 scans) | **new half (35 scans)** |
|---|---|---|
| unmatched candidates / scan @0.5 | 2.00 | **2.83** |
| of which graded **Malignant** | 3 | **27** |
| of which absorbed as Normal | 5 (25%) | 28 (28%) |
| **scan-level accuracy** | 5/10 | **23/45 (51.1%)** overall |
| Benign scans over-called Malignant | 3 (2 driven by an unmatched candidate) | **19 overall** |

**Scan-level accuracy is 51.1% over 45 scans and 19 Benign scans are called Malignant.** Phase E judged
the "highest-malignancy candidate drives the call" rule "visibly wrong as specified" on 10 scans. At 45
it is not a small-sample impression; it is the dominant failure mode of the scan-level output. Nothing
about that has been redesigned here, and §7 treats it as evidence rather than as a to-do.

---

## 5. Resources and latency, 45 scans

Confirms Phase E, with one number moving in the wrong direction.

| | Phase E (10 scans) | **Phase E.1 (35 new scans)** |
|---|---|---|
| peak VRAM, both models resident | 0.751 GB of 4.00 | **0.751 GB** — identical, unvarying |
| **peak host working set** | 8.23 GB of 15.26 | **10.23 GB of 15.26 (67%)** |
| total latency, mean (min–max) | 33.4 s (18.5–45.4) | **35.7 s (18.0–71.8)** |
| — read + resample / detect / classify | 8.1 / 25.1 / 0.2 s | **9.9 / 25.7 / 0.0 s** |

**VRAM is confirmed a non-issue and stage 2 is confirmed free.** Host RAM is the constraint Phase E
identified, and on a wider volume-size distribution the peak rose from 54% to **67% of system RAM** —
`MAX_CONCURRENT = 1` remains the only thing standing between this and an OOM next to a browser and the
dev server.

---

## 6. Live-system integrity

Re-hashed **after** all inference — three times this phase, including once immediately after a session
interruption specifically so that no stale record was trusted. sha256, first 16 hex characters, the
same formula Phase C.1/D/E used.

```
checkpoints/ct_best_model_phase35_v2.pt      sha=93b3a979884edbc4  mtime=1787555730.4296012   MATCHES Phase E §7
backend/main.py                              sha=780120553178c0a5  mtime=1788684518.831581    MATCHES Phase E §7
checkpoints/ct_lidc_nodulecrop_v1.pt         sha=bdcbd6b2e5312435                             MATCHES Phase E §7
checkpoints/ct_lidc_nodulecrop_v2_jitter.pt  sha=5d60055e02974e91                             MATCHES Phase E §7
eval_run/lidc_full/labels_manifest.csv       sha=95d734f64123a46f  3461 rows                  MATCHES Phase E §7
```

**All five match, and both mtimes are unchanged**, so the live checkpoint and `backend/main.py` were
neither rewritten nor rewritten-to-identical-content. Both classifier checkpoints were opened
**read-only**; no checkpoint was written. Every write this phase is inside `eval_run/phaseE1/`
(**4.9 GB, almost entirely cached DICOM**). No new Python dependency. Nothing deployed, nothing
integrated, no `uvicorn` started. **No process was left running** (verified: zero `python.exe`).

One transparency note: this phase was interrupted mid-run by a harness restart. On resume, the
partially-written result files were **verified stale and recomputed from scratch** rather than
reported — the interrupted volume (`LIDC-IDRI-0437`) was re-fetched to completion, and every number
above comes from a single clean run over all 45 volumes.

---

## 7. Final verdict: does this justify a Phase F?

**No. Close the investigation here.**

That is a change from Phase E, which said "report and stop, the decision is the user's" while leaving
the door open. The door is now measured, and what is behind it is worse than Phase E could see.

### What is now known that Phase E did not know

1. **~61% end-to-end is the real number, ±10 points, over 45 patients.** It is not a small-sample
   accident. The most-requested fix from Phase E is done and it confirmed the pessimistic reading.
2. **The classifier's ~74% ceiling is stable and is the binding constraint.** A *perfect* detector
   buys 13 points. Phase F's proposed work — FP aggregation, guardrail chains — cannot touch that
   ceiling; it is all downstream of a second stage that is 74% accurate on crops handed to it
   perfectly centred.
3. **The churn is real, general, and structurally explained**: `v2_jitter` has ~2 mm of through-plane
   tolerance, because the jitter augmentation and the curve that validated it were both in-plane.
   **A detector-fronted pipeline is therefore mis-specified at the component level, not the
   integration level.** No amount of aggregation logic fixes a classifier that changes its mind 40%
   of the time when the crop moves to the adjacent slice.
4. **Scan-level output is 51% with 19 false Malignant calls in 45 scans.** The one part Phase E
   flagged as needing a *design* answer got worse under measurement.

### The honest argument on the other side

**There is a cheap, well-specified fix and it is worth writing down even while recommending against
doing it now.** The through-plane lock is the same defect as Phase C's in-plane lock, on a different
axis, and Phase C.1 fixed that one in **~40 lines and a 9.6-minute training run**. The analogous fix —
jitter the *slice* as well as the in-plane position when building training crops, or move to a small
2.5D/3D crop — is a known-shape experiment with a measured precedent for working. Someone with time
should do it.

**But the honest accounting says it does not rescue this pipeline.** Phase C.1 measured that in-plane
jitter cost ~3 points of centred accuracy to buy in-plane tolerance; a second axis would plausibly cost
similar, from a ceiling that is **already the binding constraint at 74%**. The realistic outcome is a
model that is stable under displacement and *still* around 70–75% at grading — which multiplies out to
roughly where the pipeline already is. **The tolerance fix would make the compounding honest; it would
not make the product good.** That is why this reads as "close it" rather than "one more phase".

### And the constraint that was true all along

Phase A costed a detector as "not a ~10-day-with-an-exam undertaking". Phase C.1 revised that to
"NO-GO on the schedule, for the scope reason alone". **Nothing in this phase makes it smaller** — it
adds a *newly discovered* training-data change, a retrain, a re-validation of the tolerance curve on
all three axes this time, an aggregation redesign, and a guardrail chain that still does not exist.
There are ~10 days and an exam.

### What this phase does NOT show

* **Still LIDC, still the detector's own training lineage** (LUNA16 is LIDC-derived). Bigger n makes
  the *compounding* number solid; it does **not** make it a generalization claim. Phase D's and Phase
  E's identical caveat stands unchanged.
* **16 Malignant is a measurement, not a clinical one.** [0.332, 0.769] is a 44-point interval.
* The 8 mm match radius and 15 mm unmatched radius are Phase D's, reused for comparability, not
  re-derived. Nearest-candidate matching is still per-nodule and independent.
* The stability sweep uses 14 fixed directions at 4 magnitudes — a structured probe, not a uniform
  sample of the sphere. The in-plane/through-plane split is the load-bearing comparison; the
  all-directions number depends on the 4:2:8 direction mix and should be read as such.
* **Phase A's deployment no-go and the standing advice not to cite the live model's
  external-Malignant 12/12 as generalization evidence are untouched by this phase.**

### Recommendation

**Write up the CT investigation and stop.** The arc is complete and it is a genuinely good one: a
shortcut was found, characterised, and removed structurally; the resulting model was shown to learn
real lesion features; its position-lock was diagnosed and *measurably fixed on two axes*; a real
detector was obtained and measured in millimetres; the two stages were wired together and the
compounded number was measured rather than multiplied; and then **the compounding surprise was chased
to its root cause and the root cause turned out to be a third axis nobody had tested.** That is a more
valuable result than a working demo would have been, and it is the natural place to end.

**Phase E.1 closes as: 51/84 (60.7%, CI [0.500, 0.705]) end to end; the 20% churn is real, general,
and caused by a through-plane tolerance of ~2 mm; the classifier's 74% ceiling is unmoved. Do not
build Phase F.**

---

## 8. Artifacts

All under `eval_run/phaseE1/`. Nothing is referenced by `backend/`.

| path | what |
|---|---|
| `select_cases_more.py` | §0 — deterministic selection of 23 unused test-split series, excluding everything Phase D/E already used |
| `add_malignant_cases.py` | §1.3 — the 12 Malignant-carrying series added to fix the all-Benign selection artifact |
| `download_more.py` | §0 — Phase D's per-slice route (`sops`/`fetch_slice` imported) pointed at the new case list |
| `run_eval_e1.py` | §1 — **imports and calls `phaseE/run_eval.py`'s `main()`**; only two path constants change |
| `combine_e2e.py` | §1 — pools the two halves, Wilson / Clopper–Pearson / patient-clustered bootstrap, error decomposition, per-patient churn |
| `stability_sweep.py` | §2 — the per-crop sweep: 4 offsets × 14 directions × 84 nodules, `--ckpt` for the `v1` control |
| `analyze_stability.py` | §2.2/§2.3 — in-plane vs through-plane decomposition, flagged-vs-population, diameter and slice-delta detail |
| `integrity.py` | §6 — re-hash against Phase E §7's recorded digests |
| `_work/pooled_e2e.json` | the pooled 84-nodule result with all three CIs |
| `_work/e2e_results_e1.json`, `_work/scan_reports_e1.json` | the new half's per-nodule outcomes and full per-scan reports |
| `_work/stability_sweep.json`, `_work/stability_analysis.json` | the `v2_jitter` sweep and its decomposition |
| `_work/stability_sweep_v1.json`, `_work/stability_analysis_v1.json` | the `v1` control sweep and its decomposition |
| `_work/cases_e1.json`, `_work/download_log_e1*.json`, `_work/download_e1.log` | selection and download provenance |
| `_work/volumes/` | 35 cached LIDC series (4.9 GB, 9,791 slices) |
| `_work/integrity.json` | §6 |

Reused read-only and unmodified: `eval_run/phaseE/{pipeline,run_eval}.py`,
`eval_run/phaseD/{volume_io,run_detector,download_volumes_perslice}.py`,
`eval_run/phaseD/_work/{cases.json,volumes/}`, `eval_run/lidc_full/extract_crops_full.py`,
`eval_run/lidc_full/labels_manifest.csv`, `ml/{model,dataset}.py`,
`checkpoints/ct_lidc_nodulecrop_{v1,v2_jitter}.pt`.

**This was the last measurement phase. Nothing was integrated; no next phase was started.**
