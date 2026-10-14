# Phase E: the two-stage pipeline, built and measured end to end for the first time

**Scope: build the detector→classifier pipeline as a STANDALONE artifact, run it on Phase D's own
10 volumes / 22 nodules, and report the real compounded number.** Everything written this phase is
confined to `eval_run/phaseE/` (252 KB, 4 Python files + JSON results). **`backend/main.py`,
`/predict/ct` and `checkpoints/ct_best_model_phase35_v2.pt` are untouched — re-hashed against Phase
C.1/D's recorded digests in §7, not asserted.** Nothing was deployed, integrated or wired into the
live app. No new Python dependency was installed. No checkpoint was written.

---

## Verdict up front

**The pipeline works, the wiring is correct, and the compounded accuracy is 13/22 (59.1%) — which is
lower than either stage's individual numbers would lead you to expect, and the reason why is the most
useful thing this phase found.** It is a working proof of concept. It is **not** close to a
live-integration decision.

1. **End-to-end: 13/22 nodules (59.1%) correctly detected AND correctly graded**, at the score ≥ 0.5
   operating point. Detection reproduces Phase D exactly (20/22, 90.9%). Of the 9 failures: **2
   missed by the detector, 3 Benign↔Malignant crossed, 4 detected but called Normal.**
2. **The controls prove the wiring, so the 59.1% is a real result and not a plumbing bug.** The
   GT-centred crop through the identical code path scores **16/22 (72.7%)**, and all 22 of those
   predictions are **bit-identical to Phase C.1's stored `eval_test_v2.json` per-image predictions**,
   from crops that are **bit-exact (max abs diff 0) against Phase B's stored PNGs on 22/22**. The
   extraction path is provably the classifier's own training distribution.
3. **Compounding did NOT multiply cleanly, and it cost more than the tolerance curve predicts.**
   Error budget over 22: **6 the classifier gets wrong even on a perfect annotation-centred crop**,
   **1 lost to a detector miss**, **net 2 lost to the detector's crop centring** (3 flipped wrong,
   1 flipped right). The detector contributes 3 of the 9 errors; the classifier contributes 6.
4. **The new finding this phase exists to produce: 4 of 20 matched predictions (20%) CHANGE CLASS
   between the annotation-centred crop and the detector-centred crop, at a mean displacement of
   2.75 mm** — where Phase C.1's tolerance curve says the accuracy cost is ~zero (0.792 @ 0 mm →
   0.797 @ 4.2 mm). **The tolerance curve is an aggregate and it hides per-crop instability.** Three
   of those four flips are in one scan (LIDC-0321), all at 2.35–2.60 mm, all going from a *confident
   correct* Benign (0.93–0.98) to a *confident wrong* call. §3.3.
5. **Loosening the threshold to ≥ 0.3 buys recall and converts it into nothing.** Detection goes
   20/22 → 21/22, and end-to-end stays at **13/22**: the recovered nodule (LIDC-0433's 7.51 mm,
   found at 2.18 mm) is classified Normal. Phase D's advice that "effort belongs on recall, not
   centring" does not survive contact with the second stage. **0.5 is kept as the default**; §3.4.
6. **False-positive handling is the weakest part and it is a UX problem, not a model problem.**
   2.00 unmatched candidates/scan (reproducing Phase D exactly), classified **12 Benign / 3
   Malignant / 5 Normal**. The classifier's Normal class absorbs only **25%** of them. Worse: **2 of
   the 3 scan-level Malignant over-calls are driven by an unmatched candidate**, including Phase D's
   single genuinely-spurious out-of-lung box (LIDC-0321, detector 0.715), which this classifier
   grades **Malignant at 0.984 confidence**. Scan-level accuracy is **5/10**. §4.
7. **Resources: VRAM is a non-issue, host RAM is the real constraint and it was never measured
   before.** Both models resident: **0.104 GB VRAM**, peak **0.751 GB of 4.00 GB** during inference —
   the classifier adds **+0.06 GB** over Phase D's detector-only 0.69 GB, with **no contention**. But
   **peak host working set is 8.23 GB of this machine's 15.26 GB**, i.e. **54% of system RAM**. §5.
8. **Latency: 33.4 s mean per scan (18.5–45.4 s)**, matching Phase D's ~35 s prediction. **Stage 2 is
   free — 0.2 s mean, 0.6% of wall time.** The job-submission API returns **202 in 3 ms**. §6.

**Honest assessment (§8): this proves the concept and it needs more validation before an integration
decision.** Not because it failed — because 22 nodules in 10 scans is too small to carry a 59.1%
figure, because the per-crop instability in §3.3 is a newly-opened question rather than a closed one,
and because the false-positive aggregation rule in §4 is visibly wrong as designed.

---

## 1. Where the code is, and how it stays separate

All under **`eval_run/phaseE/`**. Nothing in `backend/` imports it; it imports nothing from
`backend/`. It is a different app object, a different route prefix, a different process, and its own
model instances.

| file | lines | what |
|---|---|---|
| **`pipeline.py`** | ~250 | the whole two-stage pipeline: both models in one process, RAS→original-series crop extraction, per-candidate classification, scan-level aggregation, VRAM/RSS probes |
| **`app.py`** | ~150 | standalone FastAPI app — `POST /phaseE/jobs`, `POST /phaseE/jobs/upload` (zip), `GET /phaseE/jobs/{id}`, `GET /phaseE/jobs`, `GET /phaseE/health` |
| **`run_eval.py`** | ~230 | the end-to-end harness — Phase D's matching logic, the two controls, the error decomposition |
| **`test_api.py`** | ~75 | exercises the async job pattern against a real volume via `TestClient` |
| `_work/e2e_results.json` | — | per-nodule outcomes, per-case timings, unmatched candidates, summary |
| `_work/scan_reports.json` | — | the full per-scan output for all 10 scans — what a UI would render |
| `_work/api_demo_LIDC-IDRI-{0433,0586}.json` | — | two complete job payloads as returned by the polling endpoint |
| `_work/run_eval.log` | — | the run transcript |

**What was reused rather than rewritten**, per the brief:

* `phaseD/volume_io.py` — `read_series` and the LPS→RAS affine, **verbatim, not reimplemented**.
* `phaseD/run_detector.py` — `build_detector`, `preprocess`, `vox_to_ras`, **imported**.
* `lidc_full/extract_crops_full.py` — `box_for`, `window`, `OUT_PX`, **imported**. `crop_to_pil` is
  `save_png`'s pixel path without the disk write.
* `ml/model.py` `build_model` + `ml/dataset.py` `build_transforms(train=False)` — the same loading
  and preprocessing Phase C.1's eval used.

**The one genuinely new piece of geometry is `ras_to_series_rc`** — the exact algebraic inverse of
`volume_io.annotation_to_ras`, mapping a detector box centre in RAS mm back to a (slice, row, col) in
the **original** series. This matters: Phase D §7 flagged that crops must come from the original
slice geometry, not the 0.703/0.703/1.25 mm resampled volume, or they stop being distributed like the
classifier's training data. It is ~20 lines and it is verified in §2, not trusted.

**Configuration pinned, per Phase D §2:** `roi_size=[192,192,80]`, the bundle's **native training
patch**, hard-coded in `pipeline.py` and never read from `configs/inference.json` — whose shipped
512×512×192 default silently spills to host RAM on this 4 GB card at 4.4× the runtime.

Classifier: `checkpoints/ct_lidc_nodulecrop_v2_jitter.pt` (epoch 15, val balanced 0.8179), classes
`[Benign, Malignant, Normal]`, 28 mm crop. The class order is asserted against the checkpoint at load
time rather than assumed.

---

## 2. The wiring was verified before any accuracy number was computed

Phase A was burned once by a coordinate convention and Phase D validated its forward conversion
twice. Phase E adds the **inverse** conversion, so it gets its own two checks — and both are
stronger than "plausible", because an exact answer was available.

**Check 1 — round-trip against Phase B's own stored crops.** For all 22 annotated nodules: convert
the manifest position to RAS with `annotation_to_ras`, convert it back with `ras_to_series_rc`, cut
the 28 mm crop from the original slice, and compare against the PNG Phase B actually wrote.

| | result |
|---|---|
| slice (SOPInstanceUID) recovered correctly | **22/22** |
| recovered `center_row` / `center_col` vs the manifest's | **identical to 0.1 px** on all 22 |
| **crop pixels vs Phase B's stored PNG — max absolute gray difference** | **0 on 22/22 (bit-exact)** |

**Check 2 — the classifier's predictions on those crops vs Phase C.1's stored predictions.** All 22
nodules are Phase B/C test-split rows, so Phase C.1's `eval_test_v2.json` already records what
`v2_jitter` said about each. Phase E's from-volume path reproduces **22/22 identically**.

So the extraction path is not merely similar to the classifier's training distribution — **it is
byte-identical to it**. Every number below rests on that, and any degradation measured is a real
property of the pipeline, not an artifact of a subtly different crop.

---

## 3. End-to-end accuracy — the real, compounded number

### 3.1 Headline

22 annotated nodules, 10 patients, both stages actually run in sequence on real DICOM. A nodule
counts as **detected** if a candidate lands within **8 mm** (Phase D's headline radius, same matching
logic — nearest candidate per nodule, computed independently per nodule), and **correct** only if
that same candidate's classification equals the LIDC label.

| | **score ≥ 0.5** (default) | score ≥ 0.3 |
|---|---|---|
| detected | **20/22 (0.909)** | 21/22 (0.955) |
| **END-TO-END CORRECT** | **13/22 (0.591)** | **13/22 (0.591)** |
| — Benign↔Malignant crossed | 3 | 3 |
| — detected but called Normal | 4 | 5 |
| — missed by the detector | 2 | 1 |
| classifier accuracy on what was detected | 0.650 | 0.619 |
| **control: same crops, GT-centred** | **16/22 (0.727)** | — |
| Malignant (n=4) detected / correct | 4 / **2** | 4 / 2 |
| Benign (n=18) detected / correct | 16 / **11** | 17 / 11 |

**Detection reproduces Phase D exactly** — 20/22 at 0.5, 21/22 at 0.3, same two failures — which is
the expected result given the detector is unchanged, and confirms Phase E did not perturb stage 1.

### 3.2 Where the errors come from — the decomposition

This is the part that cannot be obtained by multiplying two published numbers.

| source | nodules lost | note |
|---|---|---|
| **classifier error on a perfect, annotation-centred crop** | **6** | 22 − 16; the ceiling any detector could deliver |
| **detector miss** | **1** | 2 nodules missed, but only 1 of them (LIDC-0433's 7.51 mm) would have been classified correctly anyway; LIDC-0648's 7.11 mm is called Normal even at its own annotation centre |
| **net cost of the detector's crop centring** | **2** | 3 flipped correct→wrong, 1 flipped wrong→correct |
| **total wrong** | **9** | → 13/22 correct |

**Read that top row twice: the classifier is the bottleneck, by a factor of two over the detector.**
Phase D closed by saying "the bottleneck is recall, not centring". Measured end to end, it is
**neither** — it is the second stage's grading accuracy on crops it is already centred on. A perfect
detector would score 16/22 (0.727) here.

That 0.727 is itself consistent with Phase C.1: `v2_jitter` scores 0.7923 nodule accuracy over the
full 207-crop test set, and these 22 are a small subset of exactly that set (16/22 = 0.727 sits
within binomial noise of 0.792 at n=22). **Nothing anomalous — the second stage simply is not very
accurate, and a pipeline cannot be more accurate than it.**

### 3.3 The finding the tolerance curve did not predict: 20% prediction churn at 2.75 mm

Every matched detection lands, as Phase D promised, in the flat part of Phase C.1's tolerance curve —
mean 2.75 mm, max 5.39 mm, against a curve that reads 0.792 at 0 mm and 0.797 at 4.2 mm. On that
basis the displacement should have cost approximately **zero**.

**It cost 4 changed predictions out of 20 (20% churn), for a net of −2 (−10 points).**

| patient | label | 3D err | in-plane / z | slice offset | GT-centred → detector-centred | effect |
|---|---|---|---|---|---|---|
| **0321** | B 9.30 mm | **2.60** | 2.31 / 1.20 | 1 | **Benign (0.978) → Malignant (0.589)** | wrong |
| **0321** | B 7.30 mm | **2.35** | 2.26 / 0.65 | 1 | **Benign (0.932) → Normal (0.970)** | wrong |
| **0321** | B 5.98 mm | **2.40** | 1.92 / 1.45 | 1 | **Benign (0.648) → Normal (0.868)** | wrong |
| 0933 | B 18.59 mm | 3.84 | 1.97 / 3.30 | 2 | Malignant (0.565) → **Benign (correct)** | right |

The other 16 matched nodules keep the same class.

**Two things to say plainly about this:**

* **The tolerance curve measured aggregate accuracy vs displacement, not per-crop stability.** A
  curve can stay flat while individual predictions churn, if roughly equal numbers flip each way.
  Phase C.1's measurement is not wrong; it simply does not answer the question a pipeline asks. This
  is a **new open question**, not a refutation of Phase C.1 — and it is exactly the kind of thing
  that only appears when the two stages actually run together.
* **It is concentrated, not diffuse.** Three of the four flips are the *entire* nodule complement of
  one scan, at displacements of 2.35–2.60 mm — the smallest end of the observed range — and all three
  go from confident-correct to confident-wrong. That is not a gentle degradation with distance; it
  looks like a per-scan effect (some property of LIDC-0321's appearance) that a 22-nodule sample
  cannot distinguish from bad luck. **With n=22 and 3 of the 4 flips in one patient, this phase
  cannot tell you whether the true churn rate is 20%, 5%, or 40%.** That is a sample-size problem and
  §8 treats it as one.

### 3.4 Threshold choice: 0.5 is the default, and 0.3 buys nothing here

Phase D found that dropping the threshold to 0.3 recovers one nodule (LIDC-0433's 7.51 mm Benign,
scored 0.467, localized at 2.17 mm) and framed that as the cheap win. **Run through both stages, the
recovered nodule is classified Normal, so end-to-end accuracy is unchanged at 13/22** — while
unmatched candidates rise from 2.00 to 2.70 per scan (§4). Both operating points are computed on
every run and reported side by side; **0.5 is the default because 0.3 adds noise for zero measured
gain.** The API accepts both, so the comparison stays available rather than being baked in.

---

## 4. False positives, run through the full pipeline — the weakest part

Phase D counted 2.00 unmatched detections per scan and showed most were real-but-unannotated nodules.
The question Phase E can answer that Phase D could not: **what does the second stage do with them,
and does the final scan report look sensible?**

| unmatched candidates (>15 mm from every annotated nodule) | @ 0.5 | @ 0.3 |
|---|---|---|
| count / per scan | **20 / 2.00** | 27 / 2.70 |
| classified **Benign** | 12 | 15 |
| classified **Malignant** | **3** | **3** |
| classified **Normal** (i.e. filtered out by the second stage) | **5 (25%)** | 9 (33%) |

**The classifier's Normal class absorbs only a quarter of them.** Phase D speculated that "a
classifier with a Normal class is exactly what this rate is for" and that a handful of extra crops
per scan "is a workload the second stage is already built to absorb". Measured: it absorbs 25% at the
default threshold. Three-quarters survive into the report as findings.

That is defensible on its own — Phase D showed 13/20 of these are within 4 mm of a real LIDC reader
ROI, so calling them "Benign" is often *correct*. **The problem is what the aggregation rule then
does with them.**

### Scan-level: 5/10, and 2 of the 3 over-calls are driven by a false positive

Scan truth = Malignant if the series carries any annotated Malignant nodule.

| patient | true | pipeline call @0.5 | driver | driver is unmatched? | |
|---|---|---|---|---|---|
| 0657 | Malignant | Benign | c00 | no | **miss** |
| 0828 | Malignant | **Malignant** | c00 | no | OK |
| 0059 | Malignant | **Malignant** | c00 | no | OK |
| 0670 | Malignant | No nodule confirmed | c00 | no | **miss** |
| 0182 | Benign | **Malignant** | c00 | no | **over-call** (real nodule misgraded) |
| 0586 | Benign | **Benign** | c01 | no | OK |
| 0933 | Benign | **Benign** | c00 | no | OK |
| 0648 | Benign | **Benign** | c02 | yes | OK |
| 0433 | Benign | **Malignant** | c02 | **yes** | **over-call from a false positive** |
| 0321 | Benign | **Malignant** | c09 | **yes** | **over-call from a false positive** |

**5/10 scan-level. Three over-calls to Malignant, two of them driven by a candidate with no annotated
nodule anywhere near it.** The worst example deserves to be quoted in full, because it is the exact
box Phase D identified as the single genuinely spurious detection in all 10 whole-chest scans
(LIDC-0321, score 0.715, lung fraction 0.00):

```
LIDC-IDRI-0321 (265 sl, GE) — 30 raw detections → 10 candidates @0.5 — CALL: Malignant
  c09  det 0.715  Malignant  B/M/N [0.0044, 0.984, 0.0117]  size 17.6x17.8x14.7 mm  UNMATCHED (248 mm from any GT)
  c00  det 0.996  Malignant  B/M/N [0.4013, 0.5891, 0.0096]  size  8.1x 8.2x 8.1 mm  on annotated nodule (Benign 9.30 mm)
  c01  det 0.991  Malignant  B/M/N [0.0167, 0.5159, 0.4674]  size  6.2x 6.3x 6.3 mm  UNMATCHED (168 mm)
  c02  det 0.976  Benign     B/M/N [0.6969, 0.1759, 0.1272]  ...                     UNMATCHED (101 mm)
  c06  det 0.809  Normal     B/M/N [0.1734, 0.1460, 0.6806]  ...                     UNMATCHED ( 90 mm)
  c03  det 0.972  Benign     B/M/N [0.9748, 0.0223, 0.0029]  ...                     UNMATCHED ( 78 mm)
  c08  det 0.792  Benign     B/M/N [0.5806, 0.0129, 0.4065]  ...                     UNMATCHED ( 75 mm)
  c04  det 0.956  Normal     B/M/N [0.0185, 0.0109, 0.9705]  ...                     on annotated nodule (Benign 7.30 mm)
  c05  det 0.887  Benign     B/M/N [0.8492, 0.0067, 0.1441]  ...                     UNMATCHED (189 mm)
  c07  det 0.804  Normal     B/M/N [0.1287, 0.0031, 0.8682]  ...                     on annotated nodule (Benign 5.98 mm)
  GROUND TRUTH: three Benign nodules (9.30, 7.30, 5.98 mm)
```

**A scan whose truth is three Benign nodules produces ten findings, three of them graded Malignant,
and an overall call of Malignant driven by a 17.6 mm box outside the lung that the classifier is
98.4% confident about.** That is not a subtle failure. It is exactly the "noisy over-calling" the
brief asked about, and the answer is yes, on this scan it looks like noisy over-calling.

For contrast, the same pipeline on a clean scan looks entirely sensible:

```
LIDC-IDRI-0586 (120 sl, GE) — 15 raw → 5 candidates @0.5 — CALL: Benign
  c01  det 0.929  Benign  [0.9108, 0.0590, 0.0302]  on annotated nodule (Benign 5.76 mm)   correct
  c00  det 0.940  Benign  [0.9437, 0.0335, 0.0229]  on annotated nodule (Benign 4.03 mm)   correct
  c03  det 0.586  Benign  [0.9739, 0.0260, 0.0001]  UNMATCHED (66 mm)
  c02  det 0.837  Benign  [0.9811, 0.0185, 0.0004]  on annotated nodule (Benign 6.91 mm)   correct
  c04  det 0.521  Normal  [0.0152, 0.0071, 0.9777]  UNMATCHED (90 mm)  <- absorbed by the Normal class
  GROUND TRUTH: three Benign nodules (6.91, 5.76, 4.03 mm) — all three found and graded correctly
```

and LIDC-0433 (the API demo payload) shows the threshold artifact in situ — c04 at detector 0.467 is
the nodule Phase D recovers at 0.3, and the second stage calls it Normal:

```
LIDC-IDRI-0433 — CALL: Malignant  (driver c02, which is UNMATCHED at 92 mm)
  c02  det 0.957  Malignant [0.0512, 0.5443, 0.4045]  UNMATCHED (92 mm)   <- drives the whole call, on a 0.544 probability
  c01  det 0.972  Benign    [0.9342, 0.0601, 0.0057]  on annotated nodule  correct
  c00  det 0.992  Benign    [0.9747, 0.0017, 0.0236]  on annotated nodule  correct
  c03  det 0.598  Benign    [0.9750, 0.0003, 0.0247]  UNMATCHED
  c04  det 0.467  Normal    [0.2087, 0.0002, 0.7911]  (only @0.3) on annotated nodule (Benign 7.51 mm) -> Normal
```

### What this says about the aggregation rule

The brief proposed "highest-malignancy candidate drives the call, but report everything", and that is
what was implemented — and it is **visibly the wrong rule as specified**. A single high-confidence
false positive flips an entire scan, and a 0.544 probability is treated identically to a 0.984 one.
**Reporting all candidates (which the implementation does) is right and should be kept. Letting the
maximum drive a single binary scan-level call is not.** That is a design decision that needs a real
answer — a probability threshold, a size/anatomy prior, an in-lung gate on candidates, or simply
declining to emit a scan-level call at all and presenting a ranked finding list. **This phase does
not pick one; that is a UX decision for the user, and §8 lists it as a blocker rather than a
detail.**

---

## 5. Resources — both models in one process, measured for the first time

The brief specifically asked not to assume 0.69 GB + classifier adds up cleanly. It does, on the GPU.
The surprise is elsewhere.

### GPU: no contention, and the classifier is nearly free

| state | allocated | peak allocated | reserved |
|---|---|---|---|
| process start | 0.000 GB | 0.000 | 0.000 |
| **detector loaded** | 0.078 GB | 0.078 | 0.123 |
| **+ classifier loaded (both resident)** | **0.104 GB** | 0.131 | 0.172 |
| **peak during a full scan, both resident** | — | **0.751 GB of 4.00** | 0.941 |

**Phase D's detector-only peak was 0.69 GB; with the classifier coexisting it is 0.751 GB — +0.06 GB,
i.e. +9%, and unvarying across all 10 scans.** No fragmentation, no allocator contention, no OOM, no
observed interaction of any kind. Peak reserved (0.941 GB) is 24% of the card. The classifier's
resident cost is 26 MB — it is a DenseNet-121 on 128 px crops, three orders of magnitude cheaper than
the sliding-window detector it follows.

Model load time: **detector 0.7 s, classifier 0.3 s**, once per process.

### Host RAM: this is the actual constraint, and nobody had measured it

| | GB |
|---|---|
| process RSS before any model | 0.71 |
| RSS with both models loaded, idle | 1.10 |
| RSS during scans (range across the 10) | 2.34 – 6.56 |
| **peak working set over the whole run** | **8.23** |
| machine total physical RAM | **15.26** |

**Peak host memory is 54% of this machine's RAM — roughly 11× the peak VRAM.** The causes are
structural, not a leak (the per-scan figures fluctuate with volume size rather than climbing
monotonically): the float32 HU volume, MetaTensor copies through `Orientation` and `Spacing`, and —
the big one — the sliding-window inferer's output buffer, which Phase D deliberately placed on the
**CPU** (`device=torch.device("cpu")`, `sw_device=cuda`) precisely to keep VRAM at 0.69 GB. **That
trade was never priced. It is paid in host RAM, and this is where it shows up.**

Practical consequence worth stating plainly: on this laptop, with a browser and the project's own
`uvicorn` dev server also running, an 8.23 GB peak is close enough to 15.26 GB that a second
concurrent job would be a real risk. `app.py`'s `MAX_CONCURRENT = 1` is not a placeholder — it is the
correct setting for this machine, and §8 lists "measure and bound host RAM" as required work.

---

## 6. Latency and the async job pattern

### Measured, full pipeline, 10 scans, RTX 3050 Laptop

| stage | mean | min | max |
|---|---|---|---|
| read DICOM series + resample | **8.1 s** | 3.4 | 18.1 |
| **stage 1 — detection** | **25.1 s** | 14.9 | 33.6 |
| **stage 2 — classification (all candidates)** | **0.2 s** | 0.0 | 0.4 |
| **total** | **33.4 s** | **18.5** | **45.4** |

**Phase D predicted "20–50 s per scan, ~35 s typical" from its component numbers. Measured end to
end: 18.5–45.4 s, mean 33.4 s. The prediction held.** Detection is 75% of the wall time, DICOM
parsing 24%, and **the second stage is 0.6%** — classifying up to 10 crops costs less than a fifth of
a second, so the classifier is free at any plausible candidate count. On CPU the whole thing would be
~3.5 min (Phase D measured the detector at 14× slower; stage 2 is negligible either way).

### The job/polling implementation

`app.py` implements exactly what Phase D said was needed and nothing more:

* `POST /phaseE/jobs` (a series directory) or `POST /phaseE/jobs/upload` (a zip) → **HTTP 202 with a
  job id, measured at 3 ms**, while a daemon thread runs the pipeline;
* `GET /phaseE/jobs/{id}` → live `status` / `progress` / `stage` ("reading DICOM series" → "stage 1:
  detecting on [607, 607, 244] volume" → "stage 2: classifying 5 candidates" → "done"), then the
  full result;
* input validation is **synchronous** — a non-existent directory is a 400 before any work is
  scheduled, an unknown job id is a 404;
* both models are loaded **once, lazily, behind a lock**, and shared by every job — loading per job
  would re-pay ~1 s and briefly double VRAM;
* `MAX_CONCURRENT = 1`, because one 4 GB GPU and an 8 GB host-RAM peak means queueing beats OOM.

**Deliberately not Celery/Redis/a broker**, per `PROJECT_STATUS.md` §8's local-only scope. The honest
cost of that choice, written down rather than glossed: **jobs die with the process, there is no
cross-worker sharing, results are never evicted so the dict grows until restart, and there is no
retry.** For a local demo of a ~35 s pipeline that is the right size of solution; for anything else it
is not, and swapping the in-memory dict for a real queue is the change.

Verified end to end by `test_api.py` against LIDC-0433 (a 764-slice-class Philips series): submitted
in 3 ms, polled through all three stages, completed at 45.6 s, full payload returned.

---

## 7. Live-system integrity

Re-hashed **after** all inference, against the digests recorded in `phaseD_detector_test.md` §9 and
`phaseC1_followups.md` §7. sha256, first 16 hex characters — the same formula those phases used,
confirmed by the fact that all five reproduce exactly.

```
checkpoints/ct_best_model_phase35_v2.pt      sha=93b3a979884edbc4  mtime=1787555730.4296012   MATCHES Phase D §9
backend/main.py                              sha=780120553178c0a5  mtime=1788684518.831581    MATCHES Phase D §9
checkpoints/ct_lidc_nodulecrop_v1.pt         sha=bdcbd6b2e5312435  mtime=1788893737.965406    MATCHES Phase D §9
checkpoints/ct_lidc_nodulecrop_v2_jitter.pt  sha=5d60055e02974e91  mtime=1788895758.058575    MATCHES Phase D §9
eval_run/lidc_full/labels_manifest.csv       sha=95d734f64123a46f  3461 rows                  MATCHES Phase D §9
```

**All five match. `backend/main.py` and the live checkpoint are byte-identical to their state at the
end of Phase D**, and their modification times are unchanged, so they were neither rewritten nor
rewritten-to-identical-content. The classifier checkpoint was opened **read-only**; no checkpoint was
written this phase.

Also true:

* **Every write this phase is inside `eval_run/phaseE/`** — 252 KB total (4 source files + ~160 KB of
  result JSON). No new DICOM was downloaded; Phase D's cached `_work/volumes/` was read only.
* **No new dependency was installed.** The host-RAM probe goes to `psapi` through `ctypes` precisely
  so that `psutil` did not have to be added.
* **Nothing was deployed or integrated.** `app.py` is a separate module with its own `app` object and
  a `/phaseE/` route prefix; it was only ever run in-process via `TestClient`. No `uvicorn` server was
  started for it, and no frontend was touched.
* **No process was left running.** The project's own `uvicorn main:app --reload` dev server was
  running before this phase and was not touched.

---

## 8. Honest assessment: is this ready for a live-integration decision?

**No. It is a working proof of concept that has answered its question and opened three new ones.**

The pipeline runs, the geometry is provably correct, the resources fit, and the latency is what was
predicted. That was the engineering risk and it is now retired. What is not retired is whether the
thing is *good enough to show anyone*, and on the evidence here it is not yet.

### What genuinely landed

* Both stages run together in one process on real DICOM, at 33 s/scan and 0.751 GB of VRAM, with
  **no contention between the models**.
* The crop path is **bit-exact** against Phase B's stored crops and reproduces Phase C.1's
  predictions **22/22** — so the compounded number is a real measurement, not a plumbing artifact.
* The async job shape works and is the right size for the scope.
* **A real compounded number now exists: 13/22 (59.1%).** Nobody has to multiply 0.909 by 0.792 and
  guess any more — and note that guess would have given 0.72, materially higher than the measured
  0.591. **Compounding did not multiply cleanly, exactly as the brief suspected.**

### What blocks an integration decision

1. **22 nodules in 10 scans cannot carry a 59.1% headline.** The 95% confidence interval on 13/22 is
   roughly 0.39–0.78. Three of the four displacement flips are in *one patient*. The scan-level 5/10
   is ten scans. **Every important number in this report is one or two patients away from moving
   several points.** A larger test set is not a nice-to-have; it is the precondition for any of this
   being decision-grade.
2. **The per-crop instability in §3.3 is a new, unanswered question.** 20% of predictions change
   class at a displacement the tolerance curve says is free. Either the churn is real and the
   classifier is less stable than Phase C.1's aggregate curve suggests, or it is a LIDC-0321-specific
   artifact. **Those two possibilities have very different implications and 22 nodules cannot
   separate them.** The measurement to run is a per-crop stability sweep (same nodule, many small
   displacements, count class changes rather than mean accuracy) — a Phase C.1-style experiment,
   cheap, and it should happen before anything is wired anywhere.
3. **The false-positive UX has no defensible design yet (§4).** The max-driven scan call is wrong as
   implemented — it lets one spurious box flip a scan, and it did so on 2 of 10. This is not a bug to
   patch quietly; it is a product decision about what a scan-level output even *is* when the tool
   emits ~4 findings of which ~2 are unannotated. Until that is decided, there is nothing coherent to
   put in a UI.
4. **The second stage, not the detector, is the accuracy ceiling — and it is 0.727 here.** Even a
   perfect detector yields 16/22. Integrating this pipeline would be shipping `v2_jitter`'s grading
   accuracy behind a more impressive front end. That may still be worth doing as a demonstration, but
   **it should be done with eyes open: the detector is the strong half.**
5. **The guardrail chain Phase D §6 identified was not built and is not in this artifact.** There is
   no series-level sanity gate, no modality gate, no OOD check — `pipeline.py` will happily run on any
   directory of DICOM files. That was correct for a standalone measurement harness and is
   disqualifying for anything user-facing.
6. **Host RAM (§5) needs a bound before this shares a machine with the live app.** 8.23 GB peak of
   15.26 GB, next to a dev server and a browser, with `MAX_CONCURRENT = 1` as the only protection.

### What this does not show

* **Still LIDC, still the detector's own training lineage** (LUNA16 is LIDC-derived). This measures
  wiring and compounding, **not generalization**. Nothing here licenses a generalization claim, and
  Phase D's identical caveat stands.
* **18 Benign / 4 Malignant.** All 4 Malignant were detected; 2 of 4 were graded correctly. **n=4 is
  not a Malignant accuracy measurement** and must not be quoted as one.
* The 8 mm match radius and the 15 mm unmatched radius are Phase D's, chosen there and reused here
  for comparability — not re-derived.
* Nearest-candidate matching is per-nodule and independent, so in principle one candidate could match
  two nodules. It did not happen here, but the harness does not forbid it.
* Phase A's deployment no-go and the standing advice not to cite the live model's external-Malignant
  12/12 as generalization evidence are both untouched by this phase.

### Recommendation

**Report and stop, which is what the brief asked for.** The integration decision is the user's and
this phase does not make it. If asked for a lean: **do not integrate yet — not because the pipeline
failed, but because 59.1% on 22 nodules with an unresolved false-positive UX is a result to
investigate, not a feature to ship.** The two cheapest things that would move the decision are (a) a
per-crop stability sweep to resolve §3.3, and (b) a bigger test set — neither of which is a
~10-days-before-an-exam job either, and both of which are smaller than what Phase E just did.

**Phase E closes as: built, wired correctly, measured honestly — 13/22 end to end, concept proven,
not ready.**

---

## 9. Artifacts

All under `eval_run/phaseE/`. Nothing is referenced by `backend/`.

| path | what |
|---|---|
| **`pipeline.py`** | the two-stage pipeline; `ras_to_series_rc` (the new inverse geometry), `crop_at_ras`, `TwoStagePipeline`, `aggregate`, VRAM/RSS probes |
| **`app.py`** | the standalone FastAPI job-submission/polling app (`/phaseE/*`) — never served, never integrated |
| `run_eval.py` | §2/§3/§4/§5/§6 — the end-to-end harness, both controls, the error decomposition |
| `test_api.py` | §6 — the async job pattern exercised against a real volume |
| `_work/e2e_results.json` | per-nodule outcomes at both thresholds, per-case timings/memory, unmatched candidates, summary |
| `_work/scan_reports.json` | the complete per-scan pipeline output for all 10 scans (§4's examples come from here) |
| `_work/api_demo_LIDC-IDRI-0433.json`, `_work/api_demo_LIDC-IDRI-0586.json` | two full job payloads as the polling endpoint returns them |
| `_work/run_eval.log` | the run transcript |

Reused read-only and unmodified: `eval_run/phaseD/{volume_io,run_detector}.py`,
`eval_run/phaseD/_work/{cases.json,volumes/}`, `eval_run/lidc_full/extract_crops_full.py`,
`eval_run/lidc_full/labels_manifest.csv`, `ml/{model,dataset}.py`,
`checkpoints/ct_lidc_nodulecrop_v2_jitter.pt`.

**Nothing was integrated into the live app. That decision is deferred to the user.**
