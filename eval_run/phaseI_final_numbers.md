# Phase I — final measurement pass: one consistent set of current numbers

**Pure measurement. Zero code, checkpoint, or config changes.** Every number below was produced in
this phase, on 2026-09-10, against a live backend started on an isolated scratch DB
(`DB_PATH=<scratchpad>/phaseI.db python -m uvicorn main:app --host 127.0.0.1 --port 8091` from
`backend/`), with every request made over real HTTP from a separate process. **Nothing is copied
forward from an earlier phase.** Backend stopped cleanly afterwards: `Get-Process python` returns 0
and port 8091 is free.

Pre-flight, from the startup log (`<scratchpad>/backend_phaseI.log`): **all six reproduction checks
pass**, identical to Phases H and the X-ray crop-fix — xray 0.9646, ct 0.9904, gate 1.0000,
xray_sanity 0.9996, ct_sanity 1.0000, seg dice 0.9830 / iou 0.9671. `GET /health` lists the same
five resident models; `GET /predict/ct/volume` answers with `models_loaded: false`, and the volume
models lazy-loaded on the first job as designed.

Integrity, re-hashed after all testing (first 16 hex, unchanged from Phase H §8):
`ct_best_model_phase35_v2.pt` `93b3a979884edbc4` · `ct_lidc_nodulecrop_v4_stack3.pt`
`5ee4d8ec9f8328c1` · `xray_txrv_densenet121_best.pt` `1e38a54bf94d466e` · `backend/main.py`
`816b06d0b78ec572` · `backend/seg_utils.py` `2a0adcb358ad55d7` · `eval_run/phaseE/pipeline.py`
`936427fc20d389a0`. `git status --porcelain checkpoints/` is empty. No volume was re-downloaded.

**Rate limiting did not touch the measurement: 0 HTTP 429 across all 347 `/predict/*` calls**,
paced globally at under 20/min (the documented per-route limit).

---

## Master table

Accuracy is **over images the API actually served (HTTP 200)**. Some images are refused before the
classifier by pre-existing guardrails (OOD 422, colour-content 400); those are counted in the `n`
column as `served/submitted` and the stricter convention is given in the second table.

| Model | Class | Dataset n | Dataset accuracy % | External n | External accuracy % |
|---|---|---|---|---|---|
| X-ray | COVID-19 | 40/40 served | **87.5** (35/40) | 7/10 served | **71.4** (5/7) — n=10 small-sample |
| X-ray | Normal | 40/40 served | **90.0** (36/40) | 10/10 served | **100.0** (10/10) — n=10 small-sample |
| X-ray | Pneumonia | 40/40 served | **92.5** (37/40) | 10/10 served | **100.0** (10/10) — n=10 small-sample |
| X-ray | Tuberculosis | 56/63 served | **92.9** (52/56) | 6/10 served | **100.0** (6/6) — n=10 small-sample |
| CT (single-image) | Benign | 12/12 served | **83.3** (10/12) | **0** | **unavailable — n=0** |
| CT (single-image) | Malignant | 50/50 served | **100.0** (50/50) | 12/12 served | **100.0** (12/12) |
| CT (single-image) | Normal | 41/41 served | **100.0** (41/41) | 9/9 served | **0.0** (0/9) |

X-ray Tuberculosis used **all 63** images on disk; CT used **all** of `data/ct_scan/test/`. COVID-19,
Normal and Pneumonia were sampled at 40 each (seed 3) from the 357 / 1368 / 134 available.

**CT (single-image) Benign external is n=0 and is reported as a row, not omitted.** No independent
non-IQ-OTH/NCCD source of benign lung-CT images was ever found (documented in
`eval_run/external_samples/EXTERNAL_EVAL_SUMMARY.md`); this row was not filled with an
IQ-OTH/NCCD-derived image and no accuracy is estimated for it.

### Same numbers counting guardrail rejections as failures

| Model | Class | Dataset | External |
|---|---|---|---|
| X-ray | COVID-19 | 87.5 (35/40) | **50.0** (5/10) — 3 rejected 422 |
| X-ray | Normal | 90.0 (36/40) | 100.0 (10/10) |
| X-ray | Pneumonia | 92.5 (37/40) | 100.0 (10/10) |
| X-ray | Tuberculosis | **82.5** (52/63) — 7 rejected 422 | **60.0** (6/10) — 4 rejected 400 |
| CT (single-image) | all three | unchanged (nothing rejected) | unchanged (nothing rejected) |

Overall: X-ray dataset **90.9%** over served (160/176) / 87.4% (160/183) counting rejections;
X-ray external **93.9%** over served (31/33) / 77.5% (31/40). CT dataset **98.1%** (101/103);
CT external **57.1%** (12/21).

### CT volume pipeline — `POST /predict/ct/volume`, live HTTP submit + poll

Not per-class, so it sits alongside the table rather than inside it. All 45 LIDC volumes already on
disk (Phase D + Phase E.1) were submitted as loose `.dcm` multipart uploads and polled to
completion. **45/45 jobs succeeded**, 29.2 min wall, mean 38.7 s/scan (range 20.4–84.3 s, median
41.0 s), host RSS peak 9.9 GB. Scoring reproduces `phaseG3/run_g3_validation.py::end_to_end_pipeline`
exactly — GT match radius 8 mm, FP radius 15 mm, detector threshold 0.5, and G.3's own scan-level
rule (any candidate graded Malignant makes the scan Malignant).

| Metric | Dataset n | Dataset result | 95% Wilson CI | External n | External result |
|---|---|---|---|---|---|
| End-to-end nodule accuracy | 84 nodules / 45 patients | **63.1%** (53/84) | **[52.4%, 72.6%]** | **0** | **unavailable — see below** |
| Scan-level accuracy | 45 patients | **53.3%** (24/45) | **[39.1%, 67.1%]** | **0** | **unavailable — see below** |

Supporting figures from the same run: detector recall **84.5%** (71/84 nodules proposed within
8 mm); end-to-end at threshold 0.3 **64.3%** (54/84, [53.6%, 73.7%]); of the 84 nodules, 53 correct,
9 Benign/Malignant crossed, 9 called Normal, 13 never proposed by the detector. Benign nodules
67.6% (46/68), Malignant nodules 43.8% (7/16). 10 of the benign-only scans were over-called
Malignant. Unmatched candidates at ≥0.5: 119 (69 Benign / 17 Malignant / 33 Normal).

### CT volume pipeline — external data: unavailable, and why

**Exactly 45 LIDC-IDRI volumes exist anywhere on disk** (`eval_run/phaseD/_work/volumes/` 10 +
`eval_run/phaseE1/_work/volumes/` 35; a repo-wide search for `LIDC-IDRI-*` directories returns those
45 and nothing else). All 45 are the Phase G.3 test set, i.e. all "already used". **No genuinely
unused LIDC patient IDs are on disk, so this row is unavailable.** No new volumes were downloaded:
manufacturing an "external" row from freshly-downloaded LIDC patients would still be the detector's
own training lineage, and re-labelling the Phase G.3 set as external would double-count already-seen
data, which the brief explicitly warns against. Reported unavailable rather than filled.

---

## Confusion breakdowns — what the wrong predictions actually became

### 1. X-ray, dataset (`data/xray/test/`), live `/predict/xray`

Rows = truth, columns = what the API returned.

| true \ pred | COVID-19 | Normal | Pneumonia | Tuberculosis | rejected 422 (OOD) |
|---|---|---|---|---|---|
| **COVID-19** (40) | **35** | 1 | 3 | 1 | 0 |
| **Normal** (40) | 1 | **36** | 1 | 2 | 0 |
| **Pneumonia** (40) | 1 | 2 | **37** | 0 | 0 |
| **Tuberculosis** (63) | 4 | 0 | 0 | **52** | 7 |

The only concentrated error mode left is **Tuberculosis → COVID-19 (4)**, which is exactly the
residual class-conditional crop mismatch the crop-fix write-up predicted would survive
(`xray_cropfix_applied.md` §4, item 1). COVID-19's errors are diffuse (3 Pneumonia, 1 Normal,
1 TB) rather than pointing at one attractor. The 7 TB rejections are the OOD detector, the
pre-existing **N3** finding.

### 2. X-ray, external (`eval_run/external_samples/xray/`), live `/predict/xray` — **all n=10**

| true \ pred | COVID-19 | Normal | Pneumonia | Tuberculosis | rejected |
|---|---|---|---|---|---|
| **COVID-19** (10) | **5** | 2 | 0 | 0 | 3 × 422 (OOD) |
| **Normal** (10) | 0 | **10** | 0 | 0 | 0 |
| **Pneumonia** (10) | 0 | 0 | **10** | 0 | 0 |
| **Tuberculosis** (10) | 0 | 0 | 0 | **6** | 4 × 400 (colour content) |

Every one of these rows is n=10 by construction and is **reported unpadded**. At n=10 a single
image is 10 percentage points, so none of these should be read as a precise rate.

---

## Freshness

**Every number in this document was measured in this phase.** Nothing is inherited. Specifically:

* All 347 single-image predictions are fresh live HTTP calls made 2026-09-10 against port 8091.
* All 45 volume jobs are fresh live HTTP submissions and polls, run this phase; only the DICOM
  files themselves are reused from disk, which is what the brief instructed.
* The volume scoring re-derives ground-truth RAS positions from the DICOM series itself; it does
  not read Phase G.3's stored per-nodule outcomes as input. G.3's stored results are used **only**
  afterwards, as an independent cross-check (below).
* The 63.1% figure below is a **fresh measurement that happens to equal** the earlier one — it is
  not the earlier one copied in.

---

## Numbers that differ from what an earlier phase reported

**1. X-ray external COVID-19 got worse, and this is genuinely new information.** Pre-crop-fix
(`external_samples/results_external.json`) it was 7 correct / 1 Pneumonia / 1 Normal / 1 rejected
422. Now it is **5 correct / 2 Normal / 3 rejected 422**. Only Tuberculosis had been re-tested
post-fix, so this is the first look at external COVID-19 since the fix. Best read: the crop fix
changes the tensor the OOD detector scores, so OOD distances moved — `xray_cropfix_applied.md` §3.5
measured that movement on `data/xray/test/` and found it *helpful* there (COVID-19 mean distance
5.028 → 4.878). These external files are a different source distribution, and on them the shift went
the other way, costing 2 extra rejections. The 2 remaining errors are COVID-19 → Normal, i.e.
under-calling, which is the opposite of the pre-fix false-COVID failure mode. **At n=10 this is 2
images; it is a signal worth noting, not a measured regression.**

**2. X-ray external Normal and Pneumonia got better, for a well-understood reason.** Normal went
7/10 → **10/10** and Pneumonia 9/10 → **10/10**. Every pre-fix error in those two rows was a flip to
COVID-19 (3 and 1 respectively). That is precisely the false-COVID attractor the crop fix removed,
so the improvement is expected and consistent with the dataset-split result.

**3. X-ray external Tuberculosis is unchanged at 6/6 served, 4 rejected 400** — identical to
`xray_cropfix_applied.md` §3.4. The colour-content gate runs before any crop, so the fix could not
have moved it, and it did not.

**4. X-ray dataset per-class differs slightly from the crop-fix run, within sampling noise.** Crop-fix
§3.3 measured COVID-19 0.900 / Normal 0.900 / Pneumonia 0.950 / TB 0.925 over served on a seed-3
60-per-class sample; here it is 0.875 / 0.900 / 0.925 / 0.929 on a 40/40/40/63 sample. Different
sample sizes mean different images, and each class differs by at most one or two images. Tuberculosis
is now measured on **all 63** available rather than a 60-image subsample, so 92.9% (52/56 served) is
the more complete figure.

**5. CT single-image is unchanged, as it should be.** External CT is **bit-identical** to the
pre-existing run — Malignant 12/12, Normal 0/9 with the same 7 Malignant / 2 Benign split. Dataset
Benign is 10/12, the same 10/12 as before. Malignant and Normal were previously measured on 25-image
samples (25/25 each) and are now measured on **all 50 and all 41**, still 100%. Nothing in the CT
path was touched by Phases H or the crop fix, and the numbers confirm it.

**6. CT external Normal remains 0/9 — the known shortcut-learning result, re-confirmed live.** Seven
of nine independent Normal CTs are called Malignant and two Benign; none is called Normal, while the
same class scores 41/41 on the project's own test split. This is the finding the whole CT
investigation was built around, and it is unchanged.

**7. The volume pipeline reproduces Phase G.3 exactly — no difference at all.** The brief allowed for
a discrepancy from real HTTP round-trips; there is none.

| | fresh live-HTTP measurement (this phase) | Phase G.3 |
|---|---|---|
| end-to-end @0.5 | **53/84 = 63.1%**, Wilson [0.5242, 0.7263] | 53/84 = 0.631, [0.5242, 0.7263] |
| end-to-end @0.3 | 54/84 = 64.3%, [0.5362, 0.737] | 54/84 = 0.6429, [0.536, 0.737] |
| detector recall | 71/84 = 84.5% | 71/84 = 84.5% |
| scan-level | 24/45 = 53.3% | 24/45 = 0.5333 |
| benign scans over-called Malignant | 10 | 10 |
| unmatched candidates ≥0.5 | 119 (69 B / 17 M / 33 N) | 119 |

Two stronger checks were run on top of the aggregate agreement, and both are exact:

* **All 229 candidates returned by the live endpoint across all 45 patients** are bit-identical to
  the stored Phase E/E.1 detector pool — max absolute detector-score difference **0.00000**, max
  absolute `center_ras_mm` difference **0.0 mm**, no missing or extra candidates on any patient.
  This extends Phase H §5.2's 16-candidate / 4-patient check to the **entire** test set.
* **All 84 per-nodule rows** match G.3's stored `outcome@0.5` and `pred@0.5` — **0 differing**.

So the aggregation-threshold concern raised in the brief does not arise: the live route's report
floor is 0.30 and the stored detector pool's lowest score is 0.301, so the returned candidate set is
exactly the pool G.3 scored, and the aggregation `concern` levels are a separate presentation layer
that the G.3-comparable scoring does not consult. The deployed pipeline is, on the full test set and
not just a smoke sample, the pipeline that was validated.

---

## What this phase does not show

* **No external validation of the volume pipeline exists**, and none was manufactured. Every volume
  number is LIDC-IDRI, the detector's own training lineage.
* **The external X-ray and CT rows are n≤12.** They are reported as-is and should not be quoted as
  rates without their denominators.
* **Guardrail rejections are not modelled as a cost anywhere in the served-accuracy column.** The
  second table gives the strict reading; which one is appropriate depends on whether a refusal is
  treated as a safe abstention or a failure.
* This phase measures the system as deployed. It changes nothing and diagnoses nothing new — the
  open items from `xray_cropfix_applied.md` §4 (class-conditional crop mismatch, N3, N4) and from
  `phaseH_deployment.md` §7 (no volume OOD, no body-region check, uncalibrated concern levels,
  in-memory jobs) all still stand.

## Artifacts

Measurement scripts and raw outputs live in the session scratchpad, not the repo:
`phaseI_singles.py` / `phaseI_singles.json` (347 single-image calls, every row with status, prediction
and confidence), `phaseI_volumes_http.py` / `phaseI_volumes_raw.json` (45 live volume jobs with
timings and full candidate lists), `phaseI_volumes_score.py` / `phaseI_volumes_scored.json` (G.3-method
scoring, per-nodule and per-case rows), `backend_phaseI.log` (startup reproduction checks).
