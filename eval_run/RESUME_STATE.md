# Resume state — CT shortcut-learning investigation

## Update 2026-09-10 — Phase I: FINAL MEASUREMENT PASS — **one consistent set of current numbers for every model, all measured fresh over live HTTP this phase. Zero code/checkpoint/config changes. The volume pipeline reproduces Phase G.3 EXACTLY on the full test set: 53/84 = 63.1% [52.4%, 72.6%], and all 229 candidates across all 45 patients are bit-identical to the stored detector pool.**

Full write-up: `eval_run/phaseI_final_numbers.md`. **Pure measurement — nothing was modified.**
Backend started on an isolated scratch DB (`DB_PATH=<scratchpad>/phaseI.db`, port 8091), all six
startup reproduction checks passed unchanged (xray 0.9646, ct 0.9904, gate 1.0000, xray_sanity
0.9996, ct_sanity 1.0000, seg dice 0.9830), volume models lazy-loaded on the first job as designed,
and the backend was **stopped cleanly — `Get-Process python` returns 0, port 8091 free**. All
protected digests re-hashed after the work and unchanged (`ct_best_model_phase35_v2.pt`
`93b3a979884edbc4`, `v4_stack3` `5ee4d8ec9f8328c1`, `xray_txrv_densenet121_best.pt`
`1e38a54bf94d466e`, `backend/main.py` `816b06d0b78ec572`, `backend/seg_utils.py` `2a0adcb358ad55d7`,
`phaseE/pipeline.py` `936427fc20d389a0`); `git status --porcelain checkpoints/` empty. No volume
re-downloaded. **0 HTTP 429 across all 347 `/predict/*` calls** — rate limiting did not corrupt
anything. All scripts and raw JSON live in the session scratchpad, not the repo.

### 1. X-ray, live `/predict/xray` — dataset and external, both fresh
Dataset (seed 3; TB used **all 63** on disk): COVID-19 **35/40 = 87.5%**, Normal **36/40 = 90.0%**,
Pneumonia **37/40 = 92.5%**, Tuberculosis **52/56 served = 92.9%** (7 OOD 422s; 82.5% counting them).
Overall 160/176 = **90.9%** over served. The only concentrated error left is **TB → COVID-19 (4)** —
exactly the residual class-conditional crop mismatch `xray_cropfix_applied.md` §4 predicted would
survive. COVID-19's 5 errors are diffuse (3 Pneumonia, 1 Normal, 1 TB), i.e. the false-COVID
attractor is gone.
External (**all n=10, reported unpadded**): COVID-19 **5/7 served** (3 × 422), Normal **10/10**,
Pneumonia **10/10**, TB **6/6 served** (4 × 400 colour gate).

### 2. External COVID/Normal/Pneumonia post-crop-fix — genuinely new, and mixed
Only TB had been re-tested after the crop fix. Now measured: **Normal 7/10 → 10/10** and
**Pneumonia 9/10 → 10/10** (every pre-fix error in those rows was a flip to COVID-19 — the attractor
the fix removed). But **COVID-19 went 7 correct/1 rejected → 5 correct/3 rejected**: the fix moves
the tensor the OOD detector scores, and where §3.5 measured that shift as *helpful* on
`data/xray/test/`, on this different external source it went the other way. The 2 non-rejected errors
are COVID-19 → **Normal**, i.e. under-calling, the opposite of the pre-fix mode. **At n=10 that is 2
images — a signal to note, not a measured regression.** TB external unchanged at 6/6, as expected
(the colour gate runs before any crop).

### 3. CT single-image, live `/predict/ct` — unchanged, as it should be
Dataset, **everything on disk**: Benign **10/12 = 83.3%** (2 → Normal), Malignant **50/50 = 100%**,
Normal **41/41 = 100%**; overall 101/103 = 98.1%. External: Malignant **12/12**, Normal **0/9**
(7 → Malignant, 2 → Benign) — **bit-identical to the pre-existing external run**, and Benign external
is **n=0, reported as its own row** (no independent non-IQ-OTH/NCCD benign source has ever been
found; not filled, not estimated). Malignant/Normal were previously measured on 25-image samples and
are now measured on all 50 / all 41, still 100%. **The 0/9 external-Normal shortcut result — the
finding this whole investigation was built around — re-confirmed live and unchanged.**

### 4. CT volume pipeline through the LIVE endpoint — reproduces G.3 exactly, on the FULL test set
All 45 LIDC volumes on disk submitted as real multipart uploads to `POST /predict/ct/volume` and
polled to completion — **45/45 succeeded**, 29.2 min wall, mean **38.7 s/scan** (20.4–84.3 s), host
RSS peak 9.9 GB. Scored with `run_g3_validation.py`'s exact methodology (8 mm GT match, 15 mm FP
radius, det ≥ 0.5, G.3's own scan-level rule), with ground-truth RAS re-derived from the DICOM rather
than read from G.3's stored outcomes. **End-to-end 53/84 = 63.1%, Wilson [52.4%, 72.6%]. Scan-level
24/45 = 53.3%, Wilson [39.1%, 67.1%].** Detector recall 71/84 = 84.5%; @0.3 54/84 = 64.3%; 10 benign
scans over-called Malignant; unmatched ≥0.5 = 119 (69 B / 17 M / 33 N); Benign nodules 46/68, Malignant
7/16. **Every one of these equals G.3's figure.** Two exact cross-checks on top: **229/229 candidates
across all 45 patients bit-identical to the stored Phase E/E.1 detector pool** (max detector-score diff
0.00000, max `center_ras_mm` diff 0.0 mm, no missing/extra candidates) — extending Phase H's 16/4 check
to the entire set — and **84/84 per-nodule outcomes and predictions identical to G.3, 0 differing**.
The brief's anticipated aggregation-threshold discrepancy does not arise: the live report floor is 0.30
and the stored pool's lowest score is 0.301, so the returned set is exactly the pool G.3 scored.

### 5. Volume pipeline external row: UNAVAILABLE, deliberately
A repo-wide search returns **exactly 45** `LIDC-IDRI-*` volume directories (phaseD 10 + phaseE1 35) —
all of them the G.3 test set, all "already used". **No unused LIDC patient exists on disk, so this row
is unavailable.** No new volumes were downloaded: freshly-downloaded LIDC would still be the detector's
own training lineage, and re-labelling the G.3 set as external would double-count already-seen data,
which the brief explicitly warns against. Reported unavailable rather than manufactured.

### 6. Freshness
Every figure in `phaseI_final_numbers.md` was measured this phase — the 347 single-image calls, the 45
volume jobs, and the scoring. **The 63.1% is a fresh measurement that happens to equal the earlier one,
not the earlier one copied in.** G.3's stored results were used only afterwards as an independent
cross-check. Non-claims unchanged: no external validation of the volume pipeline exists, external
X-ray/CT rows are n≤12, and every open item from `xray_cropfix_applied.md` §4 (crop mismatch, N3, N4)
and `phaseH_deployment.md` §7 (no volume OOD, no body-region check, uncalibrated concern levels,
in-memory jobs) still stands.

---

## Update 2026-09-10 — Phase H: the CT-volume pipeline DEPLOYED into the live app — **new `POST /predict/ct/volume` + new frontend page, additive only. The deployed route reproduces Phase G.3 BIT-EXACTLY (16/16 candidates, max diff 0.00000). pytest still 26 passed; `/predict/ct` and `/predict/xray` unchanged and re-tested.**

Full write-up: `eval_run/phaseH_deployment.md`. **This is the first phase that writes to the real
application.** New: `backend/ct_volume_pipeline.py`, `backend/ct_volume_routes.py`,
`frontend/src/pages/CtVolumePage.jsx`. Additive edits: `backend/main.py` (+2 code lines: one
import, one `include_router`), `frontend/src/{api.js, App.jsx, pages/MainApp.jsx, App.css}`.
**`ct_best_model_phase35_v2.pt` untouched; `/predict/ct`, `/predict/xray`, `/chat`, `/auth/*`,
`/scans/*` unmodified. All five checkpoints + `phaseE/pipeline.py` re-hashed after the work — all
digests match. No new dependency. Backend run on an isolated scratch DB and stopped; zero
`python.exe` remain.** No prior phase's directory was written to (every `__pycache__` pre-dates
this session).

### 1. Gemini's G.2/G.3 work verified independently — ALL PASS
`v4_stack3` sha256 `5ee4d8ec9f8328c1…e546` matches `phaseG3_verdict.md` §7 on the **full 64-hex
digest**. `backend/main.py` (pre-edit) `780120553178c0a5` and `ct_best_model_phase35_v2.pt`
`93b3a979884edbc4` match the chain back to Phase D, mtime unchanged; `v1`/`v2_jitter`/
`v3_jitter_loosebenign`/`phaseE/pipeline.py` all match too. All five G.3 artifacts exist and
their **contents** were read against the prose — `channel_ablation.json` and `g3_combined.json`
reproduce every figure (0.8946/0.4370/0.6992, 53/84 @ [0.5242, 0.7263], 67/84, 24/45, 10
over-called) to 4 decimals.

### 2. The "3D UNet" in `phaseG3_verdict.md` is a WRITE-UP MISLABEL, not an integrity problem
`phaseG3/run_g3_validation.py` contains **zero** occurrences of `UNet`/`RetinaNet` and constructs
no detector at all — its `end_to_end_pipeline()` loads Phase E/E.1's stored
`scan_reports*.json` and iterates `cd["detector_score"]` / `cd["center_ras_mm"]`. Confirmed
numerically by G.3's own report: detected **71/84, identical to E.1**, unmatched pool **119
(2.64/scan), "Identical detector pool"** — only possible if no detector ran. **The detector behind
every D–G.3 number, and behind this deployment, is MONAI's 3D RetinaNet.** Gemini's artifact was
not edited; the correction lives in `phaseH_deployment.md` §1.4. Detector recall re-verified as
exactly **84.5% (71/84)**, not a paraphrase.

### 3. What was deployed
`POST /predict/ct/volume` (multipart `files`: one `.zip` **or** many loose `.dcm` — both tested),
`GET /predict/ct/volume/{job_id}` (poll), `GET /predict/ct/volume` (self-description). Async: 202
+ `job_id` in 0.5–1.1 s, in-memory dict + one daemon thread per job, `MAX_CONCURRENT=1`, the
`phaseE/app.py` shape with two additions (the queue actually blocks now, and the dict is bounded
by count/age). **Models load lazily on the first job, never at startup** — a backend that never
gets a volume upload pays nothing. Geometry is `phaseE/pipeline.py` imported, never
reimplemented; the classifier is `v4_stack3` fed **real 3-slice stacks** built by Phase G.3's own
`build_stack_from_volume`, with `phaseG1/stack_data.py`'s `Grayscale`-free transform chain.

### 4. The deployed route reproduces Phase G.3 bit-exactly
4 real volumes (LIDC-IDRI-0586 as a zip, 0059/0433/0008 as loose files) pushed through the live
HTTP endpoint from a separate process. On the **16 candidates** G.3 also recorded for those
patients: **16/16 identical detector score (max diff 0.00000), 16/16 identical predicted class,
16/16 identical class probabilities (max diff 0.00000).** It reproduces G.3's *errors* too
(0059's Malignant nodule graded Benign), which is what fidelity looks like. Latency **19.6–42.8 s,
mean 35.6 s** (Phase D predicted ~35 s); stage 2 still free at 0.1–0.2 s; GPU peak 0.94–0.97 GB
(higher than Phase E's 0.751 only because the five single-image models are also resident); host
RSS peak 4.3–7.9 GB. Polling observed transitioning `queued → running → succeeded` with
progress 0.05 → 0.15 → 0.75 → 1.0.

### 5. Aggregation redesigned — Phase E's "any Malignant wins" is gone
Report floor det ≥ 0.30 (candidate is returned); **decision floor det ≥ 0.50 AND top-class
prob ≥ 0.60** (candidate counts); `high` needs **≥2** counting Malignant candidates **or** one
that is strong on both stages (det ≥ 0.80, conf ≥ 0.90); otherwise `moderate` / `low` / `none`.
**Every candidate is always returned** with location, detector score, all three probabilities and
`counts_toward_summary` + `excluded_because`. Two failures stated rather than hidden: a
confidently-wrong spurious box (Phase E's det 0.715 / Malignant 0.984) still clears both floors,
and the confidence floor **cut both ways on real data** — 0059's true Malignant nodule was graded
Benign at 0.556 and dropped from the count (it stays visible in the list, which is the point).

### 6. Existing functionality unchanged — re-tested, not assumed
**pytest `26 passed`** — identical to the X-ray crop-fix phase's 26. Startup reproduction checks
all six pass unchanged (xray 0.9646, ct 0.9904, gate 1.0000, xray_sanity 0.9996, ct_sanity
1.0000, seg). **6/6 real images through `/predict/xray` and `/predict/ct` return 200 with the
correct class, a Grad-CAM heatmap and a Groq report**; a chest X-ray posted to `/predict/ct`
still 422s from the sanity gate. `/health` still lists exactly the same five models.
Six input-validation rejections confirmed over real HTTP (too few slices 400, non-DICOM 400,
two series 422, bad zip 400, no dcm 400, unknown job 404). `vite build` ✓ in 260 ms.

### 7. Honest gaps carried into the UI
63.1% [52.4%, 72.6%] end-to-end, 84.5% detector recall ceiling, 79.8% GT-centred ceiling, 53.3%
scan-level, ~30–35 s/scan — all rendered **on screen** in a banner above the upload control and a
limitations block under every result, plus in the API payload. **No guardrail parity with
`/predict/ct`**: no volume OOD detector, no scan-sanity model, no body-region check (a head or
abdomen CT passes every gate and is analysed as a chest CT), `Modality` tag trusted — all four
returned in `validation.deferred`. Volume results are **not** written to scan history (the table
needs one class + one heatmap; faking one would undo §5's whole argument). Jobs die with the
process. Non-claims unchanged: still LIDC, still the detector's own lineage, no generalization
claim; this phase measures **deployment fidelity**, not model quality, and 4 volumes is a smoke
test, not a re-validation. Concurrency was implemented but never observed under real contention.

---

## Update 2026-09-10 (FINAL) — Phase G.3: 3D-aware CT classifier VALIDATED & FINAL VERDICT — **Through-plane accuracy at 2mm raised from 48.8% to 64.3% (+15.5%), GT-centred ceiling reaches 79.8% (+6.0%), benign false-alarms cut by 47%. End-to-end reaches 63.1% (53/84) — 70% target NOT reached. INVESTIGATION COMPLETE.**

Full write-up: `eval_run/phaseG3_verdict.md`. Artifacts: evaluation outputs in `eval_run/phaseG3/_work/`
(`test_accuracy.json`, `channel_ablation.json`, `stability_sweep_v4.json`, `e2e_v4_stack3.json`,
`g3_combined.json`). **`ct_best_model_phase35_v2.pt`, `v1`, `v2_jitter`, `v3_jitter_loosebenign`,
`v4_stack3`, and `backend/main.py` all re-hashed — all digests match exactly. No file outside
`eval_run/phaseG3/` and this file was modified. No dependency added, nothing deployed, no process
left running.**

### 1. Test accuracy on 389 stacks: acc 0.8946, bal 0.8480
Evaluated on test set with `min_real_slices=3`. Per-class recall: Benign 0.833, Malignant 0.750, Normal 0.961.
Zero Malignant nodules classified as Normal (0 false negatives to Normal).

### 2. Channel ablation: model genuinely uses 3D adjacent slice context
Ablation confirms the model is truly 3D-aware: zeroing neighbour channels collapses accuracy to 0.4370
(flips 55.8% of predictions); replacing neighbours with noise drops accuracy to 0.6992 (flips 30.3%).
Input gradient magnitude analysis confirms previous slice ($z-1$) has $0.81\times$ and next slice ($z+1$)
has $0.58\times$ the attribution of the centre slice ($z$).

### 3. Spatial stress test: through-plane stability at 2mm improved by +15.5%
On 1,176 re-extracted crops across 84 nodules:
- Through-plane only at 2.0 mm: `v4_stack3` achieves **64.3% accuracy and 29.8% flip rate**, vs `v2_jitter`'s
  **48.8% accuracy and 40.5% flip rate** (+15.5 percentage points accuracy, −10.7 points flip rate).
- Through-plane only at 3.0 mm: **48.8% vs 39.9%** (+8.9 points).
- Physical cross-section limits dominate at $\ge 5$ mm (small lesions leave the 3-slice stack).

### 4. End-to-end pipeline (84 nodules / 45 patients): 53/84 (63.1%)
- Ground-truth centred ceiling: **67/84 (79.8%, Wilson [0.700, 0.870])** vs Phase E.1's 62/84 (73.8%).
- End-to-end at `score >= 0.5`: **53/84 (63.1%, Wilson [0.524, 0.726])** vs Phase E.1's 51/84 (60.7%).
- End-to-end at `score >= 0.3`: **54/84 (64.3%, Wilson [0.536, 0.737])** vs Phase E.1's 52/84 (61.9%).
- Target of 70% end-to-end is **not reached** (falls short by 6.9 percentage points).

### 5. False-positive suppression & scan-level results
- Scan-level diagnosis: **24/45 (53.3%)** vs 23/45 (51.1%).
- Benign patients over-called Malignant: **dropped from 19 down to 10 (−47.4%)**, driven by adjacent-slice
  context enabling the classifier to correctly reject vascular structures extending across slices.

### 6. Investigation conclusion: Phase A–G COMPLETE
All empirical measurements across Phases A–G are recorded. The 70% threshold cannot be crossed
with post-hoc cascaded patches due to the joint limits of detector recall (84.5% ceiling) and 3D
mis-centring. The investigation is officially closed.

---

## Update 2026-09-10 — Phase G.2: 3D-aware CT classifier TRAINED — **val balanced accuracy 0.8637 (acc 0.8943) at epoch 13, outperforms v2_jitter by +3.5 bal / +2.5 acc. NO validation sweep, NO pipeline test — Phase G.3 is separate.**

Full write-up: `eval_run/phaseG2_training.md`. Artifacts: training outputs in `eval_run/phaseG1/`
(history JSON, curves PNG, valacc checkpoint); primary checkpoint
`checkpoints/ct_lidc_nodulecrop_v4_stack3.pt` (28.5 MB, NEW file). **`ct_best_model_phase35_v2.pt`,
`v1`, `v2_jitter`, `v3_jitter_loosebenign`, `backend/main.py` all re-hashed — all five digests
match Phase G.1 §7 exactly, both recorded mtimes unchanged. No file outside `eval_run/phaseG1/`,
`eval_run/phaseG2/`, and the new checkpoint was modified. No dependency added, nothing deployed, no
process left running.**

The exact command from Phase G.1 §6 was run with zero modifications:
`python -u train_stacks.py --min_real_slices 3 --tag v4_stack3` from `eval_run/phaseG1/`.
Batch size 8, AMP, 15 epochs (2 warmup frozen + 13 fine-tune), seed 42 — identical to Phase C.1's
recipe except the dataset is `LidcStackDataset` (3 adjacent slices as 3 channels, no `Grayscale`
duplication) and in-plane jitter is dropped.

### 1. Training completed in 8.5 min, peak VRAM 0.62 GB — within Phase G.1's 8–9 min estimate

No OOM, no batch size reduction, no retry. RTX 3050 4 GB with AMP. Channel guard passed on 64
sampled 3-slice stacks before the first epoch. Dataset: 2,058 train / 407 val (the
`min_real_slices=3` subset from Phase G.1 §4.3, which removes the class-correlated padding
shortcut). Class weights: Benign=1.3398, Malignant=1.7590, Normal=0.5934.

### 2. Best checkpoint: epoch 13, val balanced accuracy 0.8637, accuracy 0.8943

Per-class recalls at the best epoch: **Benign 0.856, Malignant 0.795, Normal 0.941** — the most
balanced profile across the run, with B–M gap only 6.1 points. Training converged cleanly: train
loss 1.04 → 0.14, train balanced accuracy 0.47 → 0.94. Val loss bottomed at epoch 6 (0.399) then
rose to 0.522 by epoch 15 while val accuracy stayed flat at ~0.89 — early overfitting in
confidence, not in accuracy.

Best-by-accuracy checkpoint: epoch 6, val_acc 0.8968, val_bal 0.8518 — saved separately as
`phaseG1/ct_lidc_nodulecrop_v4_stack3_valacc.pt`.

### 3. Comparison with prior checkpoints (val set only — not an end-to-end claim)

| checkpoint | input | val_acc | val_bal |
|---|---|---|---|
| v1 (Phase C) | single slice ×3 | 0.9030 | 0.8591 |
| v2_jitter (Phase C.1) | single slice ×3 + jitter | 0.8693 | 0.8292 |
| **v4_stack3 (this phase)** | **3 adjacent slices** | **0.8943** | **0.8637** |

v4_stack3 exceeds v2_jitter on both metrics and is within ~1 point of v1 on accuracy while
exceeding it on balanced accuracy. **However, this comparison is NOT straightforward**: v4_stack3
trains on 2,058 rows (min_real_slices=3) vs 2,450 for v1/v2; v2 has in-plane jitter that v4 does
not; and the val sets differ in size (407 vs 506). **No through-plane tolerance, in-plane tolerance,
or end-to-end pipeline number exists for v4_stack3 — those are Phase G.3's measurements and are
the ones that matter.**

### 4. Deviations from Phase G.1's specification: NONE

Zero parameters changed. Zero flags added or removed. The command, working directory, and all
outputs matched Phase G.1 §6 exactly.

### 5. Non-claims

No tolerance sweep, no Grad-CAM, no pipeline test. The val-set numbers compare annotation-centred
crops, not displaced or detector-centred crops. Phase E.1's through-plane defect is unmeasured for
this checkpoint. Phase F's verdict (c) and Phase E.1's "write it up and stop" are NOT overturned
by a training run — they require the measurements Phase G.3 would produce. Still LIDC, still the
detector's lineage, no generalization claim. Phase B's unresolved caveats carry over verbatim.

---

## Update 2026-09-10 — Phase G.1: multi-slice STACK extraction + architecture setup — **3,461/3,461 stacks built, centre channel bit-identical to Phase B, ZERO architecture change needed. NO TRAINING — Phase G.2 is a separate session.**

Full write-up: `eval_run/phaseG1_setup.md`. Artifacts: `eval_run/phaseG1/` (1.8 GB, of which 1.7 GB
is cached DICOM and 66 MB the stacks; 11 Python files + JSON). **Extraction + architecture code +
a forward-pass-only smoke test. No optimizer, no backward pass, no checkpoint written.
`labels_manifest.csv` NOT modified (still sha 95d734f64123a46f, 3,461 rows), `backend/main.py`,
`ct_best_model_phase35_v2.pt`, `v1`, `v2_jitter`, `v3_jitter_loosebenign` untouched — re-hashed,
all seven prior digests match, both recorded mtimes unchanged. No dependency added, nothing
deployed, no process left running. Unlike Phase F, no prior phase's artifact was touched at all.**

Rationale: Phase F closed post-hoc ensembling as (c), but left one datum pointing at a genuinely
3D-aware model — the ±1 slice is not blind (0.601 accurate, agrees with the true-centre-slice
prediction 72.6%), and a model taking the stack as ACTUAL input could learn to weight that, which an
average structurally cannot.

### 1. Slice adjacency: the two cheap routes are DEAD, the third is validated 189/189

**NBIA's `getSOPInstanceUIDs` returns UIDs in effectively random order** — measured on 6 cached full
series, the fraction of API-adjacent pairs that are geometrically adjacent is **0.000–0.023**. And
there is **no per-instance metadata endpoint**: v1/v2 `getDicomTags` returns HTTP 500 for every
parameter spelling; `getSeriesMetaData` is series-level only. *Do not retry either.*

The route that works costs **0 extra API calls**: every LIDC annotation `<roi>` carries BOTH
`imageSOP_UID` and `imageZposition`, and Phase A already downloaded all 1,318 XMLs. Union them into
a per-series (sop → z) grid, take the modal positive gap as the slice pitch, name the neighbours at
z0 ± pitch. **Validated against the 45 series Phase D/E.1 downloaded in full: 189 of 189 proposed
neighbours are the exact true ±1 slice, 0 wrong.** Cross-checked a second way on all 1,989 centre
slices against each DICOM's own spacing tags — 664 cases where the XML pitch is SMALLER than
`SliceThickness` are LIDC's overlapping reconstructions (2.0 mm thick every 1.0 mm, no
`SpacingBetweenSlices` tag in 200/200 sampled), and 62 of the 107 validated slices are in exactly
that bucket with exact neighbours. Only **3 series** had a pitch LARGER than the DICOM bound
(support 1–2 gaps); a guard now rejects those and they fall back to padding.

### 2. Extraction: 3,461/3,461, under the ~2× Phase B budget, no whole-volume download

3,461 crops sit on only **1,989 unique centre slices**, so **3,386 new single slices** were fetched
(140 neighbours were already Phase B's own centre slices, reused in place): **3,386 ok / 0 failed /
0 retries, 1,782 MB, 29.7 min at 1.90 slices/s (8 workers)** vs Phase B's 1,999 in 21.1 min at
1.54/s — **1.69× the slices in 1.41× the time.** Extraction 31 s.

Stacks are **3-channel 128×128 PNGs, `[centre−1, centre, centre+1]`, one slice per channel**, using
Phase B's own `box_for`/`to_hu`/`window`/resize chain by import with the integer box held fixed and
only z varying. **The centre channel is bit-identical to Phase B's stored PNG for all 3,461 rows,
max abs difference 0 — checked per row, not sampled.** 0 rows skipped, 0 DICOM failures, 0 shape
mismatches, 3,461 unique paths. Counts reproduce Phase B row for row (562 M / 913 B / 1,986 N; 797
patients; train 2,450 / val 506 / test 505) and **splits were copied through, never reshuffled** —
0 rows disagree with Phase B on label/split/patient, patient-disjointness re-verified.

**The edge case is NOT the one anticipated.** Across the 45 full series, **zero** crops sit on a
series' first/last slice (chest-CT end slices are above/below the lungs). The real cause is
annotation coverage: **646 of 6,922 sides (9.3%)** have no contoured neighbour. Fallback = **repeat
the CENTRE slice** (not the opposite neighbour, which would invert channel ordering); this degrades
a row to exactly Phase B's single-slice input. **82.5% of stacks have 3 real slices, 16.4% two,
1.1% one.**

**Flagged honestly: the gap is class-correlated — Malignant 93.8% complete vs Benign 77.2%, Normal
81.7% (16.6-point M–B gap)**, because big malignant nodules get contoured on more slices. Since the
fallback duplicates a channel, "are my channels identical" weakly predicts class — a shortcut risk
in the one investigation that exists because of shortcuts. **G.2 should use `--min_real_slices 3`
(2,058/407/389, patient-disjointness re-verified); all-rows is a one-flag ablation.**

### 3. Architecture: NO change needed — checked, not assumed. Smoke test PASSES.

`ml/model.py`'s densenet121 `features.conv0` is `Conv2d(3,64,k=7,s=2)`, **weight `[64,3,7,7]`**, and
**`v2_jitter`'s own trained checkpoint carries the identical `[64,3,7,7]`**. 3 channels in, 3
expected → `architecture_change_needed: false`. Nothing added, no weight reshaped; pretrained
ImageNet conv0 weights kept as-is, which is correct when only channel *semantics* change.

**The change that IS needed is a one-line deletion in the DATA pipeline, and it is a trap.**
`ml/dataset.py::build_transforms` starts with `Grayscale(num_output_channels=3)` — on a real stack
that collapses three different slices to one luminance channel and re-duplicates it, **yielding a
model that trains, converges, reports a number and never saw more than one slice.**
`phaseG1/stack_data.py` therefore does not import it; it reproduces Phase C's chain verbatim minus
that op, plus `assert_channels_differ()` which raises if any stack's channels are identical
(64/64 val stacks passed). The three channels genuinely are a different input at layer 1: pretrained
conv0's per-channel filters are **not** identical (cosine 0.648 / 0.362 / 0.603) and a real stack
vs the same centre duplicated 3× moves the logits by **0.690**.

**Smoke test: `[8,3,224,224]` → `[8,3]`, no NaN, no Inf, softmax rows sum to 1, peak VRAM 0.194 GB
of 4.00 at bs=16. Forward only — no backward pass, no optimizer, no checkpoint.**

### 4. Phase G.2 — exact resume command, already written and NOT run

```
cd "D:/7th sem/sdp/eval_run/phaseG1"
python -u train_stacks.py --min_real_slices 3 --tag v4_stack3 > _work/train_v4_stack3.log 2>&1
```

`train_stacks.py` = `phaseC1/train_lidc_crops_jitter.py` with exactly two changes (dataset is
`LidcStackDataset`; in-plane jitter dropped, since C.1's jitter source is one oversized slice with no
z extent). Everything else identical to C.1 so the number is comparable to `v2_jitter`. Produces
`checkpoints/ct_lidc_nodulecrop_v4_stack3.pt` (+ `_valacc.pt`, `training_history_v4_stack3.json`,
`training_curves_v4_stack3.png`); it **hard-refuses** to write any of the four protected checkpoints.
**Estimate ~8–9 min at `--min_real_slices 3`** (~9.5–11 min all-rows) against Phase C's 9.4 min /
C.1's 9.6 min on this RTX 3050 4 GB. **VRAM and batch size do NOT change** — the tensor is
`[8,3,224,224]` either way; Phase C fed one slice duplicated to 3 channels, G.2 feeds 3 distinct
slices into the same 3 channels. Do not reduce batch size; it would break comparability for nothing.
**A G.2 that only trains is not an answer** — Phase E.1 §2 put the whole failure on the through-plane
axis, so the run must be followed by an in-plane control AND a through-plane sweep (Phase F §3.3
showed gaining one axis while losing the other is a live risk). Budget that separately.

### 5. Non-claims

No model trained, no accuracy number exists; **Phase F's (c) and Phase E.1's "write it up and stop"
are untouched.** N=3 only (±2 would need the §2 route re-validated, and XML coverage worsens with
distance). Still LIDC, still the detector's lineage, no generalization claim. 9.3% of channel sides
are the centre repeated and that padding is class-correlated. Phase B's unresolved caveats carry over
verbatim — class-correlated lung-fraction rejection, radiologist-impression labels not biopsy, 23.9%
of clusters unlabelled. The 189/189 validation covers ~5.6% of series (the independent all-1,989
spacing check agrees).

---

## Update 2026-09-10 — Phase F: multi-slice ensembling at inference time — **it does NOT work. Best variant TIES the baseline at 51/84 (60.7%); the mechanism check falsifies the hypothesis outright.**

Full write-up: `eval_run/phaseF_multislice_ensemble.md`. Artifacts: `eval_run/phaseF/` (1.1 MB, 5
scripts + JSON). **No training, no new model, no download. `eval_run/phaseE/pipeline.py` was IMPORTED,
never edited (mtime still Phase E's 2026-09-09 15:09, digest now on record). `backend/main.py`,
`/predict/ct` and `ct_best_model_phase35_v2.pt` untouched — re-hashed, all five Phase E §7 digests
match, both mtimes unchanged. No checkpoint written, no dependency added, nothing deployed, no process
left running.**

The user chose to continue past Phase E.1's "close it here" for this one bounded check: since
`v2_jitter` is fine in-plane and only breaks through-plane, and it only ever sees ONE slice, does
ensembling it over N adjacent slices at inference time recover the true centre slice's answer?

### 1. Same 84 nodules / 45 patients, detection unchanged (71/84). No strategy beats the baseline.

| variant @0.5 | end-to-end | Wilson 95% | ceiling (GT-centred) | scan-level | Benign scans over-called |
|---|---|---|---|---|---|
| **N=1 single slice (baseline)** | **51/84 (0.607)** | **[0.500, 0.705]** | **62/84 (0.738)** | 23/45 | 19 |
| N=3 mean prob | 49/84 (0.583) | [0.477, 0.683] | 61/84 | **24/45** | **17** |
| **N=3 majority vote** (best) | **51/84 (0.607)** | **[0.500, 0.705]** | 60/84 | **24/45** | 18 |
| N=5 mean / vote | 36 / 37 of 84 | [0.328, 0.535] | 45/84 (0.536) | 23 / 22 of 45 | 17 |
| N=3 / N=5 max-confidence | 31 / 28 of 84 | — | 50 / 31 of 84 | 22 / 20 of 45 | 18 / 15 |

**The best result is a tie, and even that tie is 5 of 71 predictions swapped 2-for-2 — the flips
cancel, nothing stabilises.** Paired net change: N=3 mean −2, N=3 vote 0, N=5 mean **−15**, maxconf
−20/−23. Scan-level +1 scan is inside a 28-point interval and comes from candidates drifting to
`Normal`, the same drift that loses nodules. **~70% is not approached by anything.**

**Harness validated before use:** the N=1 baseline is recomputed here (this GPU's probabilities move
~1e-3 with batch size) and lands on **51/84 and 23/45 with ZERO per-nodule differences** from Phase
E.1's published numbers. Detector not re-run — its 229 stored candidates reused, verified 229/229 same
slice, 227/229 same pixel box, 229/229 same predicted class.

### 2. The mechanism check — the hypothesis is falsified, not merely unsupported

Through-plane sweep re-run with ensembling (168 crops/offset): **2 mm 0.488 → 0.482 (N=3) / 0.411
(N=5); 5 mm 0.226 → 0.226; 8 mm 0.101 → 0.083.** It does not help even at the smallest displacement,
so this is not "only patches small errors".

**The conditional test is the kill shot.** At 2 mm the true centre slice IS inside the N=3 window 48.8%
of the time — and on exactly those 82 crops the ensemble scores **0.549 against the single slice's
0.549, an identity**. At N=5 the true slice is present 73.8% of the time and accuracy **falls 0.492 →
0.395**. **The correct vote is present and gets outvoted.**

**Why, measured:** neighbour slices are not noisy-but-unbiased, they are worse and `Normal`-biased —
per-slice accuracy **0.738 (true) / 0.601 (±1) / 0.375 (±2)** with `Normal` rates **10.7% / 24.4% /
52.4%**. Averaging a 0.74 view with 0.60 and 0.38 views does what arithmetic says. Consequences:
`called_Normal` 11 → 27 nodules at N=5; **N=5 breaks 19 confidently-correct GT-centred answers (10 at
≥0.90 conf) and fixes 2**; and the **in-plane axis that actually worked is damaged** (2 mm: 0.744 →
0.720 at N=3, **0.509 at N=5**). Thin-slice stratum, where it should have worked best: **+1 nodule of
43**. Max-confidence is worst because the most confident slice is usually an off-centre one
confidently reporting `Normal`.

### 3. Cost is genuinely free — the one clean positive

5× forward passes measured on the real pipeline: stage 2 **0.129 s/scan at N=5** (10–20 crops), total
latency **33.1 s** vs baseline 33.2 s, **peak VRAM 0.751 GB — bit-identical to Phase E/E.1**, and
per-scan host RSS with N=5 in flight differs from single-slice by **≤0.01 GB** (a 28 mm crop is 40×40;
the volume is GBs). Ensembling never gets near the RAM ceiling. It is free, and it buys nothing.

### 4. Verdict: (c) it doesn't help. Phase G NOT started — the decision is the user's.

Reported as (c), not (b): the evidence does not say "helps but falls short". **The cheap version of the
idea is now closed — the through-plane defect cannot be patched at inference time; anything that
addresses it has to change the model.** One piece of evidence cuts toward a hypothetical 3D-aware
Phase G (the ±1 slice is not blind — 0.601 accurate, agrees with the true slice 72.6% — so a model
taking a stack as its ACTUAL input could learn to weight it, which a post-hoc average structurally
cannot), and everything in Phase E.1 §7 cuts against it (retrain + re-validate all three axes +
plausible ~3 points off a 74% ceiling that is already the binding constraint + aggregation redesign +
the guardrail chain that still doesn't exist, against ~10 days and an exam). **Phase E.1's "write up
the CT investigation and stop" is untouched by this phase and is now supported by one more measured
negative.** Non-claims unchanged: still LIDC, still the detector's lineage, no generalization claim;
N ∈ {3,5} with three fixed rules (a weighted average untested, though the 0.38/0.60/0.74/0.60/0.38
profile says its optimum is "all weight on the centre" = the baseline); latency probe is 5 scans.
**Transparency: `phaseE1/integrity.py` was run once at the start to establish baseline digests, so
`phaseE1/_work/integrity.json` has a new mtime with content-identical output; no E.1 measurement is
affected. Phase F has its own `integrity.py` writing to `phaseF/_work/`.**

---

## Update 2026-09-09 — Phase E.1: bigger n + per-crop stability — **51/84 (60.7%) confirmed, the churn is REAL and general, root cause found: `v2_jitter` has ~2 mm THROUGH-PLANE tolerance. CLOSE THE INVESTIGATION.**

Full write-up: `eval_run/phaseE1_verification.md`. Artifacts: `eval_run/phaseE1/` (8 scripts + JSON;
4.9 GB is cached DICOM). **Measurement only — nothing was built. `phaseE/run_eval.py` was IMPORTED and
its `main()` called with two path constants repointed, so the harness is literally the code that
produced 13/22. `backend/main.py`, `/predict/ct` and `ct_best_model_phase35_v2.pt` untouched —
re-hashed three times, all five digests match Phase E §7. No checkpoint written, no dependency added,
nothing deployed, no process left running.**

The two measurements Phase E named as blockers. Both are now answered, and they agree.

### 1. Bigger test set: 22 → 84 nodules / 10 → 45 patients. The 59.1% was NOT bad luck.

35 new LIDC test-split volumes downloaded (9,791 slices, 4.9 GB, ~2 h 5 min via Phase D's per-slice
route; the zip route was never retried). Same pipeline, same harness, same 8 mm/15 mm radii.

| | Phase E (n=22) | **Phase E.1 pooled (n=84)** |
|---|---|---|
| detected ≤8 mm | 20/22 (0.909) | **71/84 (0.845)** |
| **END-TO-END CORRECT** | **13/22 (0.591)** | **51/84 (0.607)** |
| **Wilson 95% CI** | [0.387, 0.767] — width **0.38** | **[0.500, 0.705]** — width **0.20** |
| control: GT-centred (perfect-detector ceiling) | 16/22 (0.727) | **62/84 (0.738)** |
| at score ≥ 0.3 | 13/22 (0.591) | 52/84 (0.619) |

**Point estimate moved 1.6 points, interval halved.** Clopper-Pearson and a patient-clustered
bootstrap agree at n=84 (they did not at n=22). Phase E's half reproduces exactly (13/22, [0.387,
0.767]). Error budget over 84: **22 lost to classifier error on a perfect crop**, 7 to detector
misses, net 4 to centring. **The classifier is still the bottleneck by ~3× and its ceiling is stable
at ~0.74 across 4× the data.** Crop path re-verified: **84/84 bit-exact vs Phase B's PNGs**, including
62 nodules from series never opened before.

**Malignant is no longer n=4.** Step 1's nodules-per-download ranking accidentally selected 50 Benign
and zero Malignant (every multi-nodule test series is all-Benign), so 12 Malignant-carrying series
were added deliberately: **16 Malignant, 13 detected, 9 correct = 0.563 [0.332, 0.769]** vs Benign
0.618. Indistinguishable, both ~0.6. Still not a clinical claim, but no longer a refusal.

### 2. The churn is REAL and GENERAL — and the root cause is a third axis nobody tested

**15 of 71 matched predictions (21.1%, Wilson [0.132, 0.320]) change class** vs Phase E's 4/20 (20.0%)
— **the rate replicates almost exactly at 3.5× the sample, and now spans 13 of 45 patients instead of
2.** LIDC-0321 is the worst single scan but sits at **z=1.68, 91st percentile** at 2 mm and only 60th
by 5 mm. **"One bad patient" is falsified.**

Per-crop sweep, 84 nodules × 4 offsets × 14 directions = 1,176 crops, run for `v2_jitter` AND `v1`:

| `v2_jitter` (0 mm acc 0.738) | 2 mm | 3 mm | 5 mm | 8 mm |
|---|---|---|---|---|
| **in-plane only** | 3.9% flip / **0.744** | 4.8% / 0.738 | 6.6% / **0.735** | 8.3% / 0.723 |
| **through-plane only** | **40.5%** / 0.488 | **51.2%** / 0.399 | **69.6%** / **0.226** | **82.1%** / 0.101 |

**Phase C.1's tolerance curve and the jitter augmentation that produced it are BOTH in-plane only** —
the curve's displacements are `SHIFT_MM × √2`, and `prep_jitter_source.py` cuts its oversized window
from the *same DICOM slice*. So the celebrated "~13 mm tolerance" is a **2-of-3-axis measurement**. On
the third axis `v2` is barely better than un-augmented `v1` (through-plane @5 mm: v1 0.137, v2 0.226 —
against in-plane v1 0.292, **v2 0.735**). **A 3D detector mis-centres onto a different SLICE, which is
exactly the axis with ~2 mm of tolerance.** That is Phase E's compounding surprise, explained.

Not merely "the lesion left the slice": for nodules **≥9 mm**, where 2 mm of z is well inside the
lesion, the flip rate is still **21% @2 mm / 31% @3 mm**. Method validated by reproducing Phase C.1 on
`v1` in-plane (0.292 @5 mm vs their 0.338 @4.2 / 0.121 @8.5) and `v2` in-plane (flat, as they found).

### 3. Scan-level got WORSE with more scans: 23/45 (51.1%), 19 Benign scans over-called Malignant

2.83 unmatched candidates/scan, **27 of them graded Malignant**. Phase E called the max-driven
aggregation rule "visibly wrong as specified" at 5/10; at 45 scans it is the dominant failure mode.
Resources confirm Phase E: VRAM unchanged at **0.751 GB**, stage 2 still free (0.0 s), latency 35.7 s
mean — but **peak host RAM rose 8.23 → 10.23 GB of 15.26 (67%)**.

### 4. Verdict: CLOSE IT. Do not build Phase F.

A change from Phase E's "decision is the user's" — the door is now measured. **~61% ±10 is real; the
classifier's 74% ceiling is the binding constraint and a perfect detector buys only 13 points; the
churn is structural, not integration-level.** No FP-aggregation or guardrail work touches any of that.
**The honest counter-argument, recorded rather than buried:** the through-plane lock is the same defect
Phase C.1 fixed in ~40 lines and 9.6 min, so jittering the slice axis is a known-shape cheap fix and
someone with time should do it — but it would plausibly cost ~3 points of centred accuracy off a
ceiling that is *already* the constraint, landing ~70-75% grading. **It would make the compounding
honest; it would not make the product good.** Plus it now *adds* scope (retrain + re-validate all three
axes + aggregation redesign + the guardrail chain that still doesn't exist), against ~10 days and an
exam. **Recommendation: write up the CT investigation and stop.** The arc is complete and good:
shortcut found → removed structurally → real lesion features shown → position-lock diagnosed and fixed
on two axes → real detector measured in mm → stages wired and compounding measured → **compounding
surprise chased to its root cause, which was a third axis nobody had tested.**

Explicit non-claims: still LIDC, still the detector's own lineage — bigger n makes the *compounding*
solid, **not generalization**; 16 Malignant spans a 44-point interval; match radii are Phase D's,
reused not re-derived; the sweep uses 14 fixed directions (the in-plane/through-plane split is
load-bearing, the all-directions number depends on the 4:2:8 mix). Phase A's deployment no-go and the
advice not to cite the live model's external-Malignant 12/12 as generalization evidence are untouched.
**One transparency note: a harness interruption mid-phase left partial result files; they were verified
stale and fully recomputed, and the interrupted volume re-fetched, so every number is from one clean
run over all 45 volumes.**

---

## Update 2026-09-09 — Phase E: detector + classifier wired together and run end to end — **13/22 compounded (59.1%), concept proven, NOT ready to integrate**

Full write-up: `eval_run/phaseE_pipeline_build.md`. Artifacts: `eval_run/phaseE/` (252 KB, 4 files).
**Standalone/parallel artifact only. `backend/main.py`, `/predict/ct` and
`ct_best_model_phase35_v2.pt` are untouched — re-hashed, all five digests match Phase D §9 exactly.
Nothing deployed, nothing integrated, no new dependency, no checkpoint written.** The integration
decision is deferred to the user, per the brief.

First time in this investigation that both stages ran together.

### 1. The compounded number — and it is NOT the product of the two stage numbers

MONAI detector → 28 mm crop → `ct_lidc_nodulecrop_v2_jitter.pt`, on Phase D's own 10 volumes /
22 nodules. Score ≥ 0.5 default:

| | measured |
|---|---|
| detected (≤8 mm, Phase D's matching logic) | **20/22 (0.909)** — reproduces Phase D exactly |
| **END-TO-END CORRECT (detected AND graded right)** | **13/22 (0.591)** |
| control: same crops, GT-centred (perfect-detector ceiling) | **16/22 (0.727)** |
| at score ≥ 0.3 | detected 21/22, **end-to-end still 13/22** |

**Naively multiplying 0.909 × 0.792 gives 0.72 — the real answer is 0.591.** Compounding did not
multiply cleanly, which is exactly why the brief demanded it be measured.

**Error budget over 22: 6 lost to classifier error on a *perfect* annotation-centred crop, 1 to a
detector miss, net 2 to the detector's crop centring.** The classifier is the bottleneck by 2×, not
the detector — which overturns Phase D's closing read that "the bottleneck is recall, not centring".
It is neither: it is stage-2 grading accuracy on crops it is already centred on.

### 2. The new finding: 20% prediction CHURN at 2.75 mm, where the tolerance curve says ~zero cost

**4 of 20 matched predictions change class between the annotation-centred and detector-centred crop**,
at mean displacement 2.75 mm (max 5.39) — inside the flat part of Phase C.1's curve (0.792 @ 0 mm →
0.797 @ 4.2 mm). Net −2, but 4 moved. **Phase C.1's curve measured aggregate accuracy vs
displacement, not per-crop stability, and a flat curve can hide churn that cancels.** Three of the
four flips are the entire nodule complement of ONE scan (LIDC-0321) at 2.35–2.60 mm, all going
confident-correct → confident-wrong (Benign 0.93–0.98 → Malignant 0.589 / Normal 0.87–0.97).
**With n=22 and 3 of 4 flips in one patient this cannot distinguish "real 20% churn" from "one odd
scan" — that is the single most important follow-up measurement.**

### 3. False-positive handling is the weakest part, and it is a UX problem

2.00 unmatched candidates/scan @0.5 (reproduces Phase D), classified **12 Benign / 3 Malignant /
5 Normal** — **the Normal class absorbs only 25%**, against Phase D's hope that stage 2 would absorb
them. **Scan-level accuracy 5/10, and 2 of the 3 Malignant over-calls are driven by an UNMATCHED
candidate** — including Phase D's single genuinely-spurious out-of-lung box (LIDC-0321, det 0.715,
17.6 mm), which this classifier grades **Malignant at 0.984**. That scan's truth is three Benign
nodules; the pipeline emits ten findings, three graded Malignant, overall call Malignant.
**The brief's "max-malignancy drives the call" rule is visibly wrong as specified** — reporting all
candidates is right and should stay, but one spurious box must not flip a scan. Needs a real design
decision (probability floor / in-lung gate / ranked list with no binary call) before any UI.
Clean scans do look sensible (LIDC-0586: 3/3 found and graded correctly, the one FP absorbed as
Normal).

### 4. Wiring proven correct before any accuracy claim

New geometry this phase is the **inverse** of `volume_io.annotation_to_ras` (~20 lines), mapping a
detector box centre in RAS mm back to a slice/row/col in the **ORIGINAL** series (Phase D §7's
warning: crops must not come from the resampled volume). Verified, not trusted: **22/22 correct
slice, row/col identical to 0.1 px, crop pixels bit-exact (max abs diff 0) vs Phase B's stored
PNGs**, and the resulting predictions **identical on 22/22 to Phase C.1's stored `eval_test_v2.json`**.
So the 59.1% is a real property of the pipeline, not a subtly-different-crop artifact.

### 5. Resources with BOTH models resident — GPU fine, host RAM is the real constraint

| | |
|---|---|
| both models loaded, idle | **0.104 GB VRAM** (detector alone 0.078; classifier adds 26 MB) |
| **peak during a scan, both resident** | **0.751 GB of 4.00** vs Phase D's detector-only 0.69 — **+9%, no contention, unvarying across all 10** |
| **peak host working set** | **8.23 GB of this machine's 15.26 GB (54%)** |

**Host RAM was never measured before and it is ~11× the VRAM.** Cause is structural, not a leak: the
sliding-window inferer's output buffer sits on the CPU precisely to keep VRAM at 0.69 GB — **Phase D's
VRAM win is paid for in host RAM, and that trade had not been priced.** `MAX_CONCURRENT = 1` in the
app is the correct setting here, not a placeholder.

### 6. Latency and the async shape — Phase D's prediction held

**33.4 s mean/scan (18.5–45.4)**: read+resample 8.1 s, detect 25.1 s, **classify 0.2 s (0.6% of wall
time — stage 2 is free)**. Phase D predicted "20–50 s, ~35 s typical". Job/polling implemented in
`app.py` as an in-memory dict + one daemon thread per job (deliberately NOT Celery/Redis, per
`PROJECT_STATUS.md` §8 local-only scope): **POST returns 202 in 3 ms**, GET reports live
stage/progress, bad input 400s synchronously, both models loaded once behind a lock. Costs written
down rather than glossed: jobs die with the process, no retry, dict grows until restart.

### 7. Verdict: proven, not ready — and the decision is the user's

**Do not integrate yet — not because it failed, but because 59.1% on 22 nodules with an unresolved
false-positive UX is a result to investigate, not a feature to ship.** Blockers, in order:
22 nodules / 10 scans cannot carry the headline (95% CI on 13/22 is ~0.39–0.78); the §2 churn
question is open; the §3 aggregation rule needs a design answer; **the classifier — not the detector —
is the 0.727 ceiling**, so integrating means shipping `v2_jitter`'s grading behind a better front end;
Phase D §6's guardrail chain (series-level sanity/modality/OOD gates) **was not built and is absent**;
host RAM needs a bound. Cheapest two things that would move the decision: a **per-crop stability
sweep** (same nodule, many small displacements, count class changes not mean accuracy) and a bigger
test set.

Explicit non-claims: still LIDC, still the detector's own training lineage — this measures wiring and
compounding, **not generalization**; 18 B / 4 M means **n=4 is not a Malignant accuracy measurement**
(4/4 detected, 2/4 graded right); match radii are Phase D's, reused for comparability, not
re-derived. Phase A's deployment no-go and the advice not to cite the live model's external-Malignant
12/12 as generalization evidence are both untouched.

---

## Update 2026-09-09 — Phase D (measured): MONAI detector actually run on real LIDC volumes — **localization PASSES the ~13 mm target with a 4.7× margin**

Full write-up: `eval_run/phaseD_detector_test.md`. Artifacts: `eval_run/phaseD/`.
**No checkpoint was written; `ct_best_model_phase35_v2.pt`, `backend/main.py`, `data/ct_scan/` and
Phase B's `labels_manifest.csv` are untouched (re-hashed, first two match Phase C.1 exactly).**
One environment change: `pip install nibabel`. **Phase E (building the pipeline) was NOT started.**

This is the phase the survey deferred: download the detector, run it, get real millimetres.

### 1. The headline — the AP@IoU=0.1 number converts to ~2.8 mm, not to "borderline"

MONAI `lung_nodule_ct_detection` (3D RetinaNet, LUNA16) run on **10 Phase B/C test-split LIDC series
(22 annotated nodules, 4 Malignant / 18 Benign, 4.0–18.6 mm; 3 vendors, slice spacing 0.5–2.5 mm)**,
3D Euclidean box-centre vs annotation-centre error, score ≥ 0.5:

| | measured |
|---|---|
| **detection rate ≤ 8 mm** | **20/22 (90.9%)** — 21/22 at score ≥ 0.3 |
| **mean / median 3D centre error (matched)** | **2.75 / 2.45 mm** |
| p90 / **worst matched** | 3.92 / **5.39 mm** |
| mean in-plane / z | 2.27 / 1.33 mm |
| **Phase C.1 tolerance it must beat** | **~13 mm** |

Against Phase C.1's `v2_jitter` tolerance curve (0.797 @ 4.2 mm, 0.783 @ 8.5 mm, 0.725 @ 12.7 mm,
cliff at 15.6 mm): **every matched detection lands in the flat part of the curve**, below the 8.5 mm
point. The margin is ~4.7×. Even the pre-jitter `v1`'s much tighter <4.2 mm tolerance is met on
**19 of the 20 matched** — so **relaxing localization further buys nothing; the bottleneck is
recall, not centring.** That inverts the priority Phase C/C.1 assumed.

**The error distribution is bimodal, not a tail** — the ≤5 mm, ≤8 mm, ≤13 mm and ≤20 mm detection
rates are *identical or near-identical at every score threshold tested*, and the worst matched error
never exceeds 5.39 mm. A nodule is either found and placed within ~5 mm or not found at all. The crop
handed to the classifier is right or absent, never subtly wrong. **The two failures:** one nodule is
found at **2.17 mm but scores 0.467**, just under the 0.5 cut (a threshold choice, not a
localization failure); the other sits in a **cluster of adjacent nodules** and the detector locked
onto a neighbour 20 mm away.

### 2. VRAM — fits easily, but the bundle's own config does NOT

| ROI | peak | time | note |
|---|---|---|---|
| **192×192×80** (native *training* patch) | **0.69 GB / 4.00** | 15–35 s | same top box |
| 512×512×192 (`configs/inference.json` **default**) | **9.85 GB** | 144 s | only completes via the Windows driver's silent **shared-system-memory spill** — 4.4× slower |
| 192×192×80 on CPU | — | 211 s vs 15 s | byte-identical output |

**Trap worth remembering: a naive `monai.bundle run` on this laptop appears to succeed while paging
through host RAM.** Phase E must pin `roi_size=[192,192,80]`. Also checked: inference is
**bit-reproducible** run-to-run here (max |score diff| 0.0, max |centre diff| 0.0 mm) despite the
bundle's documented non-determinism caveat.

### 3. False positives — 2.0/scan, and two-thirds of them are real nodules

40 detections / 10 scans at score ≥ 0.5 → **20 unmatched (2.00/scan)**. Characterised, not just
counted: **17/20 inside lung parenchyma** (Phase B's own `lung_mask`), **19/20 focally denser than
surroundings** (+40 to +680 HU contour-vs-annulus), and **13/20 within 4 mm of a LIDC reader ROI
that Phase B's manifest filtered out** — i.e. genuine annotated nodules. Unexplained: **0.70/scan**.
**Exactly one unmatched box across all 10 whole-chest scans is both outside lung tissue and
unexplained** (the other two not-in-lung boxes are juxtapleural nodules within 1.5 mm of a reader
ROI). It does not flag noise as nodules. Context for
the gap: these series carry **9–121 reader ROIs each** against 1–3 nodules per series in Phase B's
manifest, so unmatched ≠ wrong.

### 4. Coordinate conversion validated BEFORE trusting any mm number (Phase A's axis lesson)

* contour-vs-annulus HU at the annotation position: **+295.6 mean, 22/22 positive**; at the
  deliberately **row/col-swapped** position: +0.1, only 10/22.
* the 28 mm patch cut at the derived RAS position correlates **r = 0.882 mean (min 0.709, several
  ≥0.98, one 1.000)** with **Phase B's own stored crop PNG** for that nodule, against a null of 200
  random in-lung patches (p95 0.08–0.63). The winning orientation is `transpose + reverse both axes`
  on **20/22** — exactly what LPS→RAS predicts.

### 5. The 3D/2D integration read, now evidence-based

**The volume half is cheap: ~135 new lines** (`volume_io.py` 89 + affine helpers ~45) — slice
sorting along the true normal, median inter-slice spacing, an LPS→RAS affine, a SOP→position map.
Everything else is Phase B's code unchanged. **It is a geometry problem, not an architecture one.**

**The app half is the real cost, and not for the reason the survey guessed.** `/predict/ct` is not a
classifier wrapper — `_run_prediction` runs a **chain of four single-2D-image models** (sanity gate →
modality gate → feature-space OOD → classifier, then Grad-CAM), each added after a specific tested
failure. A DICOM series has no PIL image to hand any of them, so a volume route needs its **own**
guardrails. Verdict: **a second parallel prediction path** (new endpoint, new validation, new
guardrails, new response schema, new result view) sharing the classifier weights — not a rewrite of
the app, not "add a detector step" either. `/predict/ct` would stay exactly as it is.

**Measured end-to-end latency for a Phase E design: ~35 s/scan on this GPU** (4–16 s DICOM
read+resample, 15–35 s detection, ~1 s for the crops), **~3.5 min on CPU** — outside a synchronous
POST budget, so Phase E needs a job/polling shape. Detection cost is **≈0.44 s per sliding window**
and window count follows the *resampled* shape, so latency is estimable per series in advance.

### 6. What went badly, plainly

**Downloads, not the model.** `getImage` (whole-series zip) worked for four series (70–578 s each)
then **made zero visible progress for over an hour on five concurrent requests with no error** —
killed, and replaced with `getSOPInstanceUIDs` + parallel `getSingleImage` (Phase A/B's validated
route, issued for every slice), which recovered the rest with no failures. Throughput swung over an
order of magnitude (~60 KB/s to ~1 MB/s) → **2.5–15 min per series**; per-case size **33–402 MB,
2,739 slices / 1.24 GB for all 10 — not multi-GB per case**. **10 of the 12 pre-selected series
finished**, which meets the brief's 10–15 target; downloading was stopped rather than ground out for
the last two, and the partial 11th was deleted. All 12 were selected before any result was seen.

### 7. Verdict: feasible, costed, **deferred** — not closed

A real two-stage pipeline is **feasible**: none of the four things that could have killed it did
(localization 2.75 mm vs 13 mm; recall 20/22 at 0.5 and 21/22 at 0.3; FP 2.0/scan and mostly real,
one genuinely spurious box in 10 whole-chest scans; 0.69 GB on a 4 GB card). **But do not start Phase E now** — the
exam-deadline argument from Phase C/C.1 is unchanged and §5 above *adds* work the survey had not
accounted for. **Phase D closes as: the detector half works, here are the numbers, the pipeline is a
scoped engineering job for after the exam.**

Explicit non-claims: 10 patients of LIDC (the detector's own training lineage) measures *wiring*,
not generalization; the nodule mix is 18 B / 4 M (multi-nodule test series are mostly Benign), so
nothing here says it is equally good at both — though all 4 Malignant were found at 1.92–3.05 mm;
**no end-to-end pipeline accuracy number exists** and the detector + classifier figures must not be
quoted as one; FP characterisation is anatomical, not clinical. Phase A's deployment no-go and the
advice not to cite the live model's external-Malignant 12/12 as generalization evidence are both
untouched.

---

## Update 2026-09-09 — Phase D: candidate lung-nodule detector survey — **research only, nothing downloaded or run**

Full write-up: `eval_run/phaseD_detector_survey.md`. **No model was downloaded, installed, or
executed this session** — search/documentation only, per the task's explicit low-compute scope.

Surveyed against Phase C.1's ~13mm localization tolerance (v2_jitter checkpoint). **The central
finding: every well-documented candidate (MONAI's official `lung_nodule_ct_detection`, SANet)
operates on full 3D CT volumes, not the single 2D image `/predict/ct` currently accepts** — so
integrating any of them is not "add a detector," it's a change to what the app accepts as input in
the first place. This matters more than any per-candidate accuracy number for feasibility ranking.

**Ranked:**
1. **MONAI `lung_nodule_ct_detection`** (RetinaNet, LUNA16-trained, Apache 2.0, weights
   downloadable now, mAP 0.852 / AP@IoU=0.1 0.858) — best-documented, permissive license, same
   LUNA16/LIDC-IDRI lineage this whole investigation already uses. **Recommended first test for
   next session**: run it against Phase B/C's own LIDC volumes (ground-truth nodule locations
   already known from the manifest) and measure real center-offset error in mm directly — the
   IoU=0.1 metric is not convertible to an mm figure without knowing box sizes, so no mm-equivalent
   number is claimed here; that conversion requires actually running it.
2. A couple of small, unreviewed GitHub repos claim 2D-slice-oriented pipelines, which would
   sidestep the 3D-volume mismatch if their accuracy holds up — worth a quick inspection alongside
   #1, but unranked above it given no verifiable accuracy claims found in a search-only pass.
3. SANet — 3D volume like #1, but **non-commercial-only license** and trained on a different
   dataset (PN9, not LUNA16) — lower priority unless the license fits this project's use.
4. nnDetection — a self-configuring framework, not a ready pretrained model; requires a real
   training investment before it's known to be competitive. Not realistic before the exam
   deadline; noted for later only.

**Explicit deferral, not a silent stop:** testing candidate 1 against real data is the next
session's task, not started here, per the session-usage constraint this phase was scoped under.

Live system untouched (no code/checkpoint/data changes this phase — survey only).

---

## Update 2026-09-09 — Phase C.1: the two follow-ups Phase C flagged — **the position lock is fixable; the Benign caveat is real but small**

Full write-up: `eval_run/phaseC1_followups.md`. Artifacts: `eval_run/phaseC1/`, two new
checkpoints `checkpoints/ct_lidc_nodulecrop_v2_jitter.pt` and
`ct_lidc_nodulecrop_v3_jitter_loosebenign.pt` (**neither deployed**).
Phase C's `ct_lidc_nodulecrop_v1.pt` and Phase B's `labels_manifest.csv` untouched.
**Phase D was NOT started, per the brief.**

Exactly the two bounded follow-ups named in Phase C §9, and nothing else.

### 1. Translation augmentation — **it works, and it closes 96% of the gap**

**The 9 mm off-centre stress test, same script, same 207 crops, same DICOMs:**

| | v1 (Phase C) centred → shifted | **v2 (jitter) centred → shifted** |
|---|---|---|
| nodule accuracy (n=207) | 0.816 → **0.034** | 0.792 → **0.725** |
| Benign (n=132) | 0.841 → 0.008 | 0.818 → 0.788 |
| Malignant (n=75) | 0.773 → 0.080 | 0.747 → 0.613 |
| **Normal control (n=298)** | 0.963 → 0.936 | 0.923 → 0.913 |
| CAM enrichment at displaced lesion / at empty centre | **0.57× / 3.03×** | **4.21× / 0.50×** |
| CAM peak nearer the lesion than the frame centre | **2.4%** | **87.0%** |
| shifted crops called Normal | **195/207** @ 0.942 conf | **20/207** @ 0.688 conf |

Grad-CAM inverts completely — the model now searches instead of reading the middle. And
centred lesion attention is **preserved**: centre-disk enrichment 3.07× (v1) → 3.30×
(v2) against Phase C's measured 1.05× untrained baseline.

**Implementation detail that was load-bearing, not incidental.** `RandomAffine(translate=)`
would have been wrong: the stored crops are 128 px covering exactly 28 mm, so a 9 mm shift
means 41 px of *padding* — a synthetic artifact, tested against a probe deliberately built
to use real re-extracted pixels. Instead an oversized native-resolution window (28 mm + 10 mm
margin) is cut from the original DICOM per train row, and a 28 mm sub-window is taken at a
random offset at train time. **Zero-offset reproduces the stored Phase B PNGs bit-exactly
(98/98 and 110/110 spot-checked, max gray diff 0)**; 2,450/2,450 rows got the full ±10 mm
margin. Offsets drawn in **mm** and converted per-row via `px_mm`. **Applied to all three
classes** — jittering only nodules would make "was this jittered" predict the label.

**NEW measurement Phase C did not have — the tolerance CURVE, not one point:**

| shift (mm/axis) | Euclidean | **v1 nodule acc** | **v2 nodule acc** |
|---|---|---|---|
| 0 | 0.0 | 0.816 | 0.792 |
| 3 | 4.2 | **0.338** | **0.797** |
| 6 | 8.5 | 0.121 | 0.783 |
| 9 | 12.7 | 0.034 | **0.725** |
| 11 | 15.6 | 0.043 | **0.396** |
| 13 | 18.4 | 0.068 | 0.126 |

Two readings, one unflattering each way. **v1 is worse than Phase C reported** — "a few
millimetres" is under four: 0.338 at 4.2 mm, Benign specifically 0.129. And **v2's tolerance
is bought, not free** — it is flat to 8.5 mm, usable at 12.7 mm, then falls off a cliff at
15.6 mm, *exactly where the ±10 mm/axis training range ends*. The model learned the range it
was shown, **not general translation invariance**; no extrapolation claim is supported.
Widening it much further needs a bigger crop (±10 mm/axis in a 28 mm box already puts the
lesion 4 mm from the frame edge).

**Regression, not minimised: ~3 points across the board.** Test accuracy **0.9030 → 0.8693**,
balanced **0.8591 → 0.8292**, "is there a nodule here" 0.970 → 0.931, B-vs-M grading
0.816 → 0.792, F1 B/M/N 0.838/0.744/0.975 → 0.785/0.747/0.940. Vendor spread stays flat
(0.841–0.931 balanced across four manufacturers), so the acquisition-style channel did not
reopen. **Three points of centred accuracy for a ~3× wider centring tolerance** is the trade.

Training identical to Phase C otherwise (densenet121, seed 42, 15 epochs, batch 8 + AMP,
same class weights 1.2603/1.9822/0.5875 on the same 648/412/1390 counts — a free confirmation
nothing else moved). **9.6 min, peak 0.62 GB.** Trained from scratch, not fine-tuned from v1,
so "did jitter teach tolerance" isn't confounded with "how much lock survived".

**Methodology check:** the Phase C.1 copy of `offcentre_probe.py` was first run against **v1**
and reproduced Phase C §4 *exactly* (0.816→0.034, 0.57×/3.03×, 2.4%, 195 Normal @ 0.942), and
the eval script reproduced Phase C §2 exactly (0.9030/0.8591, same confusion matrix). The
before/after is one measurement, not two.

### 2. Looser Benign floor — **0.50 fails Phase B's own control; used narrowly anyway, and why**

Phase B's `floor_calibration.py` re-run with the grid extended down. Deterministic parts
reproduce Phase B **exactly** (28 mm Malignant p5/p25/p50 0.622/0.789/0.975, Benign
0.522/0.648/0.843, chest-wall control p75 **0.373** / max **0.753**).

| floor | keep M% | keep B% | **M−B gap** | **wall admitted** |
|---|---|---|---|---|
| **0.70** (Phase B) | 87.18 | 69.73 | **17.5** | **0.29%** (2/694) |
| 0.60 | 96.58 | 81.63 | 15.0 | 1.73% |
| 0.55 | 98.72 | 90.40 | 8.3 | 3.46% |
| **0.50** (Phase C's target) | 99.15 | 96.45 | **2.7** | **7.49%** (52/694) |
| 0.40 | 99.57 | 97.49 | 2.1 | 20.89% |

**Honest answer to "does 0.50 still keep wall-centred junk out": no, not to Phase B's
standard** — 7.49% vs 0.29%, a 26× rise. But it nearly eliminates the defect Phase B left
unresolved: the class-correlated retention gap goes **17.5 → 2.7 points**.

**Resolution: the loose floor was applied to the Benign nodule class ONLY; Normal was left at
0.70 and never re-sampled.** The chest-wall control exists to stop the *blind Normal sampler*
landing on the wall. A Benign crop is anchored on a radiologist-annotated nodule, so low lung
fraction there means a real juxtapleural lesion, not a crop of nothing. That argument licenses
loosening for the annotated class only — and nothing below is built on a looser Normal class.

**414 new Benign crops from 290 patients**, `lung_frac ∈ [0.50, 0.70)` — the band that was
never written to disk. Two free checks: the re-run counted exactly **913** crops as "already
written at 0.70", matching Phase B's Benign count; and **all 290 patients were already in
Phase B's manifest** (zero new patients), so every crop simply inherits its patient's existing
split — nothing reshuffled, zero patients in >1 split re-verified from the written CSV.
Benign **913 → 1,327 (+45.3%)**; dataset 3,461 → 3,875. Written as
`eval_run/phaseC1/labels_manifest_v2.csv`; **Phase B's manifest not modified.**

**QC (contour-vs-annulus, Phase A/B's instrument):** new band **+235.3 HU** mean, 371/414
positive, **wall fraction 0.380** — vs existing Benign **+422.4 HU**, 413/414, wall fraction
0.071. Lesions genuinely present, contrast roughly half, crops ~38% chest wall. 24 read
directly: all anatomically plausible juxtapleural nodules; in the 0.50–0.52 tail about half
are visibly wall-dominated. That is the honest visible cost.

### 3. What the looser floor answered — (a) yes, (b) redistribution not improvement, (c) confirmed then inverted

Benign recall on the extended 573-crop test set, by lung-fraction band. **v1/v2 never saw the
new band; v3 did.**

| Benign recall | ≥0.85 (n=95) | [0.70,0.85) (n=37) | **[0.50,0.70) NEW (n=68)** | all Benign (n=200) |
|---|---|---|---|---|
| v1 (no jitter, unseen band) | **0.863** | 0.784 | **0.765** | 0.815 |
| v2 (jitter, unseen band) | 0.800 | 0.865 | **0.662** | 0.765 |
| v3 (jitter + trained on band) | 0.716 | 0.703 | **0.868** | 0.765 |

**(a) Meaningfully more and different crops?** Yes — +45.3%, and a measurably different
population (38% vs 7% wall, +235 vs +422 HU contrast, in-contour −324 vs −374 HU).

**(b) Changed Benign performance?** **It redistributed it.** Holding jitter constant
(v2→v3), the new band goes **0.662 → 0.868 (+20.6 pts)** while the stricter bands *fall*
(0.800→0.716, 0.865→0.703). **Overall Benign recall is identical at 0.765**; F1 rises
0.789 → 0.805 through precision alone. The extra data moved where the capability sits rather
than adding capability — arguably the right redistribution clinically, but **not an
improvement, and calling it one would overstate the numbers.**

**(c) Does Phase C's bias direction persist?** **Yes for a model that never saw the band —
and it inverts for one that did.** v1's Benign recall falls monotonically with lung fraction
(0.863 → 0.784 → 0.765), so Phase C's reasoning from the [0.70,0.85) slice was correct. But
the magnitude is **~2–3 points, not a collapse**: adding the excluded population back moves
Benign recall **0.841 → 0.815** (precision 0.835 → 0.881, F1 0.838 → 0.847) and overall test
accuracy 0.9030 → 0.8866. And once the band is in training it becomes the band the model does
*best* on — **these crops are not intrinsically harder, they were simply absent.**
**So "treat Benign as an upper bound" updates to: an upper bound by about 2–3 points of
recall, measured — no longer an unquantified risk.**

**Design choice:** 3 of 4 cells of a 2×2 were run. v1→v2 isolates the augmentation; v2→v3
isolates the data change with jitter held constant. The missing cell (loose floor, no jitter)
answers nothing Phase C asked, so it was skipped rather than run for symmetry.

### Phase D: **recommendation unchanged — do not start it before the exam — but one of Phase C's two reasons is now falsified**

Phase C §9 said the schedule no-go "is not a close call" because "the detector has to centre
it to within a few millimetres, because that is the tolerance the classifier actually has …
§4 converts an unknown tolerance into a demonstrated hard requirement."

**That requirement is not hard — it was a property of the training distribution.** A one-line
augmentation concept, ~40 lines of real-pixel plumbing and a 9.6-minute run moved the usable
centring tolerance from **under 4.2 mm to ~12.7 mm** (accuracy at 12.7 mm: **0.034 → 0.725**),
for ~3 points of centred accuracy. A detector that must land within ~13 mm is a categorically
easier component than one that must land within 4 mm. Phase C guessed this follow-up "might
change the Phase D cost estimate materially"; it does, in that direction.

**What did not change:** the cliff sits at the edge of the augmentation range (no
extrapolation); the ~3-point accuracy cost is paid on every crop; a nodule detector is still
a project, not a task, and Phase A's costing of it stands; `POST /predict/ct` still takes a
whole slice; and "Normal" still means *this 28 mm patch has no nodule*, not *this patient is
healthy*. Phase A's deployment no-go holds unchanged.

**Updated assessment: GO on the evidence, NO-GO on the schedule — still not close, but now
for the scope reason alone, not the tolerance reason.** The tolerance objection has been
measured, attacked and largely removed.

**Do instead: write it up, as Phase C said.** Phase C.1 strengthens that write-up rather than
complicating it. The CT arc now reads end to end: the shortcut was real and structural →
removing it structurally worked → the resulting model learns real lesion features → its one
severe limitation was diagnosed *and then fixed*, with a measured before/after → and the
largest remaining uncertainty was bounded at 2–3 points instead of left open.

**Standing recommendation unchanged:** keep `ct_best_model_phase35_v2.pt` live; stop citing
its external-Malignant 12/12 as generalization evidence (Phase C §7(d) remains the reason —
most of those slices contain no visible lesion).

**Budget:** 9.6 min + 11.5 min training, ~45 min for all probes, sweeps, extraction and
calibration. Well inside the ~10-day exam window; nothing needed grinding through.

**Live system integrity, re-verified by re-hashing this session:**
`checkpoints/ct_best_model_phase35_v2.pt` sha **93b3a979884edbc4** (unchanged, matches
Phase C §8), `backend/main.py` sha **780120553178c0a5** (unchanged), `data/ct_scan/`
**1,036 files, newest mtime 2026-08-23** i.e. before this investigation began (Phase C's
exact aggregate-hash formula was not available, so file count + mtime is given rather than
presenting a differently-computed digest as a match). `ct_lidc_nodulecrop_v1.pt` and
`eval_run/lidc_full/labels_manifest.csv` both unmodified. Only two checkpoints written, both
new; all other writes confined to `eval_run/phaseC1/`. Disk added **~131 MB**.
**One process is running and it is not mine:** `uvicorn main:app --reload --port 8000`, the
project's own backend dev server, already running before this phase and untouched. No
training or evaluation process left running.

---

## Update 2026-09-09 (earlier) — Phase C: trained on the LIDC nodule crops — **real lesion signal, but zero localization**

Full write-up: `eval_run/phaseC_lidc_training.md`. Artifacts: `eval_run/phaseC/`, new
checkpoint `checkpoints/ct_lidc_nodulecrop_v1.pt` (**not deployed**).
**Phase D was NOT started, per the brief.**

This closes the question the whole investigation was built toward. **Answer: yes, there
is genuine lesion-level signal once the background/framing shortcut is structurally
removed — this is the first phase where Grad-CAM shows the model looking at pathology.
But the model has no localization ability at all, and that is measured, not guessed.**

**Training.** DenseNet-121 via `ml/model.py`'s `build_model`, loop adapted from
`ml/train.py` (same warmup→fine-tune schedule, Adam, seed 42, inverse-frequency weights
1.2603/1.9822/0.5875). Batch 8 + AMP on the RTX 3050: **peak 0.62 GB of 4.00 GB, no OOM,
no back-off needed** — ~6× headroom, kept at 8 anyway for comparability since the run is
only **9.4 min**. 15 epochs; best val balanced accuracy **0.8528 at epoch 7**. Manifest
`split` column used as-is, nothing re-split. (An earlier run was interrupted at epoch 8
by a pause and was **restarted from scratch rather than resumed** — a partial run's
checkpoint should not be treated as a finished model. The restart reproduced the
interrupted run's epoch-7 checkpoint exactly, a free seed-determinism check.)

**Test set (192 patients / 505 crops, patient-disjoint): accuracy 0.903, balanced
0.859.** Benign F1 0.838 / Malignant 0.744 / Normal 0.975. The error structure is the
informative part: only 4 of 207 nodule crops are called Normal and 11 of 298 Normals are
called a nodule, so **"is there a nodule in this patch" runs at 0.970 (balanced 0.972)**
and essentially all error is Benign↔Malignant grading (0.816 on the 207 nodule crops).
Finding the nodule in a centred patch is near-solved; grading it is hard, as
four-radiologist subjective labels imply.

**Grad-CAM — the check this whole investigation existed for. It passes.** All 505 test
crops, same enrichment metric as `ct_shortcut_inversion_analysis.md` /
`ct_framing_normalization_experiment.md`. **Centre enrichment 2.96×** against a
**measured** architectural baseline of **1.05×** (ImageNet densenet121, untrained head,
identical pipeline). **Border enrichment 0.17× vs the control's 0.88×** — training pushed
attention *away* from the crop edge, so the obvious new shortcut did not appear.
Misclassified crops show consistently weaker, more diffuse attention (Benign 3.77×→1.85×,
Normal 4.18×→2.50×), i.e. the metric is coupled to the decision. For comparison, phase37
sat at **1.00–1.07×** on images it classified correctly. This is a categorical change.

**The decisive new probe — off-centre re-extraction (`offcentre_probe.py`).** Grad-CAM
alone could not separate "found the lesion" from "always looks in the middle", and there
was a specific reason to worry: **Normal crops, which contain no lesion, showed the
*highest* centre enrichment (4.18×, median peak 0.09 mm from dead centre)**. So the same
lesions were re-extracted from the same DICOM slices with the box shifted 9 mm — real
pixels, identical windowing code, no synthetic translation.

| | centred | shifted 9 mm |
|---|---|---|
| nodule accuracy (n=207) | 0.816 | **0.034** |
| **Normal control (n=298)** | 0.963 | **0.936** |
| CAM enrichment at lesion / at empty centre | — | **0.57× / 3.03×** |
| peak nearer lesion than frame centre | — | **2.4%** |

**Read this correctly — it is not another shortcut.** The Normal control is load-bearing:
Normals go through the identical shift and barely move, so the shift does not damage the
input; the collapse is caused specifically by moving a lesion out of the centre.
**195 of 207 shifted crops become "Normal" at 0.942 mean confidence** — which is the
*correct* answer about the centre of the frame. Re-extracting unshifted reproduces the
stored PNGs' predictions on **100.0%** of crops, so the probe's code path is faithful.
The model learned a genuine local lesion-presence-and-character feature and applies it at
a fixed location. That is a **scope** limitation, not a validity one — but a severe one.

**Style probe at 32× Phase A's data (`style_probe.py`).** 3-class balanced accuracy
**0.523 RF / 0.535 LR** (chance 0.333), vs Phase A's 0.433 on 107 crops and **0.944** on
the old whole-slice data. The rise is *not* a returning shortcut: Phase A's number came
from 26 patients and carries huge variance, and the top RF features are `std`, `mean`,
`frac_gt200`, `p75`, `mean_HU_crop` — all measures of *how much dense tissue is in the
box*, which is crude pathology, not acquisition style. **The control settles it:
manufacturer one-hot alone gives 0.363 against chance 0.333** — the vendor/style channel
that drove every prior failure is dead, matching the test-set vendor spread
(0.839–0.928 across four vendors). **The shortcut surface stayed collapsed.**

**Benign is still not trustworthy, and Phase B's caveat is why.** Phase B flagged that no
lung-fraction floor is class-neutral (Benign rejected 12–22 points more often). Benign
recall is 0.841 and its confusions are with Malignant (17) not Normal (4), so localization
is fine; but the surviving Benign crops are biased toward the larger/more central end of
the benign distribution. The `lung_frac ∈ [0.70, 0.85)` slice does worst (balanced 0.756
vs 0.894 for ≥0.85, Malignant recall 0.860→0.500), and those near-wall crops are
disproportionately the benign ones the floor already discards — bias direction and
performance gradient agree. **Treat Normal/Malignant as trustworthy within this dataset
and Benign as an upper bound.** Settling it needs re-extraction at a looser floor, not a
filter — the looser crops were never written.

**External comparison ATTEMPTED and DECLARED INVALID, on four measured grounds** — no
external-Malignant number was manufactured. (a) mm/px is unknown and must be guessed from
chest width (±15%). (b) 28 mm works out to 28–70 px vs LIDC's 128 px — 2–4× short on the
detail that *is* the classification. (c) Window mismatch: LIDC parenchyma sits at gray
~82; external Malignant reads 29–65 and external Normal **0–112** — `normal_21` is a
soft-tissue window (lung pure black, gray 0.0) and `normal_6` carries a tiled
SCIENCEPHOTOLIBRARY watermark. **The two external sets are windowed differently from each
other**, so comparing them compares rendering, not pathology. (d) Decisively: **all 12
external Malignant slices were opened and only ONE (`adenocarcinoma_000113`) contains a
discrete, confidently localizable focal lesion** — four are at the thoracic inlet/apices
where no lung lesion exists at that level, five show grossly clear lungs. They are labelled
by carcinoma subtype at *case* level; the slice is not nodule-annotated.
**This retro-illuminates the whole investigation: the external-Malignant 12/12 that three
checkpoints scored was never lesion detection, because most of those slices do not
visibly contain a lesion.** Independent new support for the standing recommendation.
The one producible number — external Normal lung crops, which need no localization —
is 51/54 called Normal, but it is **explicitly not a generalization score**: a degenerate
always-Normal model scores 54/54, and §4 shows this model's dominant failure mode *is*
answering Normal when a lesion is not centred.

**Live-model comparison, and why the lower number is the better result.** `phase35_v2`
scores ~98.1% on 103 single-source whole slices with Grad-CAM enrichment of **1.00×** on
images it gets right and a class-from-style probe at **0.944**. This scores **90.3%** on
patient-disjoint splits across four vendors, on four-radiologist labels with the ambiguous
quarter excluded, with attention at **2.96×** and the style probe at **0.523 / vendor
0.363**. Harder, more legitimate task, measured more honestly — that is how it should be
reported, not apologised for.

### Phase D: **GO on the evidence, NO-GO on the schedule. Do not start it before the exam.**

The evidence says go — Phase D's premise needs the classifier to be worth front-ending,
and that was genuinely open at the end of Phase B and is now closed: 2.96× vs a 1.05×
baseline, attention that tracks lesion presence under a controlled off-centre
re-extraction, a dead vendor channel, and 0.970 on "is there a nodule here". **This is
not the sophisticated-negative-result outcome the brief allowed for.** Earlier phases
produced models that were right for the wrong reasons; this one is right for the right
reason, in a narrow scope.

The schedule says no-go, and it is not close. **Accuracy 0.816 → 0.034 the moment the
lesion moves 9 mm** means the entire burden of finding the lesion falls on a detector
that must centre it to a few millimetres — a tolerance this phase has now *measured* and
found tight. Phase A costed a detector as "a larger project than everything in Phase A
combined, and not a ~10-day-with-an-exam undertaking"; §4 makes that estimate more
pessimistic, not less, by converting an unknown tolerance into a demonstrated hard
requirement. Plus the unchanged deployment facts: `POST /predict/ct` takes a whole slice,
and "Normal" here means *this 28 mm patch has no nodule*, not *this patient is healthy*.

**Do instead: write it up.** Phase C converts the CT story from "we report 98% and cannot
explain the external failure" into a documented finding — the shortcut was real, it was
structural, removing it structurally *worked*, and the resulting model learns real lesion
features but cannot localize. Stronger to present than either a fragile 98% or an
unfinished two-stage system.

**If Phase D is picked up after the exam**, two cheap things first: (1) re-extract at a
looser lung-fraction floor to bound the Benign selection bias — currently the largest
un-quantified uncertainty; (2) **train with random translation augmentation of the crop
centre** — a one-line change and a 9-minute run that directly attacks the position-lock
and could materially change the detector's required tolerance.

**Standing recommendation unchanged and now reinforced from a new direction:** keep
`ct_best_model_phase35_v2.pt` live; stop citing external-Malignant 12/12 as generalization
evidence — §7(d) shows most of those slices contain no visible lesion at all.

**Budget:** 9.4 min training + ~25 min for all evaluation and probes. Well inside the
~10-day exam window; nothing needed grinding through.

**Live system integrity, re-verified by re-hashing this session:**
`checkpoints/ct_best_model_phase35_v2.pt` sha **93b3a979884edbc4** / mtime **Aug 24 12:45**
(unchanged), `backend/main.py` sha **780120553178c0a5** / mtime **Sep 6 14:18**
(unchanged), `data/ct_scan/` **1,036 files**, aggregate **f6f64d2a35e2e525** (unchanged) —
all matching the values recorded in `ct_framing_normalization_experiment.md` §8. No
existing checkpoint modified; the only one written is the new
`ct_lidc_nodulecrop_v1.pt`. All other writes confined to `eval_run/phaseC/`. Disk added
**51 MB**. No process left running.

---

## Update 2026-09-08 — Phase B: full-scale LIDC-IDRI crop extraction — dataset built, two defects fixed, one partly unresolved

Full write-up: `eval_run/phaseB_lidc_extraction.md`. Artifacts: `eval_run/lidc_full/`.
**Extraction and data quality only — no training, and none should follow without a
separate decision.**

Scaled Phase A's 30-case / 107-crop feasibility check to the whole collection.
**3,461 crops from 797 patients** — 562 Malignant / 913 Benign / 1,986 Normal — with
patient-disjoint splits and zero overlap. Phase A's cost model held: **21.1 min /
1.05 GB** to fetch 1,999 slices against a predicted ~25 min / 1.1 GB; extraction 81 s;
~35 min of compute total, well under Phase A's "~1 focused day".

**Defect 1 (label mapping) — fixed, and the fix is free.** Six reconciliation rules
compared on all 2,680 rated clusters. Phase A's mean-with-dead-band rule excludes
**38.4%** at scale (it predicted 41.5% from 30 cases — good extrapolation). The decisive
result: **on clusters that any two rules both label, they never disagree — 0 label flips
in every pairing.** The rules differ only in coverage, so this is a strictness dial, not
competing labellings. Majority-vote is *stricter* than the mean (50.2% excluded), and
2-of-4 agreement collapses the set (71.3%). Chose the union of the mean-dead-band rule
with the looser mean-split-at-3: **recovers 387 clusters, 38.4% → 23.9% excluded, with
zero relabelling risk**, and flags them `is_borderline=1` so Phase A's stricter dataset
is reproducible by filtering rather than re-extracting.

**Defect 2 (crop size / lung fraction) — fixed, but worse than Phase A thought, and one
part remains open.** Phase A's recommended **≥0.85 floor does not survive scale**: its
24 mm figures (0.88/0.83/0.93 on 30 cases) reproduce as **0.82/0.82/0.87** on 600
slices, and applying 0.85 unchanged would discard 30–51% of crops. Diagnosed first
(morphological closing of the lung mask recovers only +1.3 points and starts eating the
chest wall, so the mask is fine), then recalibrated against a **chest-wall control set**
(p75 = 0.373, max 0.753). Settled on **28 mm crops, floor 0.70** — admits 0.3% of
deliberately wall-centred crops while retaining ~20 points more real data than 0.85, and
*narrows* the class gap. `lung_frac` is a manifest column, so the strict-0.85 subset
(2,658 crops) is one filter away.

**The unresolved part, flagged rather than buried:** *no* lung-fraction floor is
class-neutral. At every size and threshold tested, **Benign is rejected 12–22 points more
often than Malignant** — LIDC's benign nodules are genuinely smaller and more often
subpleural, and the lesion-inclusion correction mechanically protects large (mostly
malignant) lesions. The mitigation was to keep the floor as loose as the control allows
and record both fraction variants per row. **Re-examine this before believing any
Benign-vs-Malignant result from this data.**

**Carry-forward facts, re-verified at scale:**
- **Coordinate convention confirmed, do not re-litigate.** Contour-vs-annulus on 400 new
  crops: `xCoord`=column **+391.4 HU (386/400 wins)** vs `xCoord`=row −0.6 HU (14/400).
  Reproduces Phase A's +372 HU / 52-of-55 at 7× the sample. The extractor hard-fails if
  this ever flips.
- **0 SOPInstanceUID mismatches across 1,997 slices.** Filenames and slice ordering are
  used nowhere.
- **283 of 1,318 XMLs still use the older `IdriReadMessage` root tag**; skipping them
  still covers **1,018/1,018 CT series, 0 unmatched**. Identical to Phase A.
- Clustering unchanged (greedy, ≤10 mm, one read per session) with one documented
  improvement: distance to the cluster's **running centroid**, not its first member.
  2,680 rated clusters; **793 (29.6%) have only a single reader**.

**New issues Phase A's 30 cases did not surface:**
- **2 slices are permanently un-downloadable** (HTTP 500 from `getSingleImage`, 3
  attempts each). Server-side, not rate limiting. **Unresolved**; costs 4 of 3,461 crops
  (0.1%), accepted.
- **Filename collisions**: 8 patients contribute two CT series, so Phase A's
  `<patient>_n<cluster>.png` naming would have silently overwritten crops. Caught before
  the run, fixed with a series suffix; 3,461 rows / 3,461 unique paths verified after.
- Phase A's extractor loads all DICOMs into memory (~1 GB at 2,000 slices) — restructured
  to stream.
- **Scanner balance is GE-dominated at scale** (GE 2,048 / Siemens 822 / Philips 328 /
  Toshiba 263). Phase A's even 4-way spread was an artifact of its round-robin 30-case
  sample, not a property of LIDC-IDRI.

**Splits — the repeat-offender bug did not recur.** train 410 patients / 2,450 crops,
val 195 / 506, test 192 / 505. Zero patient overlap, verified twice independently (from
the assignment map, then re-derived from the written CSV). Class proportions hold
(malignant share 16.8 / 14.8 / 14.9%). Note val/test are made of many low-crop-count
patients (~2.6 crops each vs train's 6.0).

**QC: 30 crops (10/class) read directly — 30/30 anatomically plausible, none
malformed.** Textbook centred nodules at both extremes of size; three Normal crops show
a visible pleural band, which is the honest, visible cost of the 0.70 floor. In-contour
HU **−336 Malignant / −374 Benign** vs **−750** crop-level Normal — correct ordering, but
much weaker separation than Phase A's −204/−411, which was the easy tail of a small draw.

**Standing recommendation unchanged.** Keep `ct_best_model_phase35_v2.pt` live; stop
citing external-Malignant 12/12 as generalization evidence. Nothing in Phase B changes
what should be *said* about the current model — and Phase A's deployment no-go still
holds, because a nodule-crop classifier cannot be served by `POST /predict/ct`, which
takes a whole uploaded slice.

**Live system integrity, re-verified this session:**
`checkpoints/ct_best_model_phase35_v2.pt` mtime **Aug 24 12:45** (unchanged),
`backend/main.py` mtime **Sep 6 14:18** (unchanged), `data/ct_scan/` **1,036 files**
(unchanged). All writes confined to `eval_run/lidc_full/`. Disk added ~1.05 GB, almost
all DICOM; Phase A's annotation XMLs were reused in place, not duplicated.

---

## Update 2026-09-08 — Sliding-window multi-crop inference — no-go, clean negative

Full write-up: `eval_run/ct_multicrop_experiment.md`. Bounded, single-session, no retraining,
no downloads. Tested whether ensembling the existing untouched `ct_best_model_phase35_v2.pt`
over a 3×3 sliding-window grid of sub-crops (3 aggregation strategies: max-pool per class,
majority vote, average probability) could improve external Normal detection while staying
deployable through the existing `/predict/ct` shape (aggregation server-side, one image in/one
prediction out — the appeal vs. the LIDC nodule-crop approach, which needs a separate detector).

**Result: no aggregation strategy moved external Normal at all (0/9 → 0/9, all three), and
all three regressed in-dataset accuracy** (Malignant 50/50→41-42/50, Normal 41/41→20-26/41,
Benign 10/12→8-11/12). External Malignant held at 12/12 throughout — consistent with the
same global-style shortcut firing on most crops of most external images regardless of which
crop is shown, so pooling doesn't change the outcome, it just adds more chances for that
shortcut to also misfire on previously-correct in-dataset Normal images. Grad-CAM sanity
check (step 4) was not triggered — no strategy showed the "meaningful improvement" its
trigger condition required.

**Why this was expected in hindsight, given `ct_framing_normalization_experiment.md`:** the
shortcut is a *global* style/framing signal (background, table visibility, contrast/HU-floor
characteristic of the source). A 50%-width sub-crop still carries most of that same global
signal — sliding-window cropping changes which part of the image is shown, not whether the
style cues in that part are still present. Only the framing-normalization experiment's
explicit background/table masking addressed that, and even that didn't produce real lesion
learning.

**Standing recommendation unchanged (now reinforced by a third independent negative
result):** keep `ct_best_model_phase35_v2.pt` live; stop citing its external-Malignant 12/12
as evidence of generalization.

---

## Update 2026-09-08 (later) — Phase A: LIDC-IDRI nodule-crop feasibility — GO for evidence, NO-GO for deployment

Full write-up: `eval_run/phaseA_lidc_feasibility.md`. Artifacts:
`eval_run/lidc_feasibility/`. **Feasibility check only — no training, no full-scale
extraction, stopped at the go/no-go as instructed.**

New idea tested (not tried in any prior phase): instead of classifying whole CT slices —
which every prior phase showed lets the model shortcut on background/table/framing —
extract small crops **centered on LIDC-IDRI's radiologist-annotated nodule locations**, so
there is no background or framing left in the input to shortcut on.

**Two findings that supersede Phase 1's conclusions:**

1. **Access is trivial, not hard.** `phase1_data_candidates.md` rated LIDC-IDRI as high
   engineering cost needing substantial DICOM work. That assessment assumed whole-volume
   downloads. In fact TCIA's NBIA REST API is **anonymous, no account, no DUA, no desktop
   tool** — and critically, `getSingleImage` fetches **one slice by SOPInstanceUID**. Since
   the annotation XMLs (**9 MB for all 1,010 cases**) name the exact `imageSOP_UID` of every
   annotated slice, you download 9 MB of annotations, parse them, and then fetch *only the
   slices that matter*. **30 cases cost 28.4 MB instead of ~3.8 GB.** The 130 GB full
   collection never needs to be paid for. Phase 1's cost estimate should be treated as
   superseded.
2. **The shortcut surface measurably collapses.** Re-running the probe that decided the
   framing experiment (§2C: can a random forest predict class from global style statistics
   alone), patient-grouped: **0.944 balanced accuracy on the old whole-slice training data →
   0.433 on these crops** (chance 0.333). Achieved *structurally*, by construction, without
   any of the four iterations of masking/histogram surgery the framing pipeline needed.
   This is the first intervention in the investigation that removes the shortcut surface at
   the level the root cause was actually identified at.

**Extraction worked. 107 crops from 26 patients: 25 Malignant / 30 Benign / 52 Normal**,
across 4 scanner vendors (Toshiba 36, Philips 26, GE 23, Siemens 22) — genuinely
multi-vendor, which no prior phase's data was. ~1 hour end to end.

**Sanity check: 12 crops read directly, all 12 anatomically correct, none malformed.**
Nodules visibly present and centred (the 32 mm rating-5 case is a textbook spiculated mass);
Normals show parenchyma with normal vessel branching and no mass. Mean HU inside contour
−204 Malignant / −411 Benign vs −631 crop-level Normal — the density ordering real pathology
would give.

**Recommendation is split by goal, and this distinction is the whole point:**

- **Improving the deployed CT model before the exam — NO-GO.** Not because extraction is
  hard, but because a nodule-crop classifier **cannot be served by `POST /predict/ct`**,
  which takes a whole uploaded slice. Deploying it needs a nodule *detector* (or a
  user-drawn ROI) to produce the crop at inference — a larger project than everything in
  Phase A combined, and not a ~10-day-with-an-exam undertaking. The class semantics also
  shift: "Normal" here means *this 3 cm patch of lung has no nodule*, not *this patient is
  healthy*.
- **Answering the investigation's open question — GO, bounded to one day.** The phase37
  report left unresolved whether there is *any* learnable lesion-level signal once the
  shortcut is gone, or whether ~1,000 single-source images is simply too few. This is the
  cheapest way to answer it and yields a defensible finding either way, including a clean
  negative. Do it only if that day is genuinely spare: **the standing recommendation is
  already correct and complete without it.** This buys a better explanation, not a better
  product.

**Standing recommendation unchanged:** keep `ct_best_model_phase35_v2.pt` live; stop citing
external-Malignant 12/12 as evidence of generalization. Nothing in Phase A changes what
should be *said* about the current model.

### Carry-forward notes for anyone doing Phase B

- **Crops are currently too large.** At 48 mm, ~50–60% contain some non-lung structure
  (chest wall/mediastinum) — partially reintroducing the anatomy the approach exists to
  exclude. It is *not* strongly class-correlated (lung fraction 0.79/0.73/0.81), and it
  shrinks at 24 mm (0.88/0.83/0.93). **Phase B must tighten to ~24–32 mm and add a hard
  lung-fraction floor (≥0.85).** Parameter change, not a redesign.
- **Coordinate handling is verified, don't re-litigate it.** `xCoord` is the **column**
  index — established empirically, not from memory: contour-vs-annulus contrast is **+372 HU
  (52/55 nodules)** under the column reading vs **+28 HU** under the row reading. Every
  downloaded file's `SOPInstanceUID` tag was re-checked against the UID requested (54/54,
  0 mismatches); slice ordering and filenames are never used.
- **Beware a misleading QC metric.** A naive "is the crop centre denser than its ring" check
  reads only +51 HU / 33-of-55-positive and *looks* like a centering bug. It is an artifact —
  for a 5 mm nodule in a 48 mm crop the ring is dominated by chest wall, which is denser than
  any nodule. Use the contour-vs-annulus test instead.
- **Annotation format:** 283 of 1,318 XMLs use a second, older root tag `IdriReadMessage`
  instead of `LidcReadMessage`. Skipping them still covers all 1,018 CT series, but a naive
  parser will crash on them.
- **`pylidc` was deliberately not used** — it needs SQLAlchemy plus a configured path to a
  complete on-disk DICOM library, i.e. it assumes whole volumes, the exact assumption this
  approach avoids. A hand-rolled `xml.etree` parser was cheaper and is in
  `parse_lidc_xml.py`.
- **Label mapping is a judgment call, not a standard.** Mean of available radiologist
  ratings: ≤2.5 Benign, ≥3.5 Malignant, in between excluded. **This excluded 39 of 94 nodule
  clusters (41.5%)** — so the retained set is the *easy tail*, and a model trained on it will
  look better than it would on the full nodule population. Also note these are radiologists'
  subjective impressions, **not biopsy-confirmed pathology**.
- LIDC gives **no cross-reader nodule correspondence** (`noduleID` is not shared across
  sessions); reads were clustered greedily at ≤10 mm 3D centroid distance. 24 of 55 nodules
  have a full 4-reader panel. A real build should use a more principled clustering step.
- Honest scale estimate, from measured rates (1.42 slices/s at 6 threads): **~2,000 labelled
  nodule crops from all 1,010 patients ≈ 25 min / 1.1 GB** of downloading (2.9 h / 7.7 GB if
  taking ±3 adjacent slices per nodule). **~1 focused day** to a complete QC'd dataset. The
  download is not the bottleneck; the judgment calls are.
- Disk added: **209 MB** (`eval_run/lidc_feasibility/`), of which 28 MB DICOM and ~170 MB
  extracted annotation XML. The 9 MB XML zip covers all 1,010 cases, so Phase B needs no
  re-download of annotations.

**Live system integrity, re-verified this session:**
`checkpoints/ct_best_model_phase35_v2.pt` mtime **Aug 24 12:45** (unchanged),
`backend/main.py` mtime **Sep 6 14:18** (unchanged), `data/ct_scan/` **1,036 files**
(unchanged). All writes were confined to `eval_run/lidc_feasibility/`; `backend/main.py` was
read only, to confirm the `/predict/ct` input-shape claim above.

---

## Update 2026-09-08 — Phase 1 data-sourcing feasibility: recommendation is to stop

A follow-up task asked for a research-only survey of non-IQ-OTH/NCCD,
non-`mohamedhanyyy` CT datasets, prioritizing multi-institution sources, as a candidate
Phase 2 (source new data → retrain again). Full candidate list and per-candidate
findings: `eval_run/phase1_data_candidates.md`. No downloads were performed — search
only, as scoped.

**Recommendation: do not proceed to Phase 2. Conclude that this CT generalization issue
is not fixable with reasonably obtainable data right now, at least not via the
"download a pre-packaged folder dataset, dedupe, retrain" approach used in every prior
step of this investigation.**

Why, plainly: every open, unrestricted-access candidate with genuine Benign/Malignant/
Normal-style labels (LIDC-IDRI, SPIE-AAPM/LUNGx) is a raw 3D DICOM volumetric collection
with nodule-level (not scan- or slice-level) annotations — none of them hand you a
ready-to-use folder of 2D images with one label per image the way IQ-OTH/NCCD and
`mohamedhanyyy/chest-ctscan-images` did. Building a usable dataset from them means writing
a DICOM-processing and nodule-to-slice-extraction pipeline from scratch — a materially
larger, slower, riskier undertaking than anything tried in phases so far, not a natural
continuation of it. The two sources that are genuinely multi-institution at real scale
compound this: LIDC-IDRI (7 centers) has no native "Normal" class at all — you'd have to
manufacture one — and NLST (~33 centers) gates its full dataset behind a formal Data Use
Agreement application, with only a small (~500-patient) subset openly available and even
that only at the weak "cancer within 1 year" exam-level label, not a per-image class.

This isn't a capacity judgment about whether more/better data could *in principle* help —
it plausibly could. It's a judgment that no dataset meeting this project's actual
constraints (open access, usable label granularity, reasonable engineering cost) was
found, despite a real search across the standard sources (TCIA, Kaggle, IEEE DataPort).
Pursuing this further would mean scoping a new, much bigger data-engineering project
(DICOM handling, nodule extraction, HU windowing, a self-built Normal class), not a Phase
2 sourcing pass like the ones that supplied the phase36/phase37 experiments' data.

**Standing recommendation from the prior phases is unchanged and reinforced by this:**
keep `ct_best_model_phase35_v2.pt` live, and stop treating its external-Malignant 12/12
as evidence of real generalization — both retrained variants reproduced or inverted that
number without ever demonstrating lesion-based learning, and there is no readily
obtainable data path in view to test whether more data would fix that.

---

Written as a safety checkpoint ahead of a session usage limit. **Correction to the
triggering request's premise: nothing was actually mid-run.** The framing-normalization
experiment (the task this checkpoint was meant to interrupt) had already finished and
its completion notification arrived in the same turn as this checkpoint request. There
was no active training process to stop, no partial checkpoint to save, and no risk of a
corrupted file — the last checkpoint-relevant action (saving
`ct_best_model_phase37_framing_normalized.pt`) completed cleanly at 20:51 on 2026-09-06,
before this request was made.

**Status: the task is fully done, not paused.** No new experiment was started in
response to this request, per its own instruction not to.

## What was in progress, and its actual completion state

Full chain of investigation, in order, all complete:

1. `eval_run/ct_normal_root_cause.md` — root cause of CT Normal-class external failure
   (0/9). **Done.**
2. `eval_run/ct_preprocessing_fix_test.md` — preprocessing-only fix attempt, failed.
   **Done.**
3. `eval_run/ct_retrain_experiment.md` — retrain with diversified Normal data
   (`ct_best_model_phase36_normal_fix.pt`): fixed Normal, broke external Malignant
   12/12→0/12. **Done.**
4. `eval_run/ct_shortcut_inversion_analysis.md` — Grad-CAM diagnosis of #3: neither
   checkpoint attends to lesions; both key on acquisition-framing (table
   visibility/crop style). **Done.**
5. `eval_run/ct_framing_normalization_experiment.md` — **this session's task**: built a
   framing-normalization pipeline, verified the confound was actually removed (iterated
   4 versions of the leak-check before it passed clean), retrained as
   `ct_best_model_phase37_framing_normalized.pt`, re-ran Grad-CAM and full validation.
   **Done. Verdict: outcome (c) — neither.** The framing confound was successfully
   removed (class-from-style probe inside training data: 0.944 balanced accuracy →
   exactly 0.333, chance), but this did **not** produce genuine lesion-based learning.
   Phase37 reproduces the same "call everything Malignant" behavior as the original
   model (224/285 external images called Malignant against a true prevalence of 4.2%),
   and Grad-CAM enrichment analysis shows no lesion-region preference even on
   confidently-classified in-domain Malignant images (1.00× enrichment, ~23% of
   attention mass on constant background). This falsifies the framing-confound
   hypothesis as *the* blocking cause. Dataset size (1,036 images) is now the leading
   unexplained factor but is not proven by this experiment.

**There is no step 6 planned or started.** No further experiment is queued.

## Next action on resume

**None required to finish this line of investigation** — it concluded with a
recommendation already stated in `ct_framing_normalization_experiment.md`:
**adopt nothing further; keep `ct_best_model_phase35_v2.pt` live, and stop citing its
external-Malignant 12/12 as evidence of real generalization** (both this and the phase36
experiment show that number is an artifact, not lesion detection — phase37 reproduces it
via a completely different, uncorrelated mechanism, which is itself evidence it's not
meaningful).

If a next session wants to continue the investigation rather than close it out, the
open, unaddressed question per the phase37 report is: **is 1,036 images fundamentally
too few for this architecture to learn real lesion-level texture, independent of the
framing confound?** That would require a differently-designed experiment (e.g., a much
larger external CT source, or a synthetic/augmented data study) — nothing toward that
has been started, and no file exists for it yet.

## Key intermediate artifacts (all present on disk, verified below)

| Path | What it is |
|---|---|
| `eval_run/framing/ct_framing_normalize.py` | The final (v4, clean) framing-normalization pipeline |
| `eval_run/framing/ct_framing_leakcheck.py`, `leakcheck.json`, `leakcheck_v4.log` | Confound-removal verification, all 4 iterations |
| `eval_run/framing/diagnose_leak.py` | Diagnostic tool built mid-experiment to find what v1–v3 were leaking on |
| `data/ct_scan_framing_normalized/` (31MB) | Rebuilt dataset, framing-normalized, same splits as `data/ct_scan/` |
| `eval_run/framing/external_normalized/` | Framing-normalized external Malignant/Normal test images |
| `checkpoints/ct_best_model_phase37_framing_normalized.pt` (28MB) | The new experimental checkpoint — **not deployed** |
| `checkpoints/ct_phase37_framing_normalized_best.pt` (28MB) | Appears to be an earlier/intermediate save from the same training run (created 20:47, 4 min before the file above) — both are experimental artifacts, neither is live |
| `eval_run/framing/gradcam_phase37_pngs/`, `gradcam_phase37_grid.png` | Grad-CAM overlays for the new checkpoint |
| `eval_run/framing/results_ct_phase37.json`, `protected_before.json` | Validation results and the pre-experiment protected-file snapshot the agent hashed against |
| `eval_run/ct_framing_normalization_experiment.md` | The full write-up with the three-way before/after comparison table and final verdict |

## Decisions/choices made worth knowing about

- **Thresholding/contour-based body-silhouette masking was used instead of a trained
  segmentation model** — no CT-specific U-Net exists in this repo (only X-ray has one).
  This was an explicitly pre-approved choice, not an improvised shortcut.
- The masking pipeline required **4 iterations** before the leak-check passed clean:
  v1–v3 each looked sufficient until a fresh probe found a residual signal (padding
  fraction, then body-silhouette fill fraction, then a subtle rank-averaging artifact
  in histogram specification affecting tied intensities). v4's fixed elliptical support
  + dithered histogram specification is what finally closed all of them. This iteration
  history is documented in the completion report and worth reading if anyone repeats
  this kind of masking exercise on medical images — tied-intensity handling in
  histogram matching is an easy, non-obvious leak source.
- Even after the class-correlated shortcut was provably removed, a **separate,
  non-class-correlated domain-shift signal remains** detectable from a 32×32 patch
  (origin-recovery AUC 0.968) — the agent flagged this explicitly rather than claiming
  full domain-invariance, since it wasn't what was being tested (only class-correlation
  needed to reach chance, which it did).
- Grad-CAM's native resolution (7×7, densenet121's final feature map) limits how finely
  "attends to lesion vs. background" can be quantified — flagged as a real caveat that
  softens conclusions slightly but was judged not to overturn them (23% attention mass
  on constant background is well beyond what resolution alone would explain).

## Live-system integrity — re-verified independently just now, not just re-stating the agent's claim

Checked directly in this turn, not taken on trust:

- `checkpoints/ct_best_model_phase35_v2.pt` — mtime **Aug 24 12:45**, unchanged since
  before any of this investigation began.
- `backend/main.py` — mtime **Sep 6 14:18**, unchanged since before this investigation's
  first task today.
- `data/ct_scan/` — **1,036 files**, matching the expected pre-investigation count.
- No `backend/main.py` edits, no `CHECKPOINTS` dict changes, nothing pointing the live
  app at any experimental checkpoint.

## Clean shutdown

- No backend dev server, frontend dev server, or training process was found running at
  checkpoint time (checked via process list — zero matching `uvicorn`, `train.py`, or
  `npm` processes). The framing-normalization agent's own training run had already
  exited on its own before this request arrived; there was nothing to stop.
- No new process was started by this checkpoint task itself.

## Disk-usage note (not urgent, just visibility)

`eval_run/` is now **4.8GB total**, almost all of it (**4.7GB**) in
`eval_run/external_samples/` — the raw Kaggle dataset downloads used across this whole
investigation (`_downloads/jtiptj` alone is ~1.7GB, plus the CT sources). Nothing here is
implicated in any risk to the live system — it's inert test data — but it's worth knowing
about if disk space becomes a concern later. Not deleted automatically; that's a call for
whoever resumes this to make.
