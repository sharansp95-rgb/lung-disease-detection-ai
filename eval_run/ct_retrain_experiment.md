# Retraining experiment: does more anatomically diverse Normal CT data fix the external-Normal collapse?

**Outcome in one line: it fixes the Normal collapse (0/264 → 250/264 on held-out
external Normals) but breaks external Malignant completely (12/12 → 0/12). Net
result is not adoptable. The experiment also produced a stronger diagnosis than
the one it set out to test.**

Nothing here is deployed. `backend/main.py`'s `CHECKPOINTS` dict is untouched,
`checkpoints/ct_best_model_phase35_v2.pt` is untouched, `data/ct_scan/` is
untouched, and the live backend was never started against the new checkpoints.

---

## 1. Data sourcing and dedup

### 1a. `mohamedhanyyy/chest-ctscan-images` (already downloaded)

215 raw files across `Data/{train,valid,test}/normal/`. Base-ID stripping
(` - Copy`, ` (N)`) alone is **not** a safe dedup here — the assumption in the
task brief does not fully hold:

* Kaggle's own train/valid/test splits share base IDs (`10`, `11`, `12`, `6`,
  `7`, `8` appear in two or three splits) — the published split is itself
  patient-leaky.
* Some files sharing a base ID are *not* the same image (base `13`, `16`, `19`,
  `normal` have intra-base phash Hamming distances of 24–28).
* Some files with *different* base IDs *are* the same image (`6` ≈ `n8`,
  Hamming 0; `12` ≈ `4`, Hamming 2).

So instead of trusting filenames, the pool was clustered with the project's own
method — phash first pass (Hamming ≤ 12) → SSIM confirmation (≥ 0.90) →
union-find — unioned with the base-ID heuristic as a second signal.
Script: [`eval_run/ct_normal_ext_cluster.py`](ct_normal_ext_cluster.py),
output: [`eval_run/ct_normal_ext_groups.json`](ct_normal_ext_groups.json).

| | count |
|---|---|
| raw files | 215 |
| SSIM-confirmed near-duplicate pairs | 809 |
| **distinct patient-proxy groups** | **28** |
| dropped as unusable | 1 (group 23) |
| **usable distinct groups** | **27** |

Group 23 (`004007_01_01_519.png`) was dropped because it is **not a chest CT at
all** — it is a pelvis/abdomen slice in a soft-tissue window with no lung fields,
mislabelled into the `normal` folder.

**Premise check (the point of the experiment):** verified visually, not assumed —
contact sheet at [`eval_run/ct_ext_normal_groups_contact.png`](ct_ext_normal_groups_contact.png).
The 27 groups genuinely span different slice levels (carina/upper mediastinum
through mid-heart to lung base), different reconstruction fields of view, and
both lung and mediastinal windowing. They are *not* another single homogeneous
template. Caveat worth recording: roughly a third of them carry **annotation
artifacts** — thin diagonal leader lines drawn over the image, evidently from
figure captions in whatever publication they were scraped from.

**New observation about the incumbent training data.** A matching contact sheet
of `data/ct_scan/train/Normal` ([`eval_run/ct_iqoth_normal_contact.png`](ct_iqoth_normal_contact.png))
shows something the root-cause writeup did not capture: IQ-OTH/NCCD Normals *do*
span slice levels reasonably well, but every single one is a full-field-of-view
512×512 frame **with the curved scanner couch visible along the bottom edge**.
The external images are all cropped to the thorax with no couch. So the
homogeneity is less about slice level than about **field of view and acquisition
framing** — which lines up with the table-shortcut finding already recorded in
PROJECT_STATUS.md §6 (Normal was 100% table-visible there).

### 1b. Second source — 27 distinct patients was too thin

27 groups against 333 existing Normal training images is ~8%; not enough to be a
meaningful intervention. A second, independent Kaggle source was added:

**`maedemaftouni/large-covid19-ct-slice-dataset`**, the curated `1NonCOVID`
(normal) class — 6,893 slices. Not IQ-OTH/NCCD-derived, and not on the exclusion
list. Filenames carry explicit patient/study IDs, giving real patient grouping,
and it is itself an aggregation of three independent sub-sources:

| sub-source | distinct patients |
|---|---|
| `6_Rahimzadeh_*` | 274 |
| `16_Morozov_*` (MosMedData) | 254 |
| `N###` series | 76 |
| **total** | **604** |

Spot-checked visually ([`eval_run/ct_maftouni_normal_contact.png`](ct_maftouni_normal_contact.png)):
genuine normal chest CTs, spanning apex to diaphragm, multiple windows and
reconstruction FOVs, couch present in some and absent in others.

Sampling was capped deliberately — using all 6,893 would have made Normal swamp
Benign (96 train) and Malignant (400 train), and 75% of it would have been one
sub-source. Instead a fixed **patient** quota per sub-source, 4 evenly-spaced
slices per patient.

---

## 2. Held-out Group B, reserved before any training data was built

Split by patient/study **first**, in
[`scripts/ct_normal_pool_split.py`](../scripts/ct_normal_pool_split.py)
(manifest: [`eval_run/ct_normal_pool_manifest.json`](ct_normal_pool_manifest.json)).

| | Group A (train pool) | Group B (held out) |
|---|---|---|
| mohamedhanyyy | 18 groups / 18 images | 9 groups / 99 images |
| maftouni | 110 patients / 405 images | 44 patients / 165 images |
| **total** | **128 patients / 423 images** | **53 patients / 264 images** |

Group B's mohamedhanyyy half is deliberately **exactly the 9 patient groups that
the 9 originally-failing external Normal images belong to** — so the before/after
comparison lands on the documented failure set itself, verbatim. Because that
half is the known-hard set and therefore a biased sample, the maftouni half is a
seeded random patient sample from the same shuffled order as Group A, giving an
unbiased second held-out set alongside it.

Verified programmatically:

* Group A ∩ Group B by SHA-256 content hash: **0**
* Group A ∩ Group B by patient/study ID: **0** (110 vs 44 maftouni, 18 vs 9 mohamedhanyyy)
* Group A ∩ `data/ct_scan/` by content hash: **0**
* Group B ∩ `data/ct_scan/` by content hash: **0**
* All 9 original external Normal images present in Group B: **9/9**; leaked into Group A: **0**

Group B lives at `eval_run/ct_normal_heldout/` and was never passed to the
leakage audit alongside the training data — deduping it against train would
destroy the exact thing it measures.

---

## 3. Combined training set

Built by [`scripts/prepare_ct_scan_normal_augmented.py`](../scripts/prepare_ct_scan_normal_augmented.py)
(new file; `prepare_ct_scan.py` and `ct_resplit_grouped.py` are unmodified) into
`data/ct_scan_normal_augmented/`. Two deliberate departures from
`ct_resplit_grouped.py`, both documented in that script's docstring:

1. **Existing images keep their existing split assignment verbatim.**
   `data/ct_scan/` is already the output of the leakage-aware resplit *and* the
   61-image protocol-mismatch exclusion; rebuilding from `raw/` would discard
   both. It also keeps the new test split byte-identical to the original one,
   which is what makes the no-regression check in §5 apples-to-apples.
2. **Group A goes to train and val only, never test.** The honest external test
   is Group B; putting Group A in test would inflate test accuracy with the
   distribution just trained on.

Group A itself is assigned **group-aware** (patient-proxy group indivisible,
largest-first greedy 85/15 train/val balancing) — the same constraint
`ct_resplit_grouped.py` enforces.

| split | Benign | Malignant | Normal | total |
|---|---|---|---|---|
| train | 96 | 400 | 693 (333 + 360 new) | 1,189 |
| val | 12 | 50 | 105 (42 + 63 new) | 167 |
| test | 12 | 50 | 41 (unchanged) | 103 |

Benign and Malignant are byte-identical to `data/ct_scan/`.

### Leakage audit — and it caught a real bug

[`scripts/ct_leakage_audit_dir.py`](../scripts/ct_leakage_audit_dir.py) (a
parameterised copy of `ct_leakage_audit.py`, which is unmodified) over the whole
combined set.

**First run: 6 clusters spanning train/val, 12 images.** All six were maftouni
Rahimzadeh pairs like
`..._normal4_patient171_SR_4_IM00017.png` vs `..._normal5_patient171_SR_4_IM00017.png`
— i.e. the `normalN` token is a **batch/folder number, not part of the patient
identity**, so patient 171 was being treated as two patients and split across
train and val. Patient-ID extraction was corrected to key on the `patientNNN`
token alone (in both the pool-split and prepare scripts), and the pool was
rebuilt.

**Second run: clean.**

| | combined set | maftouni-only ablation set |
|---|---|---|
| images audited | 1,459 | 1,441 |
| near-duplicate clusters (size ≥ 2) | 117 | 117 |
| **clusters spanning train/val/test** | **0** | **0** |
| clusters spanning > 1 class | 0 | 0 |

Full reports: [`eval_run/ct_leakage_audit_normal_augmented.json`](ct_leakage_audit_normal_augmented.json),
[`eval_run/ct_leakage_audit_normal_aug_mfonly.json`](ct_leakage_audit_normal_aug_mfonly.json).

---

## 4. Training

`ml/train.py` **unchanged**, same methodology as the incumbent: densenet121
(matching the loaded checkpoint's `backbone` field), 15 epochs, 2 frozen-backbone
warmup epochs at lr 1e-3 then fine-tune at 1e-4, batch 32, seed 42,
inverse-frequency weighted CrossEntropyLoss. Only `--data_dir` changed.

Inverse-frequency weights came out Benign 4.13 / Malignant 0.99 / Normal 0.57
(vs. a more balanced split before), a direct consequence of doubling Normal.

| run | data dir | checkpoint | best val_acc | best epoch |
|---|---|---|---|---|
| main | `data/ct_scan_normal_augmented/` | `checkpoints/ct_best_model_phase36_normal_fix.pt` | 0.9940 | 13 |
| ablation | `data/ct_scan_normal_aug_mfonly/` | `checkpoints/ct_best_model_phase36_normal_fix_mfonly.pt` | 1.0000 | 9 |

Both converged cleanly, no collapse of any class, train and val loss tracking
together throughout — nothing pathological in the curves.
Histories/curves: `ml_logs/ct_phase36_normal_fix_{history.json,curves.png}` and
`ml_logs/ct_phase36_normal_fix_mfonly_{history.json,curves.png}`.

`checkpoints/ct_best_model_phase35_v2.pt` was not touched.

### Reproduction check

Same logic as `backend/models.py::load_model_bundle` (checkpoint's claimed
`val_acc` vs. freshly computed val accuracy, 0.001 tolerance), run standalone in
[`eval_run/ct_ckpt_eval.py`](ct_ckpt_eval.py) against each checkpoint's own val
data:

| checkpoint | claimed | reproduced | diff | |
|---|---|---|---|---|
| `ct_best_model_phase35_v2` (live, vs `data/ct_scan`) | 0.9904 | 0.9904 | 0.0000 | PASS |
| `ct_best_model_phase36_normal_fix` (vs `data/ct_scan_normal_augmented`) | 0.9940 | 0.9940 | 0.0000 | PASS |
| `ct_best_model_phase36_normal_fix_mfonly` (vs `data/ct_scan_normal_aug_mfonly`) | 1.0000 | 1.0000 | 0.0000 | PASS |

---

## 5. Validation — before vs. after

All counts, all runs from [`eval_run/ct_ckpt_eval.py`](ct_ckpt_eval.py) with
`ml/dataset.py::build_transforms(train=False)` — the same preprocessing the live
`/predict/ct` route uses. Raw per-image output:
[`results_ct_retrain_before.json`](results_ct_retrain_before.json),
[`results_ct_retrain_after.json`](results_ct_retrain_after.json),
[`results_ct_retrain_after_mfonly.json`](results_ct_retrain_after_mfonly.json).

| | BEFORE `phase35_v2` | AFTER `phase36_normal_fix` | ABLATION `..._mfonly` |
|---|---|---|---|
| **`data/ct_scan/test` Benign** | 10/12 (83.3%) | **12/12 (100%)** | **12/12 (100%)** |
| **`data/ct_scan/test` Malignant** | 50/50 (100%) | 50/50 (100%) | 50/50 (100%) |
| **`data/ct_scan/test` Normal** | 41/41 (100%) | 40/41 (97.6%) | 39/41 (95.1%) |
| **`data/ct_scan/test` overall** | 101/103 (98.1%) | **102/103 (99.0%)** | 101/103 (98.1%) |
| **External Normal, original 9** (Group B) | 0/9 | **7/9** | 5/9 |
| **Group B mohamedhanyyy** (99 imgs, 9 patients) | 0/99 | **85/99** | 59/99 |
| **Group B maftouni** (165 imgs, 44 patients) | 0/165 | **165/165** | 165/165 |
| **Group B total** (264 imgs, 53 patients) | **0/264 (0%)** | **250/264 (94.7%)** | 229/264 (86.7%) |
| **External Malignant** (12 imgs) | **12/12 (100%)** | **0/12 (0%)** | **0/12 (0%)** |

The earlier `EXTERNAL_EVAL_SUMMARY.md` figures of 10/12, 25/25, 25/25 were a
25-per-class sample of the same test split; the full-split BEFORE numbers above
(10/12, 50/50, 41/41) reproduce it exactly.

### What actually happened

**The Normal fix worked, emphatically.** 0/264 → 250/264 on external Normal
images from patients never seen in training, including 7 of the 9 documented
failures — and two of those were the *confidently* wrong ones
(`normal_10` Malignant 0.93 → Normal 0.93; `normal_23` Malignant 0.69 → Normal
1.00). The preprocessing-only attempt recovered 1 of 9; this recovers 7. On the
maftouni half it is 165/165. In-dataset accuracy did not regress — it slightly
improved (98.1% → 99.0%), with Benign going 10/12 → 12/12.

**And it broke the one thing that must not break.** External Malignant went
12/12 → **0/12**, and not narrowly: *all twelve* are now predicted **Normal**,
mostly with high confidence (0.39–0.99, median ~0.90), with P(Malignant)
collapsed to 0.001–0.245. These are adenocarcinoma, large-cell and squamous-cell
cases. In a lung-cancer screening tool this is the worst possible direction of
error, and it is disqualifying on its own.

**The ablation says the cause is not what it first looks like.** The obvious
explanation is contamination: the 18 mohamedhanyyy Normals in training taught the
model "this source's rendering style = Normal", and since every Malignant image
in training is IQ-OTH/NCCD, source and label were perfectly correlated. So the
ablation retrained with **maftouni Normals only** — zero mohamedhanyyy images
anywhere in training, making the external Malignant set a fully unseen source
again.

External Malignant is **still 0/12**, still all predicted Normal.

That rules out simple same-source contamination and points somewhere worse: the
CT model was never detecting lesions across domains in the first place. It was
keying on acquisition/framing style. Before, "not IQ-OTH-styled ⇒ Malignant" —
which is why external Malignant scored a flattering 12/12 and external Normal
scored 0/9 from the *same source and same framing*. After adding any
out-of-domain Normal data, the shortcut simply inverted to "not IQ-OTH-styled ⇒
Normal", and the same 12 images flipped with it. **The original 12/12 was never
evidence of malignancy detection; it was the same shortcut pointing the other
way.** That reading is consistent with the Grad-CAM finding in
`ct_normal_root_cause.md` (attention on mediastinum/couch/border rather than
lesions) and with the table-shortcut finding already in PROJECT_STATUS.md §6.

Secondary observation: the ablation is worse than the main run on Group B
mohamedhanyyy (59/99 vs 85/99) and on the original 9 (5/9 vs 7/9) — as expected,
since the main run had 18 mohamedhanyyy patients in training and the ablation had
none. Its 165/165 on maftouni is therefore partly in-domain credit, not pure
generalization. The main run's 165/165 on maftouni held-out *patients* is the
cleaner of the two Normal results.

---

## 6. Assessment

**This did not fix the problem, and should not be adopted.** Trading external
Malignant 12/12 → 0/12 for external Normal 0/264 → 250/264 is not a win for a
cancer-screening classifier under any weighting: every externally-sourced
malignant case now reads as Normal, confidently.

More usefully, the experiment falsified a premise both earlier writeups were
resting on. `ct_normal_root_cause.md` concluded Malignant "transfers fine
(12/12)" because a lesion is a salient positive feature, and that only Normal was
fragile. The ablation shows Malignant never transferred — both classes ride on
the same acquisition-style shortcut, and the incumbent checkpoint's external
Malignant score is an artifact of which way that shortcut happens to point. The
real defect is broader than the Normal class.

What that implies for any next attempt — none of it done here, listed only so the
adoption decision has the options in front of it:

* More diverse **Normal** data alone cannot work while every Malignant training
  image comes from one source. The source→label correlation has to be broken on
  *both* sides, which needs externally-sourced Malignant training data
  (`mohamedhanyyy`'s three carcinoma subtypes are an obvious candidate, though
  using them removes that set as an external test and something else would have
  to take its place).
* Reporting external Malignant accuracy for the live model as evidence of
  generalization should stop regardless of what is adopted — the ablation shows
  that number does not mean what it appears to mean.
* Domain-invariance measures (source-balanced batches, aggressive
  framing/FOV augmentation, or a lung-field crop applied before classification,
  as already done for X-ray) are the more plausible direction than more data
  alone.

## 7. Artifacts

Experimental only — nothing below is referenced by `backend/main.py`.

| path | what |
|---|---|
| `checkpoints/ct_best_model_phase36_normal_fix.pt` | main retrained checkpoint |
| `checkpoints/ct_best_model_phase36_normal_fix_mfonly.pt` | ablation checkpoint |
| `data/ct_scan_normal_augmented/` | combined training set |
| `data/ct_scan_normal_aug_mfonly/` | ablation training set |
| `datasets/ct_normal_extra/groupA/` | Group A train pool (18 + 405 images) |
| `eval_run/ct_normal_heldout/` | **Group B, held out** (99 + 165 images) |
| `scripts/ct_normal_pool_split.py` | source dedup + A/B patient split |
| `scripts/prepare_ct_scan_normal_augmented.py` | combined-dataset build |
| `scripts/ct_leakage_audit_dir.py` | leakage audit, parameterised by data dir |
| `eval_run/ct_normal_ext_cluster.py` | phash+SSIM clustering of the mohamedhanyyy pool |
| `eval_run/ct_ckpt_eval.py` | standalone checkpoint evaluator (repro check + test + external groups) |
| `eval_run/ct_normal_pool_manifest.json` | A/B split provenance |
| `eval_run/ct_normal_augmented_build*.json` | build counts + per-group split assignment |
| `eval_run/ct_leakage_audit_normal_aug*.json` | leakage audit reports |
| `eval_run/results_ct_retrain_{before,after,after_mfonly}.json` | per-image predictions |
| `eval_run/ct_{ext_normal_groups,iqoth_normal,maftouni_normal}_contact*.png` | visual spot-checks |
| `ml_logs/ct_phase36_normal_fix*_{history.json,curves.png}` | training curves |

Raw downloads (untracked, ~2.3 GB) live under
`eval_run/external_samples/_downloads/` — `ctscan/` (mohamedhanyyy, pre-existing)
and `large-covid19-ct-slice-dataset.zip` + `covidct_slice/1NonCOVID/` (maftouni,
added by this experiment).

Stopping here as instructed — no integration, no deployment, awaiting an adoption
decision.
