# Phase G.1: multi-slice stacks — data extraction and architecture setup

**Scope: extraction + architecture code + a forward-pass-only smoke test. NO TRAINING.**
No optimizer was constructed, no backward pass was run, no checkpoint was written. Everything
written is confined to `eval_run/phaseG1/` and `eval_run/phaseG1_setup.md`.
**Phase B's `labels_manifest.csv` and its 3,461 PNGs are untouched** (re-hashed in §7 — digest
still `95d734f64123a46f`, 3,461 rows), as are `checkpoints/ct_best_model_phase35_v2.pt`,
`backend/main.py`, `v1`, `v2_jitter` and `v3_jitter_loosebenign`. No dependency added, nothing
deployed, **no process left running** (verified: zero `python.exe`).

Why this phase exists: Phase F closed post-hoc ensembling as answer **(c)** — the true centre
slice's vote is present in the window and gets outvoted, at every displacement. The one piece of
evidence that cut the other way was that the ±1 slice is **not blind**: 0.601 accurate on its own
and **agreeing with the true-centre-slice prediction 72.6%** of the time. A model that takes the
stack as its *actual input* could learn to weight that; an average that must treat all N slices as
equally authoritative structurally cannot. Phase G.1 builds the data and the code for that model.
It does not build the model's answer.

---

## Verdict up front

1. **3,461 / 3,461 stacks extracted — every Phase B crop, none lost.** 31 s of extraction on top
   of 29.7 min of download. §3.
2. **The centre channel of all 3,461 stacks is bit-identical to Phase B's stored single-slice
   PNG — max abs difference 0, checked per row, not sampled.** The z-axis is the only thing that
   changed. §3.2.
3. **The architecture needs NO change, and this is a checked fact rather than an assumption.**
   `ml/model.py`'s DenseNet-121 first conv is `Conv2d(3, 64, k=7, s=2)` with weight `[64, 3, 7, 7]`,
   and `v2_jitter`'s own trained checkpoint carries the same `[64, 3, 7, 7]`. 3 channels in,
   3 channels expected. **Nothing was added.** §5.
4. **But the change that *does* matter is in the data pipeline, not the model, and it is a one-line
   deletion that is easy to get catastrophically wrong.** `ml/dataset.py::build_transforms` begins
   with `Grayscale(num_output_channels=3)`, which would collapse three genuinely different slices
   into one luminance channel and re-duplicate it — producing a model that looks trained but saw a
   slightly blurred single slice. That op is removed, and a runtime guard fails loudly if any stack's
   channels are identical. §5.2.
5. **The three channels really are a different input to the network.** The ImageNet-pretrained first
   conv's per-channel filters are *not* identical (cosine similarity 0.65 / 0.36 / 0.60 between
   channel pairs), and feeding a real 3-slice stack versus the same centre slice duplicated three
   times moves the logits by **0.69** on a random val row. Repurposing the pretrained weights is a
   real repurposing, not a no-op. §5.3.
6. **Smoke test PASSES**: batch `[8, 3, 224, 224]` → output `[8, 3]`, no NaN, no Inf, softmax rows
   sum to 1, peak VRAM **0.194 GB of 4.00** at batch 16. §5.4.
7. **The genuine edge case is NOT the one the brief anticipated.** Across the 45 series Phase D/E.1
   downloaded in full, **zero** manifest crops sit on the first or last slice of a series — a chest
   CT's end slices are above/below the lungs, so an annotated nodule is never on one. The real
   missing-neighbour cause is **annotation coverage**: 646 of 6,922 sides (9.3%) have no contoured
   neighbour to name. 82.5% of stacks have three real slices, 16.4% have two, 1.1% have one. §4.
8. **That gap is class-correlated and it is a shortcut risk, stated plainly.** Malignant stacks are
   93.8% complete, Benign 77.2%, Normal 81.7% — a 16.6-point Malignant-vs-Benign gap, the same
   direction as Phase B §3's unresolved lung-fraction bias. Because the fallback duplicates the
   centre channel, "are my channels identical" is a weak predictor of class. **Phase G.2 should run
   `--min_real_slices 3` as its primary configuration**, which is one flag and costs 607 crops. §4.3.
9. **The route to slice adjacency is new and had to be validated, not assumed.** NBIA's
   `getSOPInstanceUIDs` returns UIDs in **effectively random order** (measured), and there is no
   per-instance metadata endpoint (`getDicomTags` returns HTTP 500 for every spelling tried). Slice
   order was recovered from the LIDC annotation XMLs' `imageZposition`, then checked against 45
   fully-downloaded series: **189 of 189 proposed neighbours exact, 0 wrong.** §2.
10. **Cost came in under the brief's ~2× Phase B budget**: 3,386 slices fetched in **29.7 min at
    1.90 slices/s, 0 failures, 0 retries** against Phase B's 1,999 in 21.1 min at 1.54 slices/s.
    1.69× the slices in 1.41× the time. **No whole-volume download was performed.** §3.1.

**Phase G.2 (training) is NOT started.** The exact command is in §6, with a ~10 min estimate.

---

## 1. What was reused, and what is new

Nothing in the crop maths was reimplemented — that is the standing rule since Phase A's coordinate
bug.

| reused, verbatim by import | from |
|---|---|
| `to_hu`, `window`, `box_for`, `OUT_PX`, and `save_png`'s exact resize pipeline | `lidc_full/extract_crops_full.py` |
| slice ordering by `ImagePositionPatient` projected on the `ImageOrientationPatient` normal | `phaseD/volume_io.py`'s rule (never filenames, never `InstanceNumber`) |
| SOPInstanceUID-verified DICOM read (`read_slice`) | `lidc_full/extract_crops_full.py`, widened to two cache dirs |
| targeted `getSingleImage` fetch: 3 retries, backoff, atomic `.part`, cache-aware | `lidc_full/download_slices_full.py` |
| patient-disjoint `split` column, class order `[Benign, Malignant, Normal]`, inverse-frequency weights, the whole Phase C.1 training recipe | `lidc_full/labels_manifest.csv`, `phaseC1/train_lidc_crops_jitter.py` |

**Splits were not reshuffled and could not have been** — every stack row carries Phase B's own
`split`, `label`, `patient_id` copied through, and all 3,461 were re-checked against Phase B's
manifest: **0 rows disagree on label/split/patient, 0 Phase B rows without a stack.** Patient
disjointness re-derived from the new manifest: `train∩val = 0`, `train∩test = 0`, `val∩test = 0`
(410/195/192 patients), and it still holds on the `min_real_slices=3` subset (390/172/162).

**The only genuinely new operation** is holding the integer pixel box `(r0, c0, r1, c1)` fixed while
varying the slice — plus §2, which is new because the cheap routes did not exist.

---

## 2. Getting slice adjacency without downloading 797 volumes

Phase D/E got slice order by downloading whole series and sorting on `ImagePositionPatient`. That
cost ~2 h for 35 patients; at 804 series it is not affordable and it is **not necessary** — this
phase needs 2 extra slices per crop, not every slice in a scan. Three routes were tried in
increasing order of cost. The first two are dead ends and are recorded so the next session does not
retry them.

### 2.1 Is NBIA's UID list already in slice order? **No — measured, not guessed.**

`probe_order.py` compares `getSOPInstanceUIDs`'s list against the true geometric order for 6 series
that are on disk in full:

| series | slices | API order == geometric | == reversed | fraction of API-adjacent pairs that are geometrically adjacent |
|---|---|---|---|---|
| LIDC-IDRI-0059 | 133 | no | no | **0.000** |
| LIDC-IDRI-0182 | 133 | no | no | 0.023 |
| LIDC-IDRI-0321 | 265 | no | no | 0.008 |
| LIDC-IDRI-0433 | 305 | no | no | 0.007 |
| LIDC-IDRI-0586 | 120 | no | no | 0.017 |
| LIDC-IDRI-0648 | 764 | no | no | 0.001 |

**The list is effectively shuffled.** Any pipeline that reads adjacency off it is silently wrong.

### 2.2 Is there a per-instance metadata endpoint? **No.**

`probe_api.py`: `v1/getDicomTags`, `v2/getDicomTags` (three parameter spellings) all return
**HTTP 500**; `v2/getSeriesMetaData` returns empty; `v1/getSeriesMetaData` works but is
**series-level** (945 bytes, no per-instance position). A `getSingleImage` call is 526 KB / 2.8 s
cold. So there is no way to learn a slice's position short of downloading it.

### 2.3 The route that works: the annotation XMLs already carry z

Every `<roi>` in a LIDC-IDRI annotation XML carries **both** `<imageSOP_UID>` and
`<imageZposition>`, for nodule and non-nodule marks, from all four readers. Phase A downloaded all
1,318 of these XMLs; they are on disk and free to re-read. `build_slice_index.py` unions them into
`series_uid -> {sop: z}` (807 series parsed, 0 unreadable), estimates the slice pitch as the
**modal positive gap between consecutive distinct z values**, and names the neighbours at
`z0 ± pitch`.

**The failure mode this has** is a sparsely-contoured series whose modal gap is a *multiple* of the
true pitch — which would make "±1" silently mean "±2". Two independent checks close it.

**Check A — `check_spacing.py`, all 1,989 centre slices, local and free.** Compare the XML pitch to
the centre DICOM's own `SpacingBetweenSlices`/`SliceThickness` (already on disk from Phase B):

| | n | share |
|---|---|---|
| XML pitch matches the DICOM tag (±5%) | 1,321 | 66.4% |
| XML pitch **smaller** than the tag (ratio 0.5 in 402 cases, 0.8–0.95 in 258) | 664 | 33.4% |
| XML pitch **larger** than the tag | **3** | **0.2%** |
| no usable reference | 1 | 0.1% |

The 664 are **not** errors: LIDC is full of overlapping reconstructions — 2.0 mm thick every 1.0 mm,
1.25 mm thick every 0.62 mm — and **none of the 200 sampled carried a `SpacingBetweenSlices` tag at
all**, so `SliceThickness` is simply the wrong reference for them. Check B settles it directly. The
**3 larger-than-reference cases have dz support of only 1–2 gaps** and are the real risk, so
`build_slice_index.py` now rejects any pitch exceeding 1.05× the DICOM bound and those 3 centre
slices (6 manifest rows) fall back to the padded path rather than carrying a wrong slice.

**Check B — `validate_index.py`, ground truth on 45 series.** Phase D (10) and Phase E.1 (35)
downloaded whole series, so the true order is on disk. For the 107 manifest centre slices in those
series:

| | count |
|---|---|
| **neighbours XML proposed that are exactly the true ±1 slice** | **189 / 189 (100.00%)** |
| neighbours XML proposed that are wrong | **0** |
| sides with no XML neighbour (annotation gap) | 25 |
| sides that are a **true first/last slice of the series** | **0** |

**62 of those 107 centre slices are in the "XML pitch smaller than `SliceThickness`" bucket** — i.e.
the overlapping-reconstruction case — and all of their neighbours are exact. That is the direct
evidence that the XML pitch, not `SliceThickness`, is the series' real slice pitch.

**Cost of this route: 0 extra API calls.** The per-series `getSOPInstanceUIDs` call the brief
budgeted for turned out to be useless (§2.1) and is not made during extraction.

---

## 3. Extraction

### 3.1 Download

3,461 crops sit on **1,989 unique centre slices** (Normal crops share their nodule slice), so the
neighbour requirement is ~2 × 1,989, not 2 × 3,461. **140 of the needed neighbours were already in
Phase B's cache** (one crop's neighbour is another crop's centre) and were reused in place, never
refetched.

| | Phase B (§4) | **Phase G.1** |
|---|---|---|
| slices requested | 1,999 | **3,386** (3,526 needed − 140 reused) |
| ok / failed | 1,997 / 2 | **3,386 / 0** |
| needed a retry | 0 | **0** |
| bytes | 1,051 MB | **1,782 MB** |
| wall time | **21.1 min** | **29.7 min** |
| throughput | 1.54 slices/s (6 workers) | **1.90 slices/s (8 workers)** |

**1.69× Phase B's slice count in 1.41× its time, inside the brief's "~2× Phase B" budget, with a
better failure rate than Phase B's own run** (Phase B lost 2 slices to permanent HTTP 500s; this run
lost none). **No whole-volume download was performed** — the Phase D/E route was avoided entirely,
which is the difference between 30 minutes and an afternoon.

One bookkeeping note: the download was launched against the 3,526-entry list produced *before* the
§2.3 pitch guard trimmed it to 3,525, so **exactly 1 of the 3,386 downloaded slices is unused**.

### 3.2 Stacking, and the check that makes it trustworthy

For each manifest row: read the centre DICOM, compute the box with **Phase B's own `box_for`** at
that row's stored `center_row`/`center_col`/`px_mm`/`crop_mm`, then apply **that same integer box**
to the ±1 slices. The channel order is `[centre−1, centre, centre+1]` — a 3-channel 128×128 8-bit
PNG per crop, one slice per channel, RGB-shaped and ready for a standard 3-channel CNN.

> **Format choice.** PNG, not `.npy`: it is lossless for the uint8 the windowing already produces
> (`window()` → ×255 → uint8 → BILINEAR resize is Phase B's exact chain, stopped one step short of
> `save_png`'s write), it is 66 MB instead of 170 MB, and it can be opened and looked at — which
> §4.2 does. The risk it carries (a 3-channel PNG can be silently re-grayscaled) is handled in §5.2.

| | result |
|---|---|
| manifest rows in | 3,461 |
| **stacks written** | **3,461** |
| rows skipped for any reason | **0** |
| **centre channel bit-identical to Phase B's PNG** | **3,461 / 3,461, max abs difference 0** |
| unique stack paths | 3,461 (no collisions) |
| extraction wall time | **31 s** |

The centre-channel check is run **on every row**, not a sample: if the box maths or the HU windowing
had drifted by one pixel or one grey level anywhere, it would show here. It does not.

### 3.3 Counts, Phase B's own reporting style

| split | patients | crops | Malignant | Benign | Normal |
|---|---|---|---|---|---|
| train | 410 | 2,450 | 412 | 648 | 1,390 |
| val | 195 | 506 | 75 | 133 | 298 |
| test | 192 | 505 | 75 | 132 | 298 |
| **total** | **797** | **3,461** | **562** | **913** | **1,986** |

Identical to Phase B §5 row for row — which is the point: this is the same dataset with a z extent,
not a new one. 804 series, 797 patients, unchanged.

---

## 4. The edge case — what it actually is

### 4.1 It is annotation coverage, not the ends of the scan

The brief anticipated nodules on the first or last slice of a series. **Measured across the 45
fully-downloaded series: that happens 0 times** (§2.3, Check B), and the reason is anatomical — a
chest CT's first and last slices are above and below the lungs, so no annotated nodule can sit on
one. The real cause of a missing neighbour is that **no radiologist contoured the adjacent slice**,
so it never appears in the XML. All 646 missing sides are of this kind:

| per-side status (2 sides × 3,461 stacks = 6,922) | n |
|---|---|
| real adjacent slice used | **6,276 (90.7%)** |
| no contoured neighbour in the XML | 646 (9.3%) |
| DICOM missing / unreadable / UID mismatch | **0** |
| in-plane shape mismatch between slices | **0** |

### 4.2 The fallback, and why this one

**A missing channel repeats the CENTRE slice**, not the opposite neighbour. Repeating the opposite
neighbour would put the same slice in channels 0 and 2 and silently invert that row's channel
ordering; repeating the centre degrades the row smoothly toward Phase B's exact single-slice input,
which is the correct behaviour when a neighbour is unavailable. Every affected row is flagged in the
manifest (`n_real_slices`, `pad_prev`, `pad_next`, `prev_status`, `next_status`) so Phase G.2 can
filter, weight or ablate rather than guess.

| stacks by real slice count | n | share |
|---|---|---|
| **3 real slices** | **2,854** | **82.5%** |
| 2 real (one side padded) | 568 | 16.4% |
| 1 real (both sides padded — identical to Phase B's crop) | 39 | 1.1% |

### 4.3 The completeness gap is class-correlated — say it plainly

| | n | 3 real slices | share complete |
|---|---|---|---|
| **Malignant** | 562 | 527 | **93.8%** |
| **Benign** | 913 | 705 | **77.2%** |
| **Normal** | 1,986 | 1,622 | **81.7%** |

**A 16.6-point Malignant-vs-Benign gap**, and the mechanism is obvious: malignant nodules average
18.5 mm (Phase B §3) so they appear on — and get contoured on — many more slices than an 8.1 mm
benign one, and a Normal crop inherits its nodule slice's coverage. It is the same *direction* as
Phase B §3's unresolved lung-fraction bias, and it is the same kind of problem.

**Why this matters more here than it would elsewhere.** This investigation exists because of
shortcut learning. If padded rows are kept, "are channels 0 and 1 identical" is a feature that
correlates with class, and a DenseNet is entirely capable of finding it. **Recommendation for Phase
G.2: run `--min_real_slices 3` as the primary configuration.** It costs 607 crops (3,461 → 2,854),
keeps patient disjointness (390/172/162 patients, verified), and removes the correlated artifact
outright:

| min_real_slices=3 | train | val | test |
|---|---|---|---|
| Malignant | 390 | 73 | 64 |
| Benign | 512 | 97 | 96 |
| Normal | 1,156 | 237 | 229 |
| **total** | **2,058** | **407** | **389** |

The all-rows run remains available as a one-flag ablation. **This phase does not make the call; it
makes the flag exist and the number visible.**

### 4.4 The stacks were read, not asserted

`qc_stacks.py` — 9 stacks across all three classes, with a quantitative alignment control and a
contact sheet (`qc_stack_sheet.png`) that **was opened and looked at**.

The control is the instrument that matters: correlate each neighbour channel with the centre, and
compare against the correlation of the centre with a crop taken **at the same x/y box from a slice
≥6 slices away in the same series**. If the index were off, or the box were re-derived per slice,
near and far would look alike.

| | value |
|---|---|
| mean corr(±1 channel, centre) | **0.824** |
| min corr(±1 channel, centre) | 0.325 |
| **mean corr(far slice, centre) — control** | **0.208** |
| any ±1 channel equal to the centre | **none** |
| centre channel identical to Phase B's PNG | 9/9 |

**Read honestly, 9/9 anatomically coherent:** in `LIDC-IDRI-0546` (dz 1.0 mm) the difference map is
a thin ring at the lesion boundary — exactly what a 1 mm step through a spherical mass produces. In
`LIDC-IDRI-0141` and `LIDC-IDRI-0662` the pleural band and rib are **pixel-locked across all three
channels** while the nodule fades — direct visual proof that x/y is held fixed and only z varies,
and the signature an off-by-one or a re-derived box would destroy. `LIDC-IDRI-0749` (dz 3.0 mm)
shows the mass genuinely changing shape, which is the through-plane signal this phase is here to
capture. **The two low-correlation rows are real and are not defects:** `LIDC-IDRI-0686 n8` (0.32)
is the faint ground-glass lesion Phase B §6b already flagged as marginal, and `LIDC-IDRI-0012`
(0.38) is Normal parenchyma whose crop is mostly fine vessel texture — in both, correlation is
dominated by noise while the anatomy stays in place, and both still sit far above their far-slice
controls (0.11 and 0.24).

---

## 5. Architecture — the answer is "no change", and it was checked

### 5.1 The layer shapes, read off the actual modules

`smoke_test.py` constructs `ml/model.py`'s `build_model("densenet121", 3, pretrained=True)` and
reads the first convolution:

```
densenet121 features.conv0 : Conv2d, weight [64, 3, 7, 7], in_channels=3, out_channels=64,
                             kernel 7x7, stride 2, bias=False
classifier out_features    : 3
ct_lidc_nodulecrop_v2_jitter.pt  features.conv0.weight : [64, 3, 7, 7]   (opened read-only)
```

**3 channels in, 3 channels expected — and the trained `v2_jitter` checkpoint's own first-conv
weight has the identical shape.** `architecture_change_needed: false`. **No layer was added, no
weight was reshaped, no `in_channels` was patched.** `ml/model.py` is unmodified (digest recorded in
§7). The pretrained ImageNet first-conv weights are kept exactly as they are, which is the correct
treatment when the channel count is unchanged and only the *semantics* of each channel change.

### 5.2 The change that IS needed is in the data pipeline — and it is a trap

`ml/dataset.py::build_transforms`, which every classifier in this investigation used, begins with:

```python
transforms.Grayscale(num_output_channels=3),
```

Phase B/C's crops are grayscale PNGs, so that op **duplicates** one slice into three identical
channels. Run it on a Phase G.1 stack and it converts three *different* slices to a single luminance
channel and re-duplicates that — **producing a model that trains, converges, reports a number, and
never saw more than one slice.** That is a silent failure that would invalidate Phase G.2 entirely.

`phaseG1/stack_data.py` therefore does **not** import `build_transforms`. It reproduces Phase C's
chain verbatim minus that one op — `Resize(224)`, train-only `RandomHorizontalFlip` /
`RandomRotation(10)` / `ColorJitter(0.1, 0.1)`, `ToTensor`, ImageNet `Normalize` — so a Phase G.2
number stays comparable to `v2_jitter`. `assert_channels_differ()` samples 64 three-real-slice rows
and raises if any stack's channels are identical; it runs inside both the smoke test and the
training script before the first epoch. **64/64 val stacks passed, none collapsed.**

### 5.3 Are three slices actually a different input? Yes, at layer 1.

Worth checking rather than assuming, because if the pretrained conv0 treated its three channels
identically, "slices as channels" would be arithmetically equivalent to averaging them.

| | value |
|---|---|
| conv0 per-channel filter cosine similarity, ch0·ch1 / ch0·ch2 / ch1·ch2 | **0.648 / 0.362 / 0.603** |
| conv0 channel filters identical? | **no** |
| batch per-channel mean after normalize (ch0/ch1/ch2) | −0.344 / −0.208 / −0.062 |
| **max abs logit change: real 3-slice stack vs the same centre duplicated 3×** | **0.690** |

The pretrained filters are **substantially different per channel**, so the network can weight
`centre−1`, `centre`, `centre+1` differently from the first layer onward — which is precisely the
capability Phase F §7 identified as structurally unavailable to a post-hoc average.

### 5.4 Smoke test — forward pass only

| check | result |
|---|---|
| input batch | `[8, 3, 224, 224]`, float32, real val stacks |
| **output shape** | **`[8, 3]`** — expected `[8, 3]` ✔ |
| NaN / Inf in logits | **none / none** |
| softmax rows sum to 1 | ✔ |
| logit range | [−1.007, 0.394] |
| second batch at bs=16 | `[16, 3]`, no NaN |
| **peak VRAM (forward, bs=16)** | **0.194 GB of 4.00** |
| **backward pass / optimizer / checkpoint** | **none — not run, by design** |

**PASS.**

---

## 6. Phase G.2 — the exact next command

Everything is on disk. A next session needs no re-derivation.

```
cd "D:/7th sem/sdp/eval_run/phaseG1"
python -u train_stacks.py --min_real_slices 3 --tag v4_stack3 > _work/train_v4_stack3.log 2>&1
```

`train_stacks.py` is **already written and committed by this phase** (it was not run). It is
`phaseC1/train_lidc_crops_jitter.py` with exactly two differences — the dataset is
`stack_data.LidcStackDataset`, and the in-plane jitter is dropped because Phase C.1's jitter source
is a single oversized slice with no z extent (re-deriving a 3-slice jitter source is a Phase G.2+
decision, not a Phase G.1 one). Everything else is deliberately identical to Phase C.1 so the result
is directly comparable to `v2_jitter`: ImageNet-init densenet121 via `ml/model.py`, 2-epoch frozen
warmup at lr 1e-3 then fine-tune at 1e-4 for 15 epochs total, Adam, seed 42, batch 8, AMP,
inverse-frequency weighted `CrossEntropyLoss`, best-on-val-balanced-accuracy checkpointing, Phase B's
`split` column used as-is.

**Arguments** (all optional; the defaults are the ones above): `--min_real_slices {1,3}` (3 = the
2,058/407/389 complete-stack subset recommended in §4.3; 1 = all 3,461 rows including padded),
`--epochs 15`, `--warmup_epochs 2`, `--batch_size 8`, `--lr 1e-3`, `--finetune_lr 1e-4`, `--seed 42`,
`--amp 1`, `--num_workers 0`, `--manifest`, `--ckpt`, `--tag`.

**It will produce:**

| path | what |
|---|---|
| `checkpoints/ct_lidc_nodulecrop_v4_stack3.pt` | best val balanced accuracy |
| `phaseG1/ct_lidc_nodulecrop_v4_stack3_valacc.pt` | best val accuracy |
| `phaseG1/training_history_v4_stack3.json` | per-epoch history, args, class weights, peak VRAM |
| `phaseG1/training_curves_v4_stack3.png` | loss / accuracy / balanced accuracy |

**It cannot overwrite an existing checkpoint**: the filename is new, and the script hard-refuses to
start if `--ckpt` names `ct_best_model_phase35_v2.pt`, `ct_lidc_nodulecrop_v1.pt`,
`ct_lidc_nodulecrop_v2_jitter.pt` or `ct_lidc_nodulecrop_v3_jitter_loosebenign.pt`.

**A Phase G.2 that only trains is not an answer.** Phase E.1 §2 showed the whole `v2_jitter` failure
is on the through-plane axis, so the run must be followed by `phaseC1/tolerance_sweep.py`'s
in-plane control **and** a through-plane sweep — a model that gains through-plane tolerance while
losing Phase C.1's in-plane tolerance has bought nothing, and Phase F §3.3 showed that trade is a
real risk. Budget that separately.

### Estimated training time: **~10 minutes** (RTX 3050 4 GB)

| reference | wall | peak VRAM | mean epoch |
|---|---|---|---|
| Phase C (`v1`, 2,450 train crops, 15 epochs) | **9.4 min** | 0.616 GB | 37.4 s |
| Phase C.1 (`v2_jitter`, same, + jitter) | **9.6 min** | 0.616 GB | 38.3 s |
| **Phase G.2 estimate, `--min_real_slices 3`** | **~8–9 min** | **~0.62 GB — unchanged** | ~33 s |
| Phase G.2 estimate, all 3,461 rows | ~9.5–11 min | ~0.62 GB | ~38 s |

**VRAM does not change and neither does batch size.** The tensor entering the network is
`[8, 3, 224, 224]` in both cases — Phase C fed one slice duplicated to 3 channels, Phase G.2 feeds 3
distinct slices into the same 3 channels. Same shape, same architecture, same activations; the
measured forward-pass peak here (0.194 GB at bs=16) is consistent with Phase C's 0.616 GB training
peak. **No batch-size reduction is needed and none should be applied** — changing it would break
comparability with `v2_jitter` for no reason.

The `--min_real_slices 3` run is *faster* than Phase C, not slower: 2,058 train crops instead of
2,450 (−16%), and loading one 3-channel PNG is cheaper than Phase C.1's jitter path (which opened an
oversized PNG and sub-cropped it per sample). The all-rows run lands at roughly Phase C.1's time.
**If a G.2 run takes dramatically longer than ~12 min, something is wrong** — most likely
`num_workers`, or a transform chain that re-grayscales.

---

## 7. Live-system integrity

Re-hashed after all downloads and extraction. sha256, first 16 hex characters — the same formula
Phase C.1/D/E/E.1/F used, which is confirmed by all seven prior digests reproducing exactly.

```
checkpoints/ct_best_model_phase35_v2.pt       sha=93b3a979884edbc4  mtime=1787555730.4296012  MATCHES Phase E §7 / E.1 §6 / F §6
backend/main.py                               sha=780120553178c0a5  mtime=1788684518.831581   MATCHES
checkpoints/ct_lidc_nodulecrop_v1.pt          sha=bdcbd6b2e5312435                            MATCHES
checkpoints/ct_lidc_nodulecrop_v2_jitter.pt   sha=5d60055e02974e91                            MATCHES
eval_run/lidc_full/labels_manifest.csv        sha=95d734f64123a46f  3461 rows                 MATCHES
eval_run/phaseE/pipeline.py                   sha=936427fc20d389a0  mtime=1788946798.4063764  MATCHES Phase F §6
eval_run/phaseE/run_eval.py                   sha=cc487ad5344fc643  mtime=1788946651.7800155  MATCHES Phase F §6

checkpoints/ct_lidc_nodulecrop_v3_jitter_loosebenign.pt  sha=9d1e9d7bf1ff5e16   recorded for the first time
eval_run/lidc_full/extract_crops_full.py                 sha=7ef5dbb039574385   recorded for the first time
ml/model.py                                              sha=12c40c85d70a4eab   recorded for the first time
ml/dataset.py                                            sha=5843c2dc8e36771b   recorded for the first time
```

**ALL MATCH**, and both recorded mtimes are unchanged. `v3_jitter_loosebenign`, `ml/model.py`,
`ml/dataset.py` and Phase B's `extract_crops_full.py` are added to the watch list here because this
phase imports or depends on them — all four were opened **read-only** and none was edited; their
digests are now on record for Phase G.2+. **No checkpoint was written.** Every write this phase is
inside `eval_run/phaseG1/` plus this report. **No new dependency, nothing deployed, no `uvicorn`
started, no process left running** (verified: zero `python.exe`).

Unlike Phase F, **no prior phase's artifact was touched at all** — Phase G.1 has its own
`integrity.py` writing only to `phaseG1/_work/`.

---

## 8. Artifacts

All under `eval_run/phaseG1/` (**1.8 GB, of which 1.7 GB is cached DICOM and 66 MB is the stacks**).
Nothing is referenced by `backend/`.

| path | what |
|---|---|
| **`stacks/{Malignant,Benign,Normal}/`** | **3,461 three-channel 128×128 PNGs, one slice per channel (66 MB)** |
| **`stacks_manifest.csv` / `.json`** | per-stack manifest; references Phase B by `phaseB_image` + `sop_uid`, carries `prev_sop`/`next_sop`/`dz_mm`/`n_real_slices`/`pad_prev`/`pad_next`/`prev_status`/`next_status` and Phase B's `label`/`split`/`patient_id` unchanged |
| `probe_order.py` | §2.1 — is the NBIA UID list in slice order? (no) |
| `probe_api.py` | §2.2 — is there a per-instance metadata endpoint? (no) |
| `build_slice_index.py` | §2.3 — XML `imageZposition` → per-series slice grid, modal pitch, ±1 neighbours, the pitch guard |
| `check_spacing.py` | §2.3 Check A — XML pitch vs the DICOM's own spacing tags, all 1,989 centre slices |
| **`validate_index.py`** | **§2.3 Check B — the derived neighbours against 45 fully-downloaded series: 189/189 exact** |
| `download_neighbours.py` | §3.1 — Phase B's `getSingleImage` fetch, 8 workers, retries, cache-aware |
| **`extract_stacks.py`** | **§3.2 — the stacking, with the per-row centre-channel bit-identity check** |
| `qc_stacks.py`, **`qc_stack_sheet.png`** | §4.4 — the alignment control and the 9 stacks that were read |
| `stack_data.py` | §5.2 — the dataset and the transform chain without `Grayscale`, plus `assert_channels_differ` |
| **`smoke_test.py`** | **§5 — layer shapes, checkpoint shape, channel-filter similarity, forward pass. No backward pass.** |
| **`train_stacks.py`** | **§6 — Phase G.2's entry point. WRITTEN, NOT RUN.** |
| `integrity.py` | §7 |
| `_work/dicom/` | 3,386 neighbour DICOM slices, 1.78 GB |
| `_work/{probe_order,probe_api,slice_index,neighbours,need_slices,check_spacing,validate_index,download_log,extract_log,qc_stacks,smoke_test,integrity}.json` | every number in this report |
| `_work/download.log` | the §3.1 transcript |

Reused read-only and unmodified: `eval_run/lidc_full/{labels_manifest.csv, extract_crops_full.py,
images/, _work/{dicom,plan_full.json}}`, `eval_run/lidc_feasibility/_work/xml/`,
`eval_run/phaseD/{volume_io.py, _work/volumes/}`, `eval_run/phaseE1/_work/volumes/`,
`eval_run/phaseC1/{train_lidc_crops_jitter.py, lidc_data_jitter.py}`, `ml/{model,dataset}.py`,
`checkpoints/ct_lidc_nodulecrop_v2_jitter.pt`.

---

## 9. What this phase does NOT show

* **No model was trained and no accuracy number exists.** Phase F's verdict (c) stands unchanged.
  Nothing here says a 3-slice model will work — Phase E.1 §7's arguments against it (a 74% ceiling
  that is already the binding constraint, the aggregation redesign, the missing guardrail chain,
  ~10 days and an exam) are untouched by this phase.
* **N = 3 only.** A wider stack was not built, and §2.3's route would need re-validating at ±2 (the
  XML coverage gap grows with distance).
* **Still LIDC, still the detector's own lineage.** No generalization claim.
* **9.3% of channel sides are the centre repeated**, and that padding is class-correlated (§4.3).
  The `min_real_slices=3` subset removes it at the cost of 607 crops; neither option has been
  trained on.
* **Phase B's unresolved caveats carry over verbatim** — the lung-fraction floor's class-correlated
  rejection, radiologist-impression labels rather than biopsy, and the 641 clusters (23.9%) that
  remain unlabelled. A z extent fixes none of them.
* **The 45-series validation set is Phase D/E.1's test-split volumes**, so the 189/189 figure is
  measured on ~5.6% of the series. The independent `check_spacing.py` check covers all 1,989 centre
  slices and agrees.

**Phase G.1 closes as: 3,461/3,461 stacks built with a bit-identical centre channel, adjacency
recovered from annotation XML and validated 189/189 against full volumes, zero architecture change
required and that verified against the actual layer shapes, smoke test passed at `[8, 3]` with no
NaN. Phase G.2 was not started — that decision is the user's.**
