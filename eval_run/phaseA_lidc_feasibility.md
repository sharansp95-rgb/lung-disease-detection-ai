# Phase A: LIDC-IDRI nodule-centered crop extraction — feasibility check

**Scope: feasibility only.** No training, no full-scale extraction, no changes to
`checkpoints/ct_best_model_phase35_v2.pt`, `backend/main.py`, or `data/ct_scan/`.
30 cases sampled, 107 crops produced, then stopped as instructed.

This tests the idea that the previous phases' root cause — *whole-slice classification
leaves background/table/framing available to shortcut on* — can be removed structurally by
never showing the model a whole slice at all: only a tight, radiologist-localized
lung-tissue crop.

## Verdict up front

**Technically feasible, and much cheaper than `phase1_data_candidates.md` assumed.** Access
is a plain anonymous REST API; the whole-volume download problem is avoidable entirely; the
DICOM coordinate handling that was flagged as the likely silent-bug source worked and was
verified empirically, not assumed. **107 correctly-formed crops from 30 cases, 28.4 MB
downloaded, ~1 hour of work.**

**And the shortcut surface measurably collapses.** The decisive probe from
`ct_framing_normalization_experiment.md` §2C — can a random forest predict the class from
global image statistics alone — reads **0.433 balanced accuracy (chance 0.333)** on these
crops, against **0.944 on the original whole-slice training data**. This is the first
intervention in the investigation that attacks the root cause at the level the root cause
was actually identified at.

**But the recommendation is a qualified go, not a clean one**, and the qualification is not
about extraction difficulty — it is that this changes the task the model performs, and the
live `/predict/ct` endpoint cannot consume the result. See §7.

---

## 1. Access — what it actually looks like

Contrary to `phase1_data_candidates.md`'s "needs substantial DICOM engineering / large
download" framing, **no account, no DUA, no desktop tool, no approval wait.** Every call
below was anonymous over plain HTTPS and worked first try.

| endpoint | result |
|---|---|
| `GET .../v1/getCollectionValues` | HTTP 200, 5 KB, 7.9 s — 156 collections, `LIDC-IDRI` present |
| `GET .../v1/getPatient?Collection=LIDC-IDRI` | HTTP 200, 152 KB, 3.9 s — **1,010 patients** |
| `GET .../v1/getSeries?Collection=LIDC-IDRI&Modality=CT` | HTTP 200, 852 KB, 4.2 s — **1,018 CT series** |
| `GET .../v1/getSingleImage?SeriesInstanceUID=..&SOPInstanceUID=..` | HTTP 200, one DICOM file, ~526 KB, ~4 s |
| `GET wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip` | HTTP 200, **9.0 MB, 16 s** — all 1,318 annotation XMLs |

Base: `https://services.cancerimagingarchive.net/nbia-api/services/v1/`.
The NBIA Data Retriever desktop tool was **not needed and not used.**

### The finding that changes the cost picture

`getSingleImage` fetches **one slice by SOPInstanceUID**. Combined with the fact that the
annotation XMLs (9 MB, all 1,010 cases) name the exact `imageSOP_UID` of every annotated
slice, **the whole-volume download is unnecessary.** The workflow is:

> download 9 MB of annotations → parse → learn exactly which slices matter → download only those.

Concretely: **30 cases cost 28.4 MB instead of ~3.8 GB** (a 130× reduction). The full
LIDC-IDRI CT collection is **130 GB** across 1,018 series (mean 126 MB, mean 240
slices/series) — but at no point does that number need to be paid.

Environment: `pydicom` was absent and installed cleanly (3.0.2). `tcia_utils` / `tciaclient`
absent, **not needed** — `urllib` against the REST API was sufficient.

**Sample downloaded:** 30 patients, deliberately round-robined across all four scanner
manufacturers (GE 8, Philips 8, Siemens 7, Toshiba 7) rather than taking the first 30 IDs,
since multi-scanner diversity is the whole reason this collection was shortlisted.

## 2. Annotation parsing

**Hand-rolled `xml.etree` parser, not `pylidc`** — deliberate. `pylidc` is not installed and
requires SQLAlchemy plus a *configured local path to a complete on-disk DICOM library*; its
model assumes you have whole volumes on disk, which is exactly the assumption this approach
avoids. Wiring it up would have cost more than the parser did. Parser:
[`parse_lidc_xml.py`](lidc_feasibility/parse_lidc_xml.py),
[`plan_nodules.py`](lidc_feasibility/plan_nodules.py). It was not a time sink — the XML is
shallow and regular.

**Format surprise worth recording:** 283 of the 1,318 XMLs have root tag `IdriReadMessage`
rather than `LidcReadMessage` — a second, older schema variant. These were skipped. **All
1,018 CT series are still fully covered** by the `LidcReadMessage` files (0 unmatched), so
the variant is redundant here, but anything parsing this collection must handle or skip it
rather than crashing.

### Cross-reader nodule clustering (a judgment call)

LIDC gives up to 4 independent reading sessions and **does not tell you which reader's
nodule is which physical nodule** — `noduleID` is not shared across sessions. Reads were
clustered greedily by 3D centroid distance (**≤10 mm**, at most one read per session per
cluster). Of 55 retained nodules, **24 carry a full 4-reader panel**; the rest have 1–3
reads. `pylidc` does this with a more principled distance metric; the simple version is
adequate at this scale but is a place a real build should use a proper clustering step.

### Rating → label mapping (flagged as a real judgment call)

There is **no agreed convention** for collapsing LIDC's 1–5 malignancy scale into discrete
classes. Rule used, on the **mean** of the available radiologists' ratings:

| mean malignancy | label |
|---|---|
| ≤ 2.5 | Benign |
| ≥ 3.5 | Malignant |
| 2.5 < mean < 3.5 | **ambiguous — excluded** |

Reasoning: the mean uses the whole panel rather than discarding readers (as "most severe"
would); the explicit ≤2.5 / ≥3.5 dead band is stricter than the common "mean < 3 / > 3"
split and avoids labelling a 2.75-mean nodule "Benign". **Cost of that strictness: 39 of 94
nodule clusters (41.5%) were excluded as ambiguous.** That is a large fraction, and it means
the retained set is the *easy tail* of the distribution — a model trained on it will look
better than it would on the full nodule population. Stated plainly so it is not later
mistaken for a clean result.

**Also worth being explicit about:** these are radiologists' *subjective impressions of
malignancy from the image*, not biopsy-confirmed pathology. They are a much better label
than anything currently in the project, but they are not ground truth, and a model trained on
them learns "what a radiologist panel thinks looks malignant."

### The "Normal" class

LIDC has no Normal class; one was manufactured, as anticipated. Approach used:
**nodule-free lung-parenchyma crops taken from the same slices as the nodule crops** — same
patients, same scanners, same reconstruction, same slice. A candidate centre had to be
(a) ≥40 mm from **every** annotated point in that case — including the unrated <3 mm nodules
and non-nodule marks that carry no malignancy rating, not just the rated ones — and
(b) ≥55% lung tissue.

Sampling Normals from the *same slices* is deliberate: it means Normal and nodule crops
cannot differ by patient, scanner, or acquisition. That is precisely the class↔acquisition
correlation that produced every failure documented in this investigation, and this design
makes it structurally unavailable.

## 3. Extraction

[`extract_crops.py`](lidc_feasibility/extract_crops.py). Per nodule: representative slice =
the ROI with the most contour points across the cluster; HU via `RescaleSlope`/`Intercept`;
**lung window WL −600 / WW 1500**; crop a fixed **48 mm physical box** about the ROI bbox
centre using each slice's own `PixelSpacing` (so crops are physically comparable across
scanners with different FOVs), resampled to 128×128 PNG.

### DICOM coordinate handling — verified, not assumed

This was flagged as the likely silent-bug source. Two independent checks:

**Slice identity.** Every downloaded file's `SOPInstanceUID` tag was re-read and compared to
the UID requested. **54/54 exact match, 0 mismatches.** Slice ordering and filenames are
never used anywhere in the pipeline.

**In-plane axis order.** Rather than trust a remembered convention for whether LIDC's
`xCoord` is a row or a column, it was determined empirically: a nodule is soft tissue inside
air-density parenchyma, so the correct interpretation must make the contour interior denser
than its surroundings. Rasterizing each contour both ways and comparing mean HU inside the
contour against a surrounding annulus:

| interpretation | contour − surround | nodules where it wins |
|---|---|---|
| **`xCoord` = column** (used) | **+372 HU** (median +363), positive in **53/55** | **52/55** |
| `xCoord` = row | +28 HU (median +4), positive in 29/55 | 3/55 |

Decisive. The row interpretation is indistinguishable from noise. **No coordinate bug was
found — but only because it was tested; the wrong choice here would have produced
plausible-looking lung crops centred on nothing, which visual inspection alone would
plausibly have missed.**

## 4. Results and sanity check

**107 crops from 26 patients** (4 of the 30 yielded no unambiguous nodule):

| label | crops | source |
|---|---|---|
| Malignant | 25 | mean rating ≥ 3.5 |
| Benign | 30 | mean rating ≤ 2.5 |
| Normal | 52 | nodule-free parenchyma, same slices |

Scanner spread: Toshiba 36, Philips 26, GE 23, Siemens 22 — genuinely multi-vendor, which no
prior phase's data had.
Manifest: [`labels_manifest.csv`](lidc_feasibility/labels_manifest.csv) /
[`.json`](lidc_feasibility/labels_manifest.json) — image path, case, series/SOP UID, nodule
cluster, individual ratings, mean, scanner, pixel spacing, nodule diameter, crop HU stats.

### Visual inspection — 12 crops read directly

[`qc_contact_sheet.png`](lidc_feasibility/qc_contact_sheet.png) (red circle = crop centre,
where the nodule should be). Honest assessment: **all 12 are anatomically correct. None
malformed — no black crops, no wrong-region crops, no coordinate-error signatures.**

- **Malignant (4/4 correct).** `0001` (32 mm, ratings 5|5|5|4) is a large spiculated mass
  with radiating strands, dead-centre — textbook. `0066` n1 (12 mm) and n2 (10 mm) are
  discrete irregular dense blobs squarely in the circle.
- **Benign (4/4 correct).** Small well-circumscribed bright nodules centred as expected
  (5.2 mm, 8.2 mm); `0003` n3 is a larger, fainter, more diffuse density — plausible for a
  low-rated lesion.
- **Normal (4/4 plausible).** Lung parenchyma with normal vessel branching, no mass.

Supporting numbers, consistent with the visual read: mean HU inside contour **−204
(Malignant)** vs **−411 (Benign)** vs crop-level **−631 (Normal)** — the density ordering
real pathology would produce.

### The one real defect found

**Crops are too large: at 48 mm, ~50–60% of crops contain some non-lung structure**
(chest wall, mediastinum, bone). One Normal crop in the QC sheet includes a substantial
chest-wall band. This partially reintroduces the very anatomy the approach exists to
exclude.

Two mitigating facts, and one required fix:
- It is **not strongly class-correlated** (mean lung fraction 0.79 Malignant / 0.73 Benign /
  0.81 Normal), unlike the framing confound, which was.
- It shrinks with crop size: at **24 mm** the fractions are 0.88 / 0.83 / **0.93**.
- **Phase B must tighten the crop to ~24–32 mm and add a hard lung-fraction floor
  (≥0.85).** This is a parameter change, not a redesign.

### Shortcut-surface probe — the most informative result here

The decisive probe from `ct_framing_normalization_experiment.md` §2C, re-run on these crops
(22 global style features, 5-fold **grouped by patient** so no case spans folds):

| data | class-from-global-style, balanced accuracy | chance |
|---|---|---|
| original whole-slice training data (phase37 report §2C) | **0.944** (random forest) | 0.333 |
| **these nodule crops** | **0.433** RF / 0.464 LR | 0.333 |

**The shortcut surface largely collapses.** This is a structural property of the
representation, achieved without any of the four iterations of masking/histogram surgery the
framing experiment needed — there is simply almost no non-lesion global signal left to
exploit.

Caveat, stated rather than buried: **Benign-vs-Malignant alone is 0.687 / 0.653**, above the
0.5 chance line. Some of that is real pathology leaking into global statistics (malignant
nodules genuinely are larger and denser — that is signal, not shortcut), and some is small-n
noise at 55 nodules / 26 patients. It cannot be cleanly separated at this scale, and should
not be waved away.

## 5. Blockers hit

None fatal. Recorded for the time estimate:

1. `pydicom` absent — 30 s to install.
2. **`IdriReadMessage` schema variant** (283 files) — would crash a naive parser.
3. **No cross-reader nodule correspondence in LIDC** — required writing the clustering step;
   the largest single piece of judgment in the pipeline.
4. **41.5% of nodules excluded as ambiguous** — inherent to the label, not a bug.
5. **A misleading first QC metric.** An initial "is the crop centre denser than its ring"
   check read only +51 HU, 33/55 positive, which looked like a centering bug. It was a
   metric artifact: for a 5 mm nodule in a 48 mm crop the "ring" is dominated by chest wall,
   which is far denser than any nodule. The contour-vs-annulus test (§3) is the correct
   instrument and is unambiguous. **Noted because a Phase B run that used the naive metric
   would wrongly conclude its extraction was broken.**
6. Crop-size/lung-fraction issue in §4.

## 6. Cost to scale — grounded in measured rates

| step | measured | extrapolated to all 1,010 cases |
|---|---|---|
| annotation download | 9 MB / 16 s | **already complete** — the 9 MB is all cases |
| annotation parsing | 1,318 XMLs in ~10 s | **already complete** for all 1,018 series |
| slice download | 54 slices / 38 s / 28.4 MB, 6 threads = **1.42 slices/s** | ~2.1 labelled nodules/patient → **~2,100 slices ≈ 25 min ≈ 1.1 GB** |
| — with ±3 adjacent slices per nodule | — | ~14,700 slices ≈ **2.9 h ≈ 7.7 GB** |
| crop extraction | 107 crops in seconds | minutes |
| **this session, end to end** | **~1 hour** including verification and one false-start metric | — |

Expected yield at full scale: **~2,000 labelled nodule crops + a matched Normal set**,
across 1,010 patients, 7 institutions, 4 scanner vendors.

**Honest estimate to a complete, QC'd Phase B dataset: about one focused day** — a few hours
of download, plus crop-size retuning, lung-fraction filtering, a patient-grouped split, and
QC. The download is not the bottleneck; the judgment calls are. This is *far* cheaper than
`phase1_data_candidates.md` projected, and that projection should be considered superseded —
it assumed whole-volume downloads and `pylidc`-style tooling, both of which turned out to be
avoidable.

## 7. Go / no-go

**Split recommendation, because the answer differs by goal.**

### If the goal is to improve the deployed CT model before the exam — **NO-GO.**

Not because extraction is hard. Because **a nodule-crop classifier cannot be served by
`POST /predict/ct`,** which takes a whole uploaded CT slice. To deploy this you would need a
nodule *detector* (or a user-drawn ROI in the frontend) to produce the crop at inference
time. Building and validating a nodule detector is a larger project than everything in this
report combined, and is not a ~10-day-with-an-exam undertaking. The class semantics also
shift: "Normal" here means *this 3 cm patch of lung has no nodule*, not *this patient is
healthy* — it is not a drop-in replacement for what the app currently claims.

Attempting this path before the exam risks the working deployed system for a result that
would not be finishable. **Keep `ct_best_model_phase35_v2.pt` live**, per the standing
recommendation.

### If the goal is to answer the investigation's open question — **GO, and it is cheap.**

`ct_framing_normalization_experiment.md` §7 left one question unresolved: *is there any
lesion-level signal learnable here at all, or is ~1,000 single-source images simply too
few?* This is the cheapest available way to answer it, and the only approach tried so far
that removes the shortcut surface **structurally** rather than by post-hoc normalization —
0.944 → 0.433 on the probe that decided the last experiment, achieved by construction.

That is worth **one day**, bounded: extract the full crop set, train one model, evaluate
patient-grouped, run Grad-CAM. It yields a real, defensible finding either way — including a
clean negative, which is publishable-quality evidence for the report and strictly better
than the current position of "we don't know."

### What I actually think

The technical risk that motivated this check has been retired: access is easy, the
coordinate handling is verified correct, and the crops are real. The residual risk is
**scope**, not feasibility — this is a different modelling problem wearing the same project's
clothes, and it is easy to lose a week to it by mistake.

So: **do it as a bounded one-day experiment for the evidence, with an explicit no-deployment
rule and a hard stop.** Do not start it if that day is not genuinely spare — the standing
recommendation (keep `phase35_v2`, stop citing external-Malignant 12/12 as generalization
evidence) is already correct and complete without it, and nothing here changes what should
be *said* about the current model. This buys a better explanation, not a better product.

## 8. Artifacts

All under `eval_run/lidc_feasibility/`. Nothing referenced by `backend/`.

| path | what |
|---|---|
| `parse_lidc_xml.py` | XML → series/patient index (all 1,018 series) |
| `plan_nodules.py` | cross-reader clustering, rating→label mapping, download plan |
| `download_slices.py` | targeted `getSingleImage` fetch, 6 threads |
| `extract_crops.py` | HU + lung window, nodule crops, Normal sampling, manifest |
| `images/{Benign,Malignant,Normal}/` | **107 extracted crops**, 128×128 PNG |
| `labels_manifest.csv` / `.json` | full per-crop manifest |
| `qc_contact_sheet.png` | the 12 crops inspected in §4 |
| `_work/xml/` | 1,318 LIDC annotation XMLs (all cases) |
| `_work/dicom/` | 54 downloaded DICOM slices, 28.4 MB |
| `_work/{xml_index,plan,download_log,coord_convention_check}.json` | index, plan, timings, coordinate verification |

Disk: **209 MB** total.

**Live system untouched:** no writes outside `eval_run/lidc_feasibility/`;
`checkpoints/ct_best_model_phase35_v2.pt`, `backend/main.py` and `data/ct_scan/` were not
modified (`backend/main.py` was read only, to confirm the §7 endpoint claim).

Stopping here as instructed — no full-scale extraction, no training.
