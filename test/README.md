# test/ — generalization diagnostic harness

Drop this folder at the project root, alongside `backend/`, `data/` and
`checkpoints/`. Nothing here modifies the project; every script is read-only
with respect to your code and weights.

**Purpose:** find out *why* the classifiers do well on held-out images from the
training sources and badly on chest images from elsewhere. There are three
candidate causes and they have completely different fixes, so we diagnose
before changing anything.

| # | Cause | Fix | Cost |
|---|---|---|---|
| A | Preprocessing mismatch (bad lung crop, wrong framing) | fix `seg_utils.py` / `preprocess_image()` | hours, no retraining |
| B | Intensity / format shift (windowing, gamma, compression) | normalize in preprocessing | hours + one retrain |
| C | Source-database confound (the model learned "which dataset") | retrain with augmentation | GPU time |

## Install

```bash
pip install requests pillow numpy imagehash scikit-image
```

## Run order

### Step 0 — check the probe you already used

The earlier diagnostic assumed `sample images/` was external. That was never
verified, and the class folder names (`covid`, `normal`, `Pneumonia`, `tb`,
`cancer`) match the training datasets. Check it:

```bash
cd test
python scripts/verify_provenance.py --probe "../sample images" --json results/provenance_sample.json
```

If most images come back `EXACT_MATCH_IN_*` or `NEAR_DUPLICATE_OF_*`, then
`GENERALIZATION_DIAGNOSIS.md` measured nothing and its conclusions should be
discarded, not reinterpreted.

### Step 1 — corruption test (no external data needed)

**Run this one first if you only have time for one.** It uses your own test
split and needs nothing downloaded.

```bash
# terminal 1
cd backend && uvicorn main:app --port 8000

# terminal 2
cd test
python scripts/test_corruption.py --modality xray --images ../data/xray/test \
    --per-class 5 --json results/corruption_xray.json
python scripts/test_corruption.py --modality ct --images ../data/ct_scan/test \
    --per-class 5 --json results/corruption_ct.json
```

It takes images the model already gets **right**, applies one realistic
transform at a time — resize, JPEG, black border, gamma, contrast, inversion,
histogram equalization — and re-predicts through the real API.

Adjust `--images` if your split lives elsewhere. The layout must be
`<dir>/<ClassName>/*.png`.

### Step 2 — external probe

Fill `external_probe/` first — see `SOURCING.md`. Then:

```bash
python scripts/verify_provenance.py --probe external_probe --json results/provenance_external.json
python scripts/test_external.py --modality xray --images external_probe/xray --json results/external_xray.json
python scripts/test_external.py --modality ct   --images external_probe/ct   --json results/external_ct.json
```

## Reading the results

**Corruption test shows a 20+ point drop on any transform** → cause A or B.
The model breaks on its *own* images when they are lightly altered, so it never
had a chance on images that differ in those same ways. Fix preprocessing first;
retraining will not help if the input never reaches the model in the form it
expects. A drop on **black border 10%** specifically implicates the lung-crop
step — check whether `compute_lung_crop()` is silently falling back to
`border_crop()`.

**Corruption test shows robustness everywhere, but external images fail** →
cause C. This matches the source-confound audit already in your paper: a
32×32-pixel probe recovers the source database at 97.1% accuracy, and COVID-19
survives having its lung fields blanked with only a 1.7-point drop. Retrain with
border occlusion, random crop, gamma/contrast jitter and JPEG noise.

**External images are mostly REJECTED rather than misclassified** → nothing is
broken. The OOD detector and gates are refusing input they correctly recognise
as outside the training distribution. What looks like "wrong results" in the UI
is correct refusal. Report it that way.

One thing to watch: if real external **scans** are being rejected by the
*sanity gate*, that gate is over-triggering. It exists to catch photographs and
screenshots, not radiographs from another hospital. Its negative pool probably
lacks acquisition diversity. This is the same open question as Fig. 12 in the
paper, where a genuine CT rejected on the X-ray route produced a sanity-gate
message rather than a modality-gate one.

## What this means for the paper

Whatever the outcome, it belongs in Section V-E (Honest limitations). A measured
and disclosed generalization gap strengthens the paper's central argument rather
than undermining it — the whole thesis is that failure modes were found and
reported rather than assumed absent. Suggested wording:

> Held-out test accuracy is reported on the same acquisition sources the
> classifiers were trained on. On a small probe of chest images from other
> sources, [state what you measured]. This is consistent with the
> source-database confound documented in Section IV-E. The reported figures
> should be read as within-source performance, not as evidence of
> cross-institution generalization.

Fill the bracket with a real number once you have run the scripts. Do not
report a number you did not measure.

## Files

```
test/
├── README.md                      this file
├── SOURCING.md                    where to get external images, and the rules
├── PROVENANCE.csv                 fill in as you collect images
├── scripts/
│   ├── verify_provenance.py       is the probe set actually external?
│   ├── test_corruption.py         what breaks on your own images? (no data needed)
│   └── test_external.py           which stage handles external images?
├── external_probe/                you fill this — see SOURCING.md
│   ├── xray/{covid,normal,pneumonia,tuberculosis}/
│   └── ct/{benign,malignant,normal}/
└── results/                       JSON output lands here
```
