# Lung Disease Detection — Datasets

This directory holds cleaned, split, model-ready image data for **two separate
imaging modalities**, each destined for its own model. They are kept in fully
separate folder trees and were never cross-deduplicated against each other.

```
data/
  xray/       train/ val/ test/   -> COVID-19, Normal, Pneumonia, Tuberculosis
  ct_scan/    train/ val/ test/   -> Normal, Benign, Malignant
```

Random seed used throughout: **42** (`random.seed(42)`, Python `random` module).

Prep scripts: [scripts/prepare_xray.py](../scripts/prepare_xray.py),
[scripts/prepare_ct_scan.py](../scripts/prepare_ct_scan.py) (original per-image
split; superseded by ct_resplit_grouped.py below),
[scripts/tb_audit_grid.py](../scripts/tb_audit_grid.py) (inspection grids),
[scripts/tb_artifact_detect.py](../scripts/tb_artifact_detect.py) (full-dataset
artifact scan), [scripts/tb_clean.py](../scripts/tb_clean.py) (artifact cleanup),
[scripts/ct_leakage_audit.py](../scripts/ct_leakage_audit.py) (near-duplicate /
patient-leakage detection), [scripts/ct_resplit_grouped.py](../scripts/ct_resplit_grouped.py)
(cluster-aware resplit), [scripts/ct_table_shortcut_audit.py](../scripts/ct_table_shortcut_audit.py)
(scanner-table / protocol-mismatch investigation — current `data/ct_scan/`
reflects both the resplit and the resulting exclusion).

**Revision note:** the CT-scan source dataset was replaced after the initial
build — see Part B below. The original `mohamedhanyyy/chest-ctscan-images`
dataset (adenocarcinoma/large-cell/squamous-cell/normal, ~1,000 images) was
the wrong dataset for this project's spec and has been fully removed, not
kept as an alternative.

---

## Phase 3.5 — Shortcut-Learning Fix (applied before current data/xray/ build)

After a Grad-CAM audit in Phase 3 found the X-ray model attending to corners
rather than lung regions, two source-level preprocessing fixes were applied
before rebuilding `data/xray/`. These happen inside `prepare_xray.py` during
the copy step — every image written to `data/xray/` is already cropped.
`dataset.py` and `ml/train.py` are unchanged.

### COVID-19 Radiography Dataset — mask-guided lung bbox crop

The dataset ships real per-image binary lung-segmentation masks under
`<class>/masks/<filename>.png` alongside `<class>/images/<filename>.png`,
with **100% coverage across all 4 classes** (21,165 image/mask pairs):

| Class | Images | Masks |
|---|---|---|
| COVID | 3,616 | 3,616 |
| Normal | 10,192 | 10,192 |
| Lung_Opacity | 6,012 | 6,012 *(excluded from training)* |
| Viral Pneumonia | 1,345 | 1,345 |

**Fix:** load image + mask → binarise mask at threshold 127 → compute tight
bounding box of white (lung) pixels with 4 px padding → crop image to bbox →
resize to 224×224. Corner text, watermarks, and anything outside the lung
field is removed at source, not guessed at by the transform pipeline.

Fallback for degenerate masks (all-zero or bbox < 64 px): a 5% border crop
is applied instead, and a warning is emitted. Any such cases are counted in
the script's output.

Masks may be stored at a different resolution than their paired image (e.g.
256×256 mask for a 299×299 image, discovered during inspection). The code
handles this by nearest-neighbour resizing the mask to match the image before
computing the bbox — no spatial information is fabricated.

### TB Chest Radiography Database — uniform 8% border crop

No segmentation masks exist. A Grad-CAM + visual audit showed watermarks
consistently confined to the outer border. A uniform 8% crop from each edge
is applied before resize to 224×224. The margin was chosen by visual
inspection of sample images via `scripts/inspect_crops.py` (corner zoom grid
saved at `scripts/crop_preview/`).

### Net effect on counts

The image counts per class are identical to the pre-Phase-3.5 counts (no
images are dropped by the mask crop — degenerate mask fallback covers the
rare zero-mask case). File sizes are slightly smaller (224×224 greyscale PNG
vs the original mixed resolutions), and all output images are now PNG
regardless of source format (some TB source images were JPG).

---

## Part A — X-ray (`data/xray/`)

### Source datasets (Kaggle)

1. **`tawsifurrahman/covid19-radiography-database`** — "COVID-19 Radiography
   Database". Extracted structure:
   `COVID-19_Radiography_Dataset/{COVID,Lung_Opacity,Normal,Viral Pneumonia}/images/*.png`
   (each class also has a `masks/` subfolder, ignored — segmentation masks,
   not classification inputs). Raw counts: COVID 3616, Lung_Opacity 6012,
   Normal 10192, Viral Pneumonia 1345.
2. **`tawsifurrahman/tuberculosis-tb-chest-xray-dataset`** — "Tuberculosis
   (TB) Chest X-ray Database". Extracted structure:
   `TB_Chest_Radiography_Database/{Normal,Tuberculosis}/*.png` (flat, no
   masks). Raw counts: Normal 3500, Tuberculosis 700.

### Class mapping

| Source class | Source dataset | -> Target class |
|---|---|---|
| `COVID` | covid19-radiography-database | `COVID-19` |
| `Viral Pneumonia` | covid19-radiography-database | `Pneumonia` |
| `Normal` | covid19-radiography-database | `Normal` |
| `Lung_Opacity` | covid19-radiography-database | **excluded** (not a target class) |
| `Normal` | tuberculosis-tb-chest-xray-dataset | `Normal` |
| `Tuberculosis` | tuberculosis-tb-chest-xray-dataset | `Tuberculosis` |

`Normal` is merged from **both** source datasets, since both authors published
a "Normal" chest-X-ray class independently.

### Dedup method

Both source datasets contribute to the `Normal` class, and images were
deduped by **SHA-256 content hash of the file bytes**, not by filename —
filenames like `Normal-1.png` recur in both datasets by coincidence, not
because they're the same image. Deduping happened **within each target
class** (i.e. hash pool is per-class, not global), so:

- Cross-dataset dedup: only 1 exact duplicate found between the two `Normal`
  sources out of 13,692 combined Normal images — the two "Normal" collections
  are essentially disjoint despite sharing the same dataset author.
- **Unexpected finding:** dedup also caught **within-dataset** duplicates —
  46 duplicate `COVID` images, 7 duplicate `Viral Pneumonia` images, and
  (in the current, post-TB-cleanup pipeline run — see the TB audit below)
  5 duplicate `Tuberculosis` images, all duplicated *inside a single source*,
  not across the two datasets. These were removed the same way. Worth knowing
  if you see this dataset elsewhere and get slightly different counts. (The
  Tuberculosis duplicate count rose from 3 to 5 after the TB cleanup step —
  desaturating tinted images made a couple of previously tint-differentiated
  images byte-identical.)

Filenames in the output are prefixed with a source tag (`covid19db_` /
`tbdb_`) to avoid any name collision between the two source datasets when
copying into the same class folder.

### Split

80/10/10 (train/val/test) per class, shuffled with `random.Random(42)`,
copied (not moved) from the raw scratch folders. Split is class-stratified
(each class split independently), so overall class balance is preserved
across all three splits.

### TB data-quality audit — burned-in artifacts (found and fixed)

Before finalizing the X-ray split, `raw/xray_tb/` was audited for burned-in
artifacts (text markers, redaction boxes, orientation letters, annotation
overlays) that could differ systematically between `Normal` and
`Tuberculosis` and let a model learn the artifact instead of the disease.
Method:

1. **Visual grids**: [scripts/tb_audit_grid.py](../scripts/tb_audit_grid.py)
   rendered a 36-image random sample grid for each class
   (`audit/tb_audit_normal_grid.png`, `audit/tb_audit_tuberculosis_grid.png`,
   seed 42) for direct visual inspection.
2. **Finding**: the Tuberculosis grid showed many images with a cyan/teal
   color cast and/or white rectangular boxes (redactions/labels) that never
   appeared in the Normal grid.
3. **Full-dataset scan** (not just the 36-sample grid):
   [scripts/tb_artifact_detect.py](../scripts/tb_artifact_detect.py) scanned
   all 4,200 images (3,500 Normal + 700 Tuberculosis) for (a) mean HSV
   saturation above threshold (color tint — a real chest X-ray should be
   near-neutral grayscale) and (b) large, solid-enough, box-shaped
   near-white connected components (`cv2.connectedComponentsWithStats`,
   fill_ratio ≥ 0.5, aspect ratio ≤ 7.5 to exclude thin bright strips
   hugging the bottom edge — those turned out to be a harmless artifact,
   likely sub-diaphragm overexposure or letterboxing, present in both
   classes and not diagnostic of anything).

   **Result — the class-correlation is real and severe:**

   | | Normal (n=3500) | Tuberculosis (n=700) |
   |---|---|---|
   | tinted | 0 (0%) | 370 (52.9%) |
   | has white box | 13 (0.37%) | 218 (31.1%) |
   | large box (≥5% of image) | 5 (0.14%) | 72 (10.3%) |
   | **flagged for any artifact** | **13 (0.37%)** | **372 (53.1%)** |

   Over half of all Tuberculosis images carry an artifact that's essentially
   absent from Normal — exactly the kind of shortcut a model could learn
   instead of the actual pathology.

4. **Cleanup** ([scripts/tb_clean.py](../scripts/tb_clean.py)), applied
   uniformly to *both* classes (not just Tuberculosis, for consistency):
   - **Tinted images** → desaturated to true grayscale (recoverable — the
     color cast carries no anatomical information, so removing it only
     strips the artifact).
   - **Images with a white box covering < 5% of the image** → the box
     region was inpainted (`cv2.inpaint`, Telea algorithm) using surrounding
     pixel content, removing the stark rectangular "tell" while keeping the
     rest of the image (recoverable — verified by before/after crops that
     the inpaint blends plausibly into surrounding lung/tissue texture,
     though a faint edge outline can remain on larger boxes).
   - **Images with any box covering ≥ 5% of the image** → excluded entirely
     (unrecoverable — too much of the image content is destroyed to trust
     what's underneath).
   - Output written to a new `raw/xray_tb_cleaned/` mirror;
     **`raw/xray_tb/` itself was never modified**, only read.

   **Before/after counts:**

   | class | source | unchanged | desaturated | inpainted | excluded | cleaned total |
   |---|---|---|---|---|---|---|
   | Normal | 3500 | 3487 | 0 | 8 | 5 | 3495 |
   | Tuberculosis | 700 | 328 | 298 | 146 | 72 | 628 |

5. `prepare_xray.py` was repointed from `raw/xray_tb/` to
   `raw/xray_tb_cleaned/` and rerun (idempotent — fully rebuilt
   `data/xray/` from scratch). The X-ray `COVID-19`, `Normal` (COVID-source
   half), and `Pneumonia` classes were **not touched** by this audit — only
   the TB-sourced `Normal` and `Tuberculosis` images changed, and the drop
   in `Normal`'s total (13,691 → 13,686 after rebuild) and `Tuberculosis`
   (697 → 623) reflects the 5 Normal + 72 Tuberculosis exclusions above,
   partly offset by 2 extra content-hash duplicates that emerged after
   desaturation made some previously tint-differentiated images byte-identical.

   Verification grids of the cleaned output are saved at
   `audit/tb_final_tuberculosis_grid.png` (and equivalent Normal grid) —
   spot-checked visually to confirm color casts are gone and most redaction
   boxes are no longer visible.

   **Known limitation**: the detector is heuristic (saturation + shape
   thresholds calibrated against a 300-image Normal sample to avoid false
   positives) — it is not a perfect, exhaustive audit of all 4,200 images
   pixel-by-pixel, and a small number of edge-case artifacts (e.g. thin
   bright bars near the bottom border, deliberately excluded because they
   also occur in Normal images and are more likely padding/anatomy than a
   deliberate overlay) may remain unaddressed. The scan results are saved at
   `audit/tb_artifact_scan.json` for anyone who wants to re-examine the
   raw per-image findings.

### Final counts — X-ray (after TB cleanup)

| split | COVID-19 | Normal | Pneumonia | Tuberculosis | total |
|---|---|---|---|---|---|
| train | 2856 | 10949 | 1070 | 498 | 15373 |
| val | 357 | 1369 | 134 | 62 | 1922 |
| test | 357 | 1368 | 134 | 63 | 1922 |
| **total** | **3570** | **13686** | **1338** | **623** | **19217** |

**Class imbalance flag:** `Normal` (13,686) outnumbers `Tuberculosis` (623)
by ~22x, and `Pneumonia` (1,338) by ~10x. This imbalance is inherited
directly from the source datasets, not introduced by this pipeline. It will
need addressing at training time (class weighting, oversampling, or
stratified batching) — not something to fix by discarding data here.

**Cross-source merge attempted and fully reverted (2026-09-06):** a
Shenzhen + Montgomery external-source merge (adding independently-acquired
`Normal`/`Tuberculosis` images via Kaggle `kmader/pulmonary-chest-xray-abnormalities`)
was trialed as part of a generalization investigation, briefly committed to
`data/xray/` (+406 Normal, +394 Tuberculosis at its peak), then reverted in
full at the user's request before any downstream training used it. The
revert was exact and verified: every merged file was uniquely prefixed
(`montgomery_`/`shenzhen_`) so it could be — and was — removed by prefix
with zero ambiguity, restoring the counts above precisely. **The counts and
provenance in this file reflect the current, reverted state — this merge is
not part of `data/xray/` as it exists on disk.** Recorded here only so this
attempt isn't mistaken for something that never happened; nothing about it
is currently active.
