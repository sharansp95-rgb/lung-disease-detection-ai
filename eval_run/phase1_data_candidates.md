# Phase 1: CT data-sourcing feasibility (research only, no downloads)

Goal: find a lung CT dataset with Benign/Malignant/Normal-equivalent labels that is (a) not
IQ-OTH/NCCD-derived, (b) not `mohamedhanyyy/chest-ctscan-images` (already used), and ideally
(c) genuinely multi-institution/multi-scanner, to test whether the CT classifier's
generalization problem (documented across `ct_normal_root_cause.md`, `ct_retrain_experiment.md`,
`ct_shortcut_inversion_analysis.md`, `ct_framing_normalization_experiment.md`) is fixable with
more/better data.

## Candidates found

### 1. LIDC-IDRI (TCIA / Cornell mirror)
- **Classes/labels:** No scan-level Benign/Malignant/Normal folders. Nodule-level malignancy
  score 1–5 from up to 4 radiologists per nodule (score <3 = benign-leaning, =3 = uncertain,
  >3 = malignant-leaning) — would need to be **collapsed/relabeled** into this project's
  3-class scheme, and a "Normal" class would need to be derived separately (e.g. slices with
  no annotated nodules), which is not how the dataset is structured or intended to be used.
- **Image count:** 1,010–1,018 full 3D CT scans (whole-volume DICOM), not pre-cut 2D
  slice/class folders.
- **Acquisition diversity:** **Multi-institution — 7 medical centers.** The single strongest
  diversity signal of any candidate found.
- **Access:** Fully open, no data use agreement. Via TCIA or the Cornell VIA mirror.
- **Label trustworthiness:** High — radiologist-panel annotated (up to 4 readers per nodule),
  the gold-standard source most benign/malignant nodule-classification literature is built on.
- **Engineering cost:** High. Raw DICOM volumes, need nodule-region-to-2D-slice extraction, HU
  windowing decisions, and a self-defined mapping from 5-point malignancy score to 3 discrete
  classes. No "Normal" whole-scan class exists natively.

### 2. LUNA16 (derived from LIDC-IDRI)
- **Classes/labels:** Nodule vs. non-nodule only (detection benchmark), not benign/malignant.
  **Not usable for this classification task** without going back to LIDC-IDRI's own annotations
  anyway — LUNA16 adds no new label information for this purpose. Included for completeness
  since it was named as a candidate; not a real independent option.

### 3. NLST (National Lung Screening Trial)
- **Classes/labels:** Patient/exam-level "cancer diagnosed within 1 year of this scan" vs. not
  — not a per-slice or even per-scan Benign/Malignant/Normal label, and not biopsy-confirmed at
  the image level (it's a screening-trial outcome label). Real Normal cases exist (most of the
  cohort), but distinguishing them from undiagnosed/indeterminate nodules requires the
  associated NLST Pathology data.
- **Image count:** Full dataset >75,000 CT screening exams (~500 patients in the openly
  distributable TCIA subset; the complete trial cohort is much larger under restricted access).
- **Acquisition diversity:** **Multi-institution — originally ~33 US screening centers.** Very
  strong diversity, arguably the best available for "Normal" specifically.
- **Access:** **Two-tiered.** A publicly-distributable subset (~500 patients, unrestricted) is
  available via TCIA. The full clinical dataset requires a formal application and signed Data
  Transfer and Use Agreement through NCI's Cancer Data Access System (CDAS) — a real access
  barrier for the complete dataset, not just paperwork friction.
- **Label trustworthiness:** High for the trial's own screening-outcome purpose, but weak/noisy
  as a per-image Benign/Malignant/Normal label without substantial additional processing against
  the pathology data.
- **Engineering cost:** High — DICOM volumes, exam-to-slice extraction, and a non-trivial
  labeling exercise to get from "trial outcome" to "this specific slice shows a benign nodule/
  malignant nodule/nothing."

### 4. TCIA Lung-PET-CT-Dx
- **Classes/labels:** Malignant subtypes only (251 adenocarcinoma, 61 squamous cell carcinoma).
  **No benign controls, no normal class at all.** Could only supplement Malignant-class
  diversity, not the Normal generalization problem this investigation is actually centered on.
- **Image count:** 355 patients.
- **Acquisition diversity:** Multi-institution (355 patients, biopsy-confirmed NSCLC).
- **Access:** Open via TCIA, no DUA.
- **Label trustworthiness:** High — biopsy-proven pathology labels.
- **Engineering cost:** High (DICOM + PET/CT co-registration if using both modalities), and
  fundamentally incomplete for this project's 3-class need regardless of engineering effort.

### 5. SPIE-AAPM-NCI Lung CT Challenge (LUNGx)
- **Classes/labels:** Genuine benign/malignant "truth" label per nodule, biopsy/follow-up
  confirmed, with centroid coordinates provided.
- **Image count:** **70 subjects total** (10 training + 60 test) — smaller than even this
  project's existing Benign class support.
- **Acquisition diversity:** Multi-institution in principle (TCIA-hosted challenge dataset), but
  the subject count is too small for this to matter much in practice.
- **Access:** Open via TCIA, no DUA, DOI-referenced.
- **Label trustworthiness:** High — designed as a rigorous classification challenge with
  confirmed ground truth.
- **Engineering cost:** High relative to its size — DICOM + nodule-region extraction for a net
  gain of well under 100 images.

### 6. Kaggle Data Science Bowl 2017 (NLST-sourced)
- **Classes/labels:** Binary patient-level "cancer within 1 year" vs. not — same weak-label
  problem as NLST directly, since it's sourced from NLST. No nodule location, no benign/normal
  distinction within the "no cancer" group.
- **Image count:** ~2,101 3D patient scans.
- **Acquisition diversity:** Multi-site (inherited from NLST), though not documented
  scan-by-scan in the Kaggle release itself.
- **Access:** Open on Kaggle (no DUA for this specific redistribution), competition-era dataset.
- **Label trustworthiness:** Weak — outcome-based, not per-image pathology-confirmed.
- **Engineering cost:** Very high for very little labeling value — full DICOM volumes, huge
  download size, and the label granularity is worse than what's already been tried.

## What was ruled out by design (per task instructions)
- IQ-OTH/NCCD and all its Kaggle re-uploads (`hamdallak`, `adityamahimkar`, `subhajeetdas`,
  `waseemnagahhenes`, and bundle datasets that only re-host it) — this is the dataset the
  original model overfits to; adding more copies of it would not help.
- `mohamedhanyyy/chest-ctscan-images` — already incorporated in the phase36/phase37
  experiments.

## Cross-cutting observation
Every candidate that has real Benign/Malignant/Normal-equivalent labels *and* open,
unrestricted access (LIDC-IDRI, SPIE-AAPM/LUNGx) is a **raw 3D DICOM volumetric collection**
with **nodule-level, not scan-level or slice-level, annotations** — nothing found matches the
"pre-packaged folder of 2D images, one label per image" structure that both IQ-OTH/NCCD and
`mohamedhanyyy/chest-ctscan-images` already provide and that this project's entire pipeline
(`ml/dataset.py`, `ml/train.py`) is built around. The two sources that *are* genuinely
multi-institution at meaningful scale (LIDC-IDRI: 7 centers; NLST: ~33 centers) either lack a
usable "Normal" class without a large separate labeling effort (LIDC-IDRI) or gate the full
dataset behind a formal DUA application (NLST). Nothing found offers "download a folder,
dedupe, retrain" the way the last two experiments' data sourcing did.
