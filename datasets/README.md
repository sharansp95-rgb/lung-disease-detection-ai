# Datasets — Review Reference

Map of every dataset used in this project, for quick browsing during project
review/viva. This folder used to hold physical copies of the five datasets
below; those copies were byte-identical duplicates and were removed in the
pre-evaluation cleanup (see `docs/CLEANUP_LOG.md`). **The datasets now live
only where the pipelines read them:** the raw sources in `raw/`, the
model-ready splits in `data/`.

The only dataset still stored in this folder is `ct_normal_extra/` (external
Normal CT slices used for the CT retraining experiment; see below).

For full provenance, cleaning, dedup, split, and audit methodology, see
[data/README.md](../data/README.md) — this file is just the summary table
and a folder map.

## Summary

| # | Location | Source | Type | Classes | Count / Split | Status |
|---|---|---|---|---|---|---|
| 01 | `raw/xray_covid/COVID-19_Radiography_Dataset/` | COVID-19 Radiography Database (Kaggle: tawsifurrahman) | X-ray, raw | COVID, Normal, Lung_Opacity, Viral Pneumonia | 3,616 / 10,192 / 6,012 / 1,345 (raw, unsplit) | Downloaded |
| 02 | `raw/xray_tb/TB_Chest_Radiography_Database/` | TB Chest Radiography Database (Kaggle: tawsifurrahman) | X-ray, raw | Normal, Tuberculosis | 3,500 / 700 (raw, unsplit) | Downloaded |
| 03 | `raw/ct_scan/The IQ-OTHNCCD lung cancer dataset/` | IQ-OTH/NCCD Lung Cancer Dataset (Kaggle: hamdallak) | CT, raw | Normal, Benign, Malignant | ~1,097 images, 110 patients (raw, unsplit) | Downloaded |
| 04 | `data/xray/` | Merged from 01 + 02, cleaned, cropped, deduped | X-ray, processed | COVID-19, Normal, Pneumonia, Tuberculosis | 15,373 / 1,922 / 1,922 (train/val/test) | Verified (real run) |
| 05 | `data/ct_scan/` | Derived from 03, leakage-fixed, protocol-mismatch excluded | CT, processed | Normal, Benign, Malignant | 829 / 104 / 103 (train/val/test) | Verified (real run) |

## Folder map

```
raw/
  xray_covid/COVID-19_Radiography_Dataset/      -> 01, raw source, as downloaded from Kaggle
  xray_tb/TB_Chest_Radiography_Database/        -> 02, raw source, as downloaded from Kaggle
  xray_tb_cleaned/                              -> 02 after the TB burned-in-artifact cleanup
  ct_scan/The IQ-OTHNCCD lung cancer dataset/   -> 03, raw source, as downloaded from Kaggle
data/
  xray/                                         -> 04, train/ val/ test/, model-ready
  ct_scan/                                      -> 05, train/ val/ test/, model-ready
datasets/
  ct_normal_extra/                              -> external Normal CT slices (retraining experiment only)
```

**01 and 02** merge into **04** (X-ray model). `Lung_Opacity` from 01 is
excluded — not a target class for this project.

**03** becomes **05** (CT model), after two audits: a patient-leakage
resplit (near-duplicate CT slices grouped so they never cross train/val/test)
and a protocol-mismatch exclusion (61 images from a different scan
acquisition removed, not cropped — see `data/README.md` for why).

Both processed sets (04, 05) had their class-balance, dedup, and split
methodology documented and audited before use — not just downloaded and
split blindly. See `data/README.md` for the full writeup, including the
TB burned-in-artifact cleanup and the CT scanner-table shortcut investigation.
