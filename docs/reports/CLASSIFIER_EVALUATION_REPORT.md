# Classifier Evaluation Report

Evaluated 2026-09-10. X-ray and CT results are intentionally separate.

- Rows are true classes; columns are predicted classes. In Condition B, `REJECTED` is an API rejection and is counted as an incorrect outcome in overall accuracy and class recall.
- Confidence statistics and calibration include only classified samples; rejected requests do not return class probabilities.
- Condition A uses the active checkpoint with the same modality-specific transform/model path used by offline evaluation (`train_xray_txrv.build_xrv_transform()` for X-ray; `dataset.build_transforms(False)` for CT). Condition B submits the exact same files as multipart `file` uploads to the actual FastAPI endpoints. The benchmark disabled only the 20/minute rate limiter, which otherwise makes a full 1,922-image run impractical; preprocessing, segmentation, gates, OOD, Grad-CAM, and endpoint code were unchanged. Groq report calls fell back after provider rate limiting; report generation does not affect classifier probabilities.

## Condition A — Offline held-out test split

### X-ray

```json
{
  "condition": "A_offline",
  "modality": "xray",
  "sample_size": {"total": 1922, "classified": 1922, "rejected": 0},
  "per_class_counts": {"COVID-19": 357, "Normal": 1368, "Pneumonia": 134, "Tuberculosis": 63},
  "accuracy": 0.965661,
  "per_class_metrics": {
    "COVID-19": {"precision": 0.925824, "recall": 0.943978, "f1": 0.934813},
    "Normal": {"precision": 0.982275, "recall": 0.972222, "f1": 0.977223},
    "Pneumonia": {"precision": 0.913043, "recall": 0.940299, "f1": 0.926471},
    "Tuberculosis": {"precision": 0.954545, "recall": 1.0, "f1": 0.976744}
  },
  "confusion_matrix": {
    "labels": ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"],
    "matrix": [[337, 16, 4, 0], [27, 1330, 8, 3], [0, 8, 126, 0], [0, 0, 0, 63]]
  },
  "confidence_stats": {
    "correct": {"mean": 0.965575, "median": 0.996821, "std": 0.078737},
    "incorrect": {"mean": 0.7175, "median": 0.698979, "std": 0.166199}
  },
  "calibration": {
    "0-50": {"n": 5, "accuracy": 0.6},
    "50-70": {"n": 81, "accuracy": 0.604938},
    "70-90": {"n": 158, "accuracy": 0.892405},
    "90-100": {"n": 1678, "accuracy": 0.991061}
  },
  "misclassified_examples": [
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1221.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.867371, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1242.png", "true": "COVID-19", "pred": "Pneumonia", "confidence": 0.99453, "near_miss": false},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1588.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.708501, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1595.png", "true": "COVID-19", "pred": "Pneumonia", "confidence": 0.980168, "near_miss": false},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-16.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.511875, "near_miss": false},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1601.png", "true": "COVID-19", "pred": "Pneumonia", "confidence": 0.416273, "near_miss": false},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-179.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.925517, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1848.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.538787, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1888.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.950628, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-2484.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.982595, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-2585.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.623864, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-2595.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.803648, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-2620.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.883042, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-2806.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.734576, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-29.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.636409, "near_miss": true}
  ],
  "rejection_log": []
}
```

### CT

```json
{
  "condition": "A_offline",
  "modality": "ct",
  "sample_size": {"total": 103, "classified": 103, "rejected": 0},
  "per_class_counts": {"Benign": 12, "Malignant": 50, "Normal": 41},
  "accuracy": 0.980583,
  "per_class_metrics": {
    "Benign": {"precision": 1.0, "recall": 0.833333, "f1": 0.909091},
    "Malignant": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    "Normal": {"precision": 0.953488, "recall": 1.0, "f1": 0.97619}
  },
  "confusion_matrix": {
    "labels": ["Benign", "Malignant", "Normal"],
    "matrix": [[10, 0, 2], [0, 50, 0], [0, 0, 41]]
  },
  "confidence_stats": {
    "correct": {"mean": 0.938636, "median": 0.986597, "std": 0.115129},
    "incorrect": {"mean": 0.869262, "median": 0.869262, "std": 0.096423}
  },
  "calibration": {
    "0-50": {"n": 1, "accuracy": 1.0},
    "50-70": {"n": 7, "accuracy": 1.0},
    "70-90": {"n": 7, "accuracy": 0.857143},
    "90-100": {"n": 88, "accuracy": 0.988636}
  },
  "misclassified_examples": [
    {"file": "data/ct_scan/test/Benign/Bengin case (50).jpg", "true": "Benign", "pred": "Normal", "confidence": 0.772839, "near_miss": true},
    {"file": "data/ct_scan/test/Benign/Bengin case (51).jpg", "true": "Benign", "pred": "Normal", "confidence": 0.965685, "near_miss": true}
  ],
  "rejection_log": []
}
```

## Condition B — Live FastAPI serving path, same held-out images

### X-ray

```json
{
  "condition": "B_live",
  "modality": "xray",
  "sample_size": {"total": 1922, "classified": 1909, "rejected": 13},
  "per_class_counts": {"COVID-19": 357, "Normal": 1368, "Pneumonia": 134, "Tuberculosis": 63},
  "accuracy": 0.91051,
  "per_class_metrics": {
    "COVID-19": {"precision": 0.783529, "recall": 0.932773, "f1": 0.851662},
    "Normal": {"precision": 0.989633, "recall": 0.907164, "f1": 0.946606},
    "Pneumonia": {"precision": 0.775, "recall": 0.925373, "f1": 0.843537},
    "Tuberculosis": {"precision": 0.742857, "recall": 0.825397, "f1": 0.781955}
  },
  "confusion_matrix": {
    "labels": ["COVID-19", "Normal", "Pneumonia", "Tuberculosis", "REJECTED"],
    "matrix": [[333, 9, 7, 2, 6], [82, 1241, 29, 16, 0], [6, 4, 124, 0, 0], [4, 0, 0, 52, 7], [0, 0, 0, 0, 0]]
  },
  "confidence_stats": {
    "correct": {"mean": 0.94126, "median": 0.99173, "std": 0.109013},
    "incorrect": {"mean": 0.717118, "median": 0.691289, "std": 0.174747}
  },
  "calibration": {
    "0-50": {"n": 26, "accuracy": 0.538462},
    "50-70": {"n": 157, "accuracy": 0.55414},
    "70-90": {"n": 246, "accuracy": 0.837398},
    "90-100": {"n": 1480, "accuracy": 0.975}
  },
  "misclassified_examples": [
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1221.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.797174, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1242.png", "true": "COVID-19", "pred": "Pneumonia", "confidence": 0.997802, "near_miss": false},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1313.png", "true": "COVID-19", "pred": "REJECTED", "confidence": null, "near_miss": null, "rejection_reason": "This does not look like a chest X-ray. (feature-space distance 11.7 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1361.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.638104, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1588.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.525476, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-159.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.599386, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1595.png", "true": "COVID-19", "pred": "Pneumonia", "confidence": 0.978703, "near_miss": false},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-16.png", "true": "COVID-19", "pred": "Pneumonia", "confidence": 0.555695, "near_miss": false},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1601.png", "true": "COVID-19", "pred": "Pneumonia", "confidence": 0.572782, "near_miss": false},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1732.png", "true": "COVID-19", "pred": "REJECTED", "confidence": null, "near_miss": null, "rejection_reason": "This does not look like a chest X-ray. (feature-space distance 9.1 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1783.png", "true": "COVID-19", "pred": "REJECTED", "confidence": null, "near_miss": null, "rejection_reason": "This does not look like a chest X-ray. (feature-space distance 11.1 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-179.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.882992, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1844.png", "true": "COVID-19", "pred": "Tuberculosis", "confidence": 0.513092, "near_miss": true},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-2324.png", "true": "COVID-19", "pred": "REJECTED", "confidence": null, "near_miss": null, "rejection_reason": "This does not look like a chest X-ray. (feature-space distance 9.5 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-2484.png", "true": "COVID-19", "pred": "Normal", "confidence": 0.983175, "near_miss": true}
  ],
  "rejection_log": [
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1313.png", "true": "COVID-19", "reason": "This does not look like a chest X-ray. (feature-space distance 11.7 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1732.png", "true": "COVID-19", "reason": "This does not look like a chest X-ray. (feature-space distance 9.1 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-1783.png", "true": "COVID-19", "reason": "This does not look like a chest X-ray. (feature-space distance 11.1 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-2324.png", "true": "COVID-19", "reason": "This does not look like a chest X-ray. (feature-space distance 9.5 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-309.png", "true": "COVID-19", "reason": "This does not look like a chest X-ray. (feature-space distance 8.9 exceeds threshold 8.7)"},
    {"file": "data/xray/test/COVID-19/covid19db_COVID-446.png", "true": "COVID-19", "reason": "This does not look like a chest X-ray. (feature-space distance 9.8 exceeds threshold 8.7)"},
    {"file": "data/xray/test/Tuberculosis/tbdb_Tuberculosis-128.png", "true": "Tuberculosis", "reason": "This does not look like a chest X-ray. (feature-space distance 9.0 exceeds threshold 8.7)"},
    {"file": "data/xray/test/Tuberculosis/tbdb_Tuberculosis-139.png", "true": "Tuberculosis", "reason": "This does not look like a chest X-ray. (feature-space distance 8.8 exceeds threshold 8.7)"},
    {"file": "data/xray/test/Tuberculosis/tbdb_Tuberculosis-140.png", "true": "Tuberculosis", "reason": "This does not look like a chest X-ray. (feature-space distance 8.8 exceeds threshold 8.7)"},
    {"file": "data/xray/test/Tuberculosis/tbdb_Tuberculosis-186.png", "true": "Tuberculosis", "reason": "This does not look like a chest X-ray. (feature-space distance 9.9 exceeds threshold 8.7)"},
    {"file": "data/xray/test/Tuberculosis/tbdb_Tuberculosis-194.png", "true": "Tuberculosis", "reason": "This does not look like a chest X-ray. (feature-space distance 9.6 exceeds threshold 8.7)"},
    {"file": "data/xray/test/Tuberculosis/tbdb_Tuberculosis-319.png", "true": "Tuberculosis", "reason": "This does not look like a chest X-ray. (feature-space distance 8.8 exceeds threshold 8.7)"},
    {"file": "data/xray/test/Tuberculosis/tbdb_Tuberculosis-678.png", "true": "Tuberculosis", "reason": "This does not look like a chest X-ray. (feature-space distance 9.3 exceeds threshold 8.7)"}
  ]
}
```

### CT

```json
{
  "condition": "B_live",
  "modality": "ct",
  "sample_size": {"total": 103, "classified": 103, "rejected": 0},
  "per_class_counts": {"Benign": 12, "Malignant": 50, "Normal": 41},
  "accuracy": 0.980583,
  "per_class_metrics": {
    "Benign": {"precision": 1.0, "recall": 0.833333, "f1": 0.909091},
    "Malignant": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    "Normal": {"precision": 0.953488, "recall": 1.0, "f1": 0.97619}
  },
  "confusion_matrix": {
    "labels": ["Benign", "Malignant", "Normal"],
    "matrix": [[10, 0, 2], [0, 50, 0], [0, 0, 41]]
  },
  "confidence_stats": {
    "correct": {"mean": 0.938647, "median": 0.986608, "std": 0.115093},
    "incorrect": {"mean": 0.869476, "median": 0.869476, "std": 0.096242}
  },
  "calibration": {
    "0-50": {"n": 1, "accuracy": 1.0},
    "50-70": {"n": 7, "accuracy": 1.0},
    "70-90": {"n": 8, "accuracy": 0.875},
    "90-100": {"n": 87, "accuracy": 0.988506}
  },
  "misclassified_examples": [
    {"file": "data/ct_scan/test/Benign/Bengin case (50).jpg", "true": "Benign", "pred": "Normal", "confidence": 0.773234, "near_miss": true},
    {"file": "data/ct_scan/test/Benign/Bengin case (51).jpg", "true": "Benign", "pred": "Normal", "confidence": 0.965719, "near_miss": true}
  ],
  "rejection_log": []
}
```

## Condition C — Real-world/external data

No eligible external real-world scans are present, so no Condition C JSON blocks are included. The checked `test/external_probe/{xray,ct}/` folders contain only `README.txt` placeholders. `sample images/` is not an external probe: the existing provenance scan (`test/results/provenance_sample.json`) finds **0 external files of 315** and identifies every file as an exact match in the raw/project datasets (including 10 test and 125 train/validation matches). `demo/` and `audit/` contain derived demonstration/audit artifacts, not independently sourced, labelled clinical data.

The observed comparison isolates a live-serving X-ray loss of 5.5151 percentage points (96.5661% offline vs. 91.0510% HTTP serving, including rejections); CT has no observed offline-to-serving loss on its 103-image test set.
