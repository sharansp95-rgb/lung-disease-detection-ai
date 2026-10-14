"""Builds confusion matrices + per-class P/R/F1 from the bulk HTTP sweep."""
import json, pathlib, collections, statistics
ROOT = pathlib.Path(r"d:/7th sem/sdp")
rows = []
for tag in ("A", "B"):
    p = ROOT / "eval_run" / f"results_bulk_{tag}.json"
    if p.exists():
        rows += json.loads(p.read_text(encoding="utf-8"))

CLASSES = {"xray": ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"],
           "ct": ["Benign", "Malignant", "Normal"]}
report = {}
for mod, classes in CLASSES.items():
    sub = [r for r in rows if r["modality"] == mod]
    if not sub:
        continue
    non200 = [r for r in sub if r["status"] != 200]
    ok = [r for r in sub if r["status"] == 200]
    cm = {t: collections.Counter() for t in classes}
    for r in ok:
        cm[r["true"]][r["pred"]] += 1
    per = {}
    for c in classes:
        tp = cm[c][c]
        fn = sum(cm[c].values()) - tp
        fp = sum(cm[o][c] for o in classes if o != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"n_tested": sum(cm[c].values()), "tp": tp, "precision": round(prec, 4),
                  "recall": round(rec, 4), "f1": round(f1, 4)}
    total = sum(sum(cm[c].values()) for c in classes)
    correct = sum(cm[c][c] for c in classes)
    lat = [r["latency_s"] for r in ok]
    report[mod] = {
        "n_requested": len(sub), "n_http_200": len(ok), "n_non_200": len(non200),
        "non_200_detail": collections.Counter((r["status"], str(r["detail"])[:60]) for r in non200).most_common(),
        "confusion_matrix_rows_true_cols_pred": {c: {p: cm[c][p] for p in classes} for c in classes},
        "accuracy_on_200s": round(correct / total, 4) if total else None,
        "accuracy_counting_non200_as_wrong": round(correct / len(sub), 4) if sub else None,
        "per_class": per,
        "macro_f1": round(sum(per[c]["f1"] for c in classes) / len(classes), 4),
        "latency_s": {"n": len(lat), "mean": round(statistics.mean(lat), 3),
                      "median": round(statistics.median(lat), 3),
                      "p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3),
                      "min": round(min(lat), 3), "max": round(max(lat), 3)} if lat else None,
        "report_sources": collections.Counter(r["report_source"] for r in ok),
        "n_missing_heatmap": sum(1 for r in ok if not r["heatmap_len"]),
        "n_missing_report": sum(1 for r in ok if not r["report_len"]),
        "n_with_warnings": sum(1 for r in ok if r["warnings"]),
        "mean_confidence": round(statistics.mean([r["conf"] for r in ok]), 4) if ok else None,
    }
    # prediction distribution overall
    report[mod]["pred_distribution"] = collections.Counter(r["pred"] for r in ok)

(ROOT / "eval_run" / "analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print(json.dumps(report, indent=2, default=str))
