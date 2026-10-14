"""Post-fix LIVE validation over HTTP /predict/xray.
  (1) the same seed-3 60/class stratified data/xray/test sample run_pathdiff.py used
  (2) the 10 external Tuberculosis samples from eval_run/external_samples/xray/tb/
Usage: cropfix_live_run.py <port>
"""
import sys, json, pathlib, random, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import harness
from harness import ROOT

harness.BASE = f"http://127.0.0.1:{sys.argv[1]}"
CLASSES = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]

random.seed(3)  # identical sample to run_pathdiff.py
rows = []
for cls in CLASSES:
    fs = sorted((ROOT / "data/xray/test" / cls).glob("*"))
    random.shuffle(fs)
    for i, f in enumerate(fs[:60]):
        r = harness.predict("xray", f)
        b = r["body"]
        rows.append({"set": "dataset_test", "true": cls, "file": f.name, "status": r["status"],
                     "pred": b.get("predicted_class"), "conf": b.get("confidence"),
                     "detail": b.get("detail"), "latency_s": r["latency_s"]})
        if i % 15 == 0:
            print(f"{cls} {i+1}/60 -> {r['status']} {b.get('predicted_class')}", flush=True)

ext = sorted((ROOT / "eval_run/external_samples/xray/tb").glob("*"))
for f in ext:
    r = harness.predict("xray", f)
    b = r["body"]
    rows.append({"set": "external_tb", "true": "Tuberculosis", "file": f.name, "status": r["status"],
                 "pred": b.get("predicted_class"), "conf": b.get("confidence"),
                 "detail": b.get("detail"), "latency_s": r["latency_s"]})
    print(f"ext_tb {f.name} -> {r['status']} {b.get('predicted_class')}", flush=True)


def summarize(subset):
    sub = [r for r in rows if r["set"] == subset]
    served = [r for r in sub if r["status"] == 200]
    out = {"n": len(sub), "served": len(served), "rejected": len(sub) - len(served),
           "reject_statuses": dict(collections.Counter(r["status"] for r in sub if r["status"] != 200)),
           "acc_over_served": round(sum(r["pred"] == r["true"] for r in served) / len(served), 4) if served else None,
           "acc_counting_rejects_as_wrong": round(sum(r["pred"] == r["true"] for r in served) / len(sub), 4) if sub else None,
           "confusion": {}, "per_class": {}}
    for cls in CLASSES:
        cs = [r for r in sub if r["true"] == cls]
        if not cs:
            continue
        cserved = [r for r in cs if r["status"] == 200]
        out["confusion"][cls] = dict(collections.Counter(
            (r["pred"] if r["status"] == 200 else f"REJECTED_{r['status']}") for r in cs))
        out["per_class"][cls] = {
            "n": len(cs),
            "recall_over_served": round(sum(r["pred"] == cls for r in cserved) / len(cserved), 4) if cserved else None,
            "recall_counting_rejects": round(sum(r["pred"] == cls for r in cserved) / len(cs), 4)}
    # precision over served, dataset-level
    for cls in CLASSES:
        pos = [r for r in served if r["pred"] == cls]
        if cls in out["per_class"]:
            out["per_class"][cls]["precision_over_served"] = round(
                sum(r["true"] == cls for r in pos) / len(pos), 4) if pos else None
    return out


summary = {s: summarize(s) for s in ("dataset_test", "external_tb")}
(ROOT / "eval_run" / "results_cropfix_live.json").write_text(
    json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
