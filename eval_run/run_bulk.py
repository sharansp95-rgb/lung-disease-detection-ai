"""Section 1A: in-distribution correctness sweep. Usage: run_bulk.py <port> <tag>"""
import sys, json, pathlib, random, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import harness
from harness import slim, ROOT

port = sys.argv[1]
tag = sys.argv[2]
harness.BASE = f"http://127.0.0.1:{port}"

random.seed(42)

PLAN_A = {("xray", "COVID-19"): None}                      # all
PLAN_B = {("xray", "Normal"): 40, ("xray", "Pneumonia"): 25, ("xray", "Tuberculosis"): 25,
          ("ct", "Benign"): None, ("ct", "Malignant"): 25, ("ct", "Normal"): 25}
plan = PLAN_A if tag == "A" else PLAN_B

DIR = {"xray": ROOT / "data/xray/test", "ct": ROOT / "data/ct_scan/test"}

rows = []
outpath = ROOT / "eval_run" / f"results_bulk_{tag}.json"
for (mod, cls), n in plan.items():
    files = sorted((DIR[mod] / cls).glob("*"))
    if n is not None:
        random.shuffle(files)
        files = files[:n]
    for i, f in enumerate(files):
        r = harness.predict(mod, f)
        b = r["body"]
        rows.append({
            "modality": mod, "true": cls, "file": f.name, "status": r["status"],
            "latency_s": r["latency_s"],
            "pred": b.get("predicted_class"), "conf": b.get("confidence"),
            "probs": b.get("class_probabilities"),
            "heatmap_len": len(b.get("heatmap_png_base64") or ""),
            "report_len": len(b.get("report") or ""), "report_source": b.get("report_source"),
            "warnings": b.get("warnings"), "detail": b.get("detail"),
        })
        if i % 10 == 0:
            print(f"[{tag}] {mod}/{cls} {i+1}/{len(files)} -> {rows[-1]['status']} {rows[-1]['pred']}", flush=True)
            outpath.write_text(json.dumps(rows, indent=1), encoding="utf-8")
outpath.write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("DONE", tag, len(rows), flush=True)
