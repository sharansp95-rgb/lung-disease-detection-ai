"""Section 1G: concurrency. 12 simultaneous mixed /predict calls, compared
against the same 12 run sequentially (predictions + exact heatmap bytes)."""
import json, pathlib, hashlib, time, sys
from concurrent.futures import ThreadPoolExecutor
import requests

B = "http://127.0.0.1:8000"
ROOT = pathlib.Path(r"d:/7th sem/sdp")

files = []
for cls in ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]:
    files.append(("xray", sorted((ROOT / "data/xray/test" / cls).glob("*"))[0]))
for cls in ["Benign", "Malignant", "Normal"]:
    for i in range(2):
        files.append(("ct", sorted((ROOT / "data/ct_scan/test" / cls).glob("*"))[i]))
files.append(("xray", sorted((ROOT / "data/xray/test/COVID-19").glob("*"))[1]))
files.append(("xray", sorted((ROOT / "data/xray/test/Normal").glob("*"))[1]))

def call(job):
    mod, f = job
    t0 = time.time()
    r = requests.post(f"{B}/predict/{mod}", files={"file": (f.name, f.read_bytes(), "image/png")}, timeout=300)
    b = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    hm = b.get("heatmap_png_base64") or ""
    return {"mod": mod, "file": f.name, "status": r.status_code, "latency_s": round(time.time() - t0, 3),
            "pred": b.get("predicted_class"), "conf": b.get("confidence"),
            "probs": b.get("class_probabilities"),
            "heatmap_sha": hashlib.sha256(hm.encode()).hexdigest()[:16], "heatmap_len": len(hm),
            "detail": b.get("detail")}

t0 = time.time()
with ThreadPoolExecutor(max_workers=12) as ex:
    conc = list(ex.map(call, files))
wall = round(time.time() - t0, 2)
print("concurrent wall:", wall, flush=True)
for r in conc:
    print(r["mod"], r["file"], r["status"], r["pred"], r["conf"], r["heatmap_sha"], flush=True)

print("waiting 65s for the rate-limit window to clear...", flush=True)
time.sleep(65)

seq = []
for j in files:
    seq.append(call(j))
    time.sleep(1)
print("sequential done", flush=True)

diffs = []
for a, b in zip(conc, seq):
    if a["status"] != b["status"] or a["pred"] != b["pred"] or a["probs"] != b["probs"] or a["heatmap_sha"] != b["heatmap_sha"]:
        diffs.append({"file": a["file"], "concurrent": a, "sequential": b})
uniq_hm = len({r["heatmap_sha"] for r in conc if r["heatmap_len"]})
out = {"concurrent_wall_s": wall, "concurrent": conc, "sequential": seq,
       "mismatches": diffs, "n_mismatch": len(diffs),
       "unique_heatmaps_concurrent": uniq_hm, "n_ok": sum(1 for r in conc if r["status"] == 200)}
(ROOT / "eval_run" / "results_concurrency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("mismatches:", len(diffs), "unique heatmaps:", uniq_hm, flush=True)
