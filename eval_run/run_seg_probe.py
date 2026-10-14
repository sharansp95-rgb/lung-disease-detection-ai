"""Section 1E out-of-band probe: the API never reports whether the U-Net crop
or the border-crop fallback fired (main.py discards compute_lung_crop's second
return value and logs nothing), so call it directly, read-only."""
import sys, json, pathlib, random
ROOT = pathlib.Path(r"d:/7th sem/sdp")
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "scripts"))
from PIL import Image
from seg_utils import load_seg_bundle, compute_lung_crop

bundle = load_seg_bundle(ROOT / "checkpoints" / "xray_seg_model.pt", ROOT / "seg_data")
random.seed(1)
rows = []


def probe(tag, path):
    img = Image.open(path)
    w, h = img.size
    out, used = compute_lung_crop(img, bundle)
    rows.append({"group": tag, "file": pathlib.Path(path).name, "orig": [w, h],
                 "cropped": list(out.size), "used_segmentation": used})
    return used


for cls in ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]:
    fs = sorted((ROOT / "data/xray/test" / cls).glob("*"))
    random.shuffle(fs)
    for f in fs[:15]:
        probe(cls, f)

for f in sorted((ROOT / "eval_run/fixtures").glob("seg_*.png")):
    probe("stressor", f)

lo = sorted((ROOT / "seg_data/val/images").glob("lung_opacity-*"))[:10]
for f in lo:
    probe("lung_opacity", f)

summary = {}
for r in rows:
    g = summary.setdefault(r["group"], {"n": 0, "seg": 0, "fallback": 0})
    g["n"] += 1
    g["seg" if r["used_segmentation"] else "fallback"] += 1
(ROOT / "eval_run" / "results_seg_probe.json").write_text(
    json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
