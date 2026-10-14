"""Sections B (wrong modality), C (non-scan), E (seg stressors), D (Lung_Opacity)."""
import sys, json, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from harness import predict, slim, save, ROOT

random.seed(7)
FIX = ROOT / "eval_run" / "fixtures"
results = {}

# --- B: wrong modality ---
ct_imgs = sorted((ROOT / "data/ct_scan/test/Malignant").glob("*"))[:3]
xr_imgs = sorted((ROOT / "data/xray/test/Normal").glob("*"))[:3]
b = []
for p in ct_imgs:
    b.append(slim(predict("xray", p)))
for p in xr_imgs:
    b.append(slim(predict("ct", p)))
results["B_wrong_modality"] = b
print("B done", [(r["modality"], r["status"]) for r in b])

# --- C: non-scan inputs ---
fixtures = ["photo_landscape.png", "photo_object.png", "photo_grayscale.png",
            "screenshot_app.png", "scanned_document.png", "solid_gray.png",
            "solid_blue.png", "truncated_xray.png", "not_an_image.png",
            "empty.png", "random_noise.png", "near_blank.png"]
c = []
for f in fixtures:
    for m in ("xray", "ct"):
        c.append(slim(predict(m, FIX / f)))
        print("C", f, m, c[-1]["status"], str(c[-1]["body"].get("detail"))[:80])
results["C_non_scan"] = c

# --- E: seg stressors (HTTP-visible behaviour only) ---
e = []
for f in ["seg_zoomed_center.png", "seg_wide_aspect.png", "seg_tall_aspect.png", "seg_mostly_blank.png"]:
    e.append(slim(predict("xray", FIX / f)))
    print("E", f, e[-1]["status"], e[-1]["body"].get("predicted_class"), e[-1]["body"].get("confidence"))
results["E_seg_stressors"] = e

# --- D: Lung_Opacity (real chest X-rays, class excluded from the classifier) ---
lo = sorted((ROOT / "seg_data/val/images").glob("lung_opacity-*"))
random.shuffle(lo)
d = []
for p in lo[:10]:
    d.append(slim(predict("xray", p)))
    print("D", p.name, d[-1]["status"], d[-1]["body"].get("predicted_class"), d[-1]["body"].get("confidence"))
results["D_lung_opacity"] = d

save("results_gates.json", results)
print("saved")
