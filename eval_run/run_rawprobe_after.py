"""The realistic-upload test: take the ORIGINAL, uncropped source images that
correspond to files in the held-out test split, and run them through the exact
live serving path (lung crop -> XRV transform -> classifier). This is what a
real user uploads; data/xray/test/ images are already crop-preprocessed."""
import sys, json, pathlib, random, collections
ROOT = pathlib.Path(r"d:/7th sem/sdp")
for p in ("backend", "ml", "scripts"):
    sys.path.insert(0, str(ROOT / p))
import torch
from PIL import Image
from train_xray_txrv import build_xrv_transform, XRVClassifier
from seg_utils import load_seg_bundle, compute_lung_crop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ROOT / "checkpoints/xray_txrv_densenet121_best.pt", map_location=device, weights_only=True)
classes = ckpt["class_names"]
model = XRVClassifier(ckpt["backbone"], len(classes)).to(device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
seg = load_seg_bundle(ROOT / "checkpoints/xray_seg_model.pt", ROOT / "seg_data")
tf = build_xrv_transform(train=False)

RAWDIR = {
    "COVID-19": [ROOT / "raw/xray_covid/COVID-19_Radiography_Dataset/COVID/images"],
    "Pneumonia": [ROOT / "raw/xray_covid/COVID-19_Radiography_Dataset/Viral Pneumonia/images"],
    "Normal": [ROOT / "raw/xray_covid/COVID-19_Radiography_Dataset/Normal/images",
               ROOT / "raw/xray_tb_cleaned/Normal"],
    "Tuberculosis": [ROOT / "raw/xray_tb_cleaned/Tuberculosis"],
}


def raw_for(cls, name):
    stem = name.replace("covid19db_", "").replace("tbdb_", "")
    for d in RAWDIR[cls]:
        for ext in (".png", ".jpg", ".jpeg"):
            c = d / (pathlib.Path(stem).stem + ext)
            if c.exists():
                return c
    return None


def infer(img):
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0).to(device)), 1)[0].cpu().numpy()
    return classes[int(probs.argmax())], float(probs.max())


random.seed(11)
rows, missing = [], collections.Counter()
for cls in classes:
    fs = sorted((ROOT / "data/xray/test" / cls).glob("*"))
    random.shuffle(fs)
    n = 0
    for f in fs:
        raw = raw_for(cls, f.name)
        if raw is None:
            missing[cls] += 1
            continue
        img = Image.open(raw)
        cropped, used_seg = compute_lung_crop(img, seg)
        pred, conf = infer(cropped.convert("L"))
        pred_nocrop, conf_nc = infer(img.convert("L"))
        rows.append({"true": cls, "raw": str(raw.name), "serving_pred": pred, "serving_conf": round(conf, 4),
                     "nocrop_pred": pred_nocrop, "nocrop_conf": round(conf_nc, 4), "used_seg": used_seg})
        n += 1
        if n >= 60:
            break

summary = {}
for cls in classes:
    sub = [r for r in rows if r["true"] == cls]
    summary[cls] = {"n": len(sub),
                    "serving_acc": round(sum(r["serving_pred"] == cls for r in sub) / len(sub), 4) if sub else None,
                    "serving_preds": dict(collections.Counter(r["serving_pred"] for r in sub)),
                    "nocrop_acc": round(sum(r["nocrop_pred"] == cls for r in sub) / len(sub), 4) if sub else None,
                    "nocrop_preds": dict(collections.Counter(r["nocrop_pred"] for r in sub)),
                    "seg_used_frac": round(sum(r["used_seg"] for r in sub) / len(sub), 3) if sub else None}
overall = {"n": len(rows),
           "serving_acc": round(sum(r["serving_pred"] == r["true"] for r in rows) / len(rows), 4),
           "nocrop_acc": round(sum(r["nocrop_pred"] == r["true"] for r in rows) / len(rows), 4),
           "unmatched_raw_files": dict(missing)}
(ROOT / "eval_run" / "results_rawprobe_after.json").write_text(
    json.dumps({"overall": overall, "per_class": summary, "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps({"overall": overall, "per_class": summary}, indent=2))
