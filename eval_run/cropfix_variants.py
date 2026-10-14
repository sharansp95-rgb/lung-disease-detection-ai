"""Measure candidate live-crop variants against the offline (published-accuracy)
path, on (A) the 240-image data/xray/test sample used by run_pathdiff.py (seed 3)
and (B) the ORIGINAL uncropped source files used by run_rawprobe.py (seed 11).

Views:
  offline   : file straight into the XRV transform (= how 96.57% was measured)
  live      : compute_lung_crop() -> XRV transform            (today's serving path)
  lungsq    : compute_lung_crop() -> LANCZOS squash to 224x224 -> XRV transform  [fix A]
  bordersq  : border_crop(8%)     -> LANCZOS squash to 224x224 -> XRV transform  [TB-source training framing]
  wholesq   : whole image         -> LANCZOS squash to 224x224 -> XRV transform

Combination rules over {lungsq, bordersq} and {lungsq, bordersq, wholesq}.
"""
import sys, json, pathlib, random, collections
ROOT = pathlib.Path(r"d:/7th sem/sdp")
for p in ("backend", "ml", "scripts"):
    sys.path.insert(0, str(ROOT / p))
import numpy as np
import torch
from PIL import Image
from train_xray_txrv import build_xrv_transform, XRVClassifier
from seg_utils import load_seg_bundle, compute_lung_crop
from prepare_xray import border_crop, TB_BORDER_FRAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ROOT / "checkpoints/xray_txrv_densenet121_best.pt", map_location=device, weights_only=True)
CLASSES = ckpt["class_names"]
model = XRVClassifier(ckpt["backbone"], len(CLASSES)).to(device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
seg = load_seg_bundle(ROOT / "checkpoints/xray_seg_model.pt", ROOT / "seg_data")
tf = build_xrv_transform(train=False)


def sq224(pil):
    return Image.fromarray(np.array(pil.convert("L"))).resize((224, 224), Image.LANCZOS)


def probs(pil):
    with torch.no_grad():
        return torch.softmax(model(tf(pil.convert("L")).unsqueeze(0).to(device)), 1)[0].cpu().numpy()


def all_views(pil):
    lung, used_seg = compute_lung_crop(pil, seg)
    gray = np.array(pil.convert("L"))
    border = Image.fromarray(border_crop(gray, TB_BORDER_FRAC))
    return {
        "offline": probs(pil),
        "live": probs(lung),
        "lungsq": probs(sq224(lung)),
        "bordersq": probs(sq224(border)),
        "wholesq": probs(sq224(pil)),
    }, used_seg


def combos(v):
    out = {}
    out["ens2_mean"] = (v["lungsq"] + v["bordersq"]) / 2
    out["ens3_mean"] = (v["lungsq"] + v["bordersq"] + v["wholesq"]) / 3
    eps = 1e-9
    out["ens2_geo"] = np.exp((np.log(v["lungsq"] + eps) + np.log(v["bordersq"] + eps)) / 2)
    out["ens3_geo"] = np.exp((np.log(v["lungsq"] + eps) + np.log(v["bordersq"] + eps) + np.log(v["wholesq"] + eps)) / 3)
    out["ens2_maxconf"] = v["lungsq"] if v["lungsq"].max() >= v["bordersq"].max() else v["bordersq"]
    for k in ("ens2_geo", "ens3_geo"):
        out[k] = out[k] / out[k].sum()
    return out


VIEWS = ["offline", "live", "lungsq", "bordersq", "wholesq"]
COMBOS = ["ens2_mean", "ens3_mean", "ens2_geo", "ens3_geo", "ens2_maxconf"]
ALL = VIEWS + COMBOS

COVID_SRC = ROOT / "raw/xray_covid/COVID-19_Radiography_Dataset"
RAWDIR = {
    "COVID-19": [COVID_SRC / "COVID/images"],
    "Pneumonia": [COVID_SRC / "Viral Pneumonia/images"],
    "Normal": [COVID_SRC / "Normal/images", ROOT / "raw/xray_tb_cleaned/Normal"],
    "Tuberculosis": [ROOT / "raw/xray_tb_cleaned/Tuberculosis"],
}


def raw_for(cls, name):
    stem = pathlib.Path(name.replace("covid19db_", "").replace("tbdb_", "")).stem
    for d in RAWDIR[cls]:
        for ext in (".png", ".jpg", ".jpeg"):
            c = d / (stem + ext)
            if c.exists():
                return c
    return None


def run(mode, seed):
    random.seed(seed)
    rows = []
    for cls in CLASSES:
        fs = sorted((ROOT / "data/xray/test" / cls).glob("*"))
        random.shuffle(fs)
        n = 0
        for f in fs:
            if mode == "raw":
                src = raw_for(cls, f.name)
                if src is None:
                    continue
            else:
                src = f
            v, used_seg = all_views(Image.open(src))
            v.update(combos(v))
            rows.append({"true": cls, "file": f.name, "src": src.name, "used_seg": bool(used_seg),
                         **{k: CLASSES[int(v[k].argmax())] for k in ALL},
                         **{k + "_conf": round(float(v[k].max()), 4) for k in ALL}})
            n += 1
            if n >= 60:
                break
        print(f"  {mode} {cls}: {n}")
    return rows


def summarize(rows):
    out = {"n": len(rows), "overall": {}, "per_class": {}}
    for k in ALL:
        out["overall"][k] = round(sum(r[k] == r["true"] for r in rows) / len(rows), 4)
    for cls in CLASSES:
        sub = [r for r in rows if r["true"] == cls]
        out["per_class"][cls] = {"n": len(sub)}
        for k in ALL:
            out["per_class"][cls][k] = round(sum(r[k] == cls for r in sub) / len(sub), 4)
        out["per_class"][cls]["live_preds"] = dict(collections.Counter(r["live"] for r in sub))
        out["per_class"][cls]["lungsq_preds"] = dict(collections.Counter(r["lungsq"] for r in sub))
        out["per_class"][cls]["ens2_mean_preds"] = dict(collections.Counter(r["ens2_mean"] for r in sub))
    return out


result = {}
for mode, seed in (("test", 3), ("raw", 11)):
    print(f"=== {mode} ===")
    rows = run(mode, seed)
    result[mode] = {"summary": summarize(rows), "rows": rows}
    print(json.dumps(result[mode]["summary"]["overall"], indent=2))
    for cls in CLASSES:
        pc = result[mode]["summary"]["per_class"][cls]
        print(" ", cls, {k: pc[k] for k in ALL})

(ROOT / "eval_run" / "results_cropfix_variants.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("saved eval_run/results_cropfix_variants.json")
