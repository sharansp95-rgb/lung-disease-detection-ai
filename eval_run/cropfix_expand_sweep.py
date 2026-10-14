"""Secondary sweep: with the training-matched squash-to-224 in place (fix A),
does loosening the U-Net lung bbox help match the training crop-tightness
mixture (COVID-source = tight lung bbox, TB-source = loose 8% border crop)?"""
import sys, json, pathlib, random
ROOT = pathlib.Path(r"d:/7th sem/sdp")
for p in ("backend", "ml", "scripts"):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, torch
from PIL import Image
from train_xray_txrv import build_xrv_transform, XRVClassifier
from seg_utils import load_seg_bundle, SEG_INPUT_SIZE, _to_seg_tensor, MIN_BBOX_AREA_FRAC, MAX_BBOX_AREA_FRAC
from prepare_xray import border_crop, mask_bbox, MASK_PAD_PX, TB_BORDER_FRAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ROOT / "checkpoints/xray_txrv_densenet121_best.pt", map_location=device, weights_only=True)
CLASSES = ckpt["class_names"]
model = XRVClassifier(ckpt["backbone"], len(CLASSES)).to(device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
seg = load_seg_bundle(ROOT / "checkpoints/xray_seg_model.pt", ROOT / "seg_data")
tf = build_xrv_transform(train=False)
EXPANDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def crop_expand(image, e):
    with seg.lock:
        x = _to_seg_tensor(image, seg.device)
        with torch.no_grad():
            m = (torch.sigmoid(seg.model(x))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    gray = np.array(image.convert("L"))
    if m.max() == 0:
        return Image.fromarray(border_crop(gray, TB_BORDER_FRAC))
    r0, r1, c0, c1 = mask_bbox(m, pad=MASK_PAD_PX)
    frac = ((r1 - r0) * (c1 - c0)) / (SEG_INPUT_SIZE ** 2)
    if frac < MIN_BBOX_AREA_FRAC or frac > MAX_BBOX_AREA_FRAC:
        return Image.fromarray(border_crop(gray, TB_BORDER_FRAC))
    H, W = gray.shape
    sr, sc = H / SEG_INPUT_SIZE, W / SEG_INPUT_SIZE
    a0, a1, b0, b1 = int(r0 * sr), int(r1 * sr), int(c0 * sc), int(c1 * sc)
    dh, dw = int((a1 - a0) * e), int((b1 - b0) * e)
    a0, a1 = max(0, a0 - dh), min(H, a1 + dh)
    b0, b1 = max(0, b0 - dw), min(W, b1 + dw)
    return Image.fromarray(gray[a0:a1, b0:b1])


def pred(pil):
    sq = Image.fromarray(np.array(pil.convert("L"))).resize((224, 224), Image.LANCZOS)
    with torch.no_grad():
        p = torch.softmax(model(tf(sq).unsqueeze(0).to(device)), 1)[0].cpu().numpy()
    return CLASSES[int(p.argmax())]


COVID_SRC = ROOT / "raw/xray_covid/COVID-19_Radiography_Dataset"
RAWDIR = {"COVID-19": [COVID_SRC / "COVID/images"], "Pneumonia": [COVID_SRC / "Viral Pneumonia/images"],
          "Normal": [COVID_SRC / "Normal/images", ROOT / "raw/xray_tb_cleaned/Normal"],
          "Tuberculosis": [ROOT / "raw/xray_tb_cleaned/Tuberculosis"]}


def raw_for(cls, name):
    stem = pathlib.Path(name.replace("covid19db_", "").replace("tbdb_", "")).stem
    for d in RAWDIR[cls]:
        for ext in (".png", ".jpg", ".jpeg"):
            c = d / (stem + ext)
            if c.exists():
                return c
    return None


res = {}
for mode, seed in (("test", 3), ("raw", 11)):
    random.seed(seed)
    rows = []
    for cls in CLASSES:
        fs = sorted((ROOT / "data/xray/test" / cls).glob("*"))
        random.shuffle(fs)
        n = 0
        for f in fs:
            src = f if mode == "test" else raw_for(cls, f.name)
            if src is None:
                continue
            img = Image.open(src)
            rows.append({"true": cls, **{f"e{e}": pred(crop_expand(img, e)) for e in EXPANDS}})
            n += 1
            if n >= 60:
                break
    res[mode] = {"overall": {f"e{e}": round(sum(r[f"e{e}"] == r["true"] for r in rows) / len(rows), 4) for e in EXPANDS},
                 "per_class": {cls: {f"e{e}": round(sum(r[f"e{e}"] == cls for r in rows if r["true"] == cls) / 60, 4)
                                     for e in EXPANDS} for cls in CLASSES}}
    print(mode, json.dumps(res[mode], indent=2))
(ROOT / "eval_run" / "results_cropfix_expand.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
