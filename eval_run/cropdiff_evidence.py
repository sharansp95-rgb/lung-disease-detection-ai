"""Step-1 evidence: pixel-level side-by-side of the TRAINING-TIME crop
(scripts/prepare_xray.py) vs the LIVE crop (backend/seg_utils.compute_lung_crop),
AND -- crucially -- what the XRV transform actually feeds the classifier in each
case (XRayCenterCrop lops a non-square live crop back to a square).

Writes eval_run/xray_cropfix_backups/cropdiff_<name>.png
"""
import sys, pathlib, json
ROOT = pathlib.Path(r"d:/7th sem/sdp")
for p in ("backend", "ml", "scripts"):
    sys.path.insert(0, str(ROOT / p))
import numpy as np
import torch
from PIL import Image, ImageDraw

from seg_utils import load_seg_bundle, compute_lung_crop
from prepare_xray import process_covid, process_tb, border_crop, TB_BORDER_FRAC
from train_xray_txrv import build_xrv_transform

OUT = ROOT / "eval_run" / "xray_cropfix_backups"
OUT.mkdir(parents=True, exist_ok=True)
tf = build_xrv_transform(train=False)
seg = load_seg_bundle(ROOT / "checkpoints/xray_seg_model.pt", ROOT / "seg_data")

COVID_SRC = ROOT / "raw/xray_covid/COVID-19_Radiography_Dataset"
RAW = {
    "COVID-19": [(COVID_SRC / "COVID/images", COVID_SRC / "COVID/masks")],
    "Pneumonia": [(COVID_SRC / "Viral Pneumonia/images", COVID_SRC / "Viral Pneumonia/masks")],
    "Normal": [(COVID_SRC / "Normal/images", COVID_SRC / "Normal/masks"),
               (ROOT / "raw/xray_tb_cleaned/Normal", None)],
    "Tuberculosis": [(ROOT / "raw/xray_tb_cleaned/Tuberculosis", None)],
}


def raw_for(cls, name):
    stem = pathlib.Path(name.replace("covid19db_", "").replace("tbdb_", "")).stem
    for d, mdir in RAW[cls]:
        for ext in (".png", ".jpg", ".jpeg"):
            c = d / (stem + ext)
            if c.exists():
                return c, mdir
    return None, None


def xrv_view(pil):
    """Exactly what the classifier sees, rendered back to a viewable uint8 image."""
    t = tf(pil.convert("L")).numpy()[0]
    a = np.clip((t / 1024.0 + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def panel(img, label, size=224):
    img = img.convert("L").resize((size, size), Image.NEAREST).convert("RGB")
    strip = Image.new("RGB", (size, size + 22), (20, 20, 20))
    strip.paste(img, (0, 22))
    ImageDraw.Draw(strip).text((3, 5), label, fill=(255, 220, 120))
    return strip


CASES = [
    ("Tuberculosis", "tbdb_Tuberculosis-171.png"),
    ("Tuberculosis", "tbdb_Tuberculosis-5.png"),
    ("Tuberculosis", "tbdb_Tuberculosis-85.png"),
    ("Tuberculosis", "tbdb_Tuberculosis-121.png"),
    ("Normal", "tbdb_Normal-2581.png"),
    ("Normal", "tbdb_Normal-312.png"),
    ("Normal", "covid19db_Normal-111.png"),
    ("Normal", "covid19db_Normal-879.png"),
    ("Pneumonia", "covid19db_Viral Pneumonia-49.png"),
    ("Pneumonia", "covid19db_Viral Pneumonia-491.png"),
    ("COVID-19", "covid19db_COVID-3467.png"),
]

meta = []
for cls, name in CASES:
    rawp, mdir = raw_for(cls, name)
    if rawp is None:
        print("MISSING raw for", name)
        continue
    raw = Image.open(rawp)
    # (a) training-time build from the raw source
    train_img = process_covid(rawp, mdir, []) if mdir is not None else process_tb(rawp)
    # (b) live crop today
    live_img, used_seg = compute_lung_crop(raw, seg)
    # (c) proposed: live crop, then the SAME squash-to-224 prepare_xray applies
    fixed_img = Image.fromarray(np.array(live_img.convert("L"))).resize((224, 224), Image.LANCZOS)

    panels = [
        panel(raw, "raw upload"),
        panel(train_img, "TRAIN crop (prepare_xray)"),
        panel(xrv_view(train_img), "-> model sees (train)"),
        panel(live_img, f"LIVE crop {live_img.size} seg={used_seg}"),
        panel(xrv_view(live_img), "-> model sees (LIVE, cropped!)"),
        panel(xrv_view(fixed_img), "-> model sees (FIXED)"),
    ]
    W = sum(p.width for p in panels) + 6 * (len(panels) - 1)
    canvas = Image.new("RGB", (W, panels[0].height + 20), (10, 10, 10))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 20)); x += p.width + 6
    ImageDraw.Draw(canvas).text((4, 5), f"{cls}  {name}   raw={raw.size}  live_crop={live_img.size}", fill=(160, 255, 160))
    out = OUT / f"cropdiff_{pathlib.Path(name).stem}.png"
    canvas.save(out)
    meta.append({"true": cls, "file": name, "raw_size": raw.size, "train_crop_size": train_img.size,
                 "live_crop_size": live_img.size, "used_seg": used_seg,
                 "live_aspect": round(live_img.size[0] / live_img.size[1], 3), "png": out.name})
    print("wrote", out.name, live_img.size, "seg=", used_seg)

(OUT / "cropdiff_meta.json").write_text(json.dumps(meta, indent=2))
