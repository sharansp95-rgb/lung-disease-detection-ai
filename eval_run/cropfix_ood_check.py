"""Did the crop fix change how often the X-ray OOD detector rejects genuine
test images? Compares the feature-space distance under the OLD live crop
(raw bbox, non-square -> XRayCenterCrop amputates) vs the NEW one
(bbox squashed to 224x224, training geometry), on the same 240-image sample."""
import sys, json, pathlib, random
ROOT = pathlib.Path(r"d:/7th sem/sdp")
for p in ("backend", "ml", "scripts"):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, torch
from PIL import Image
from train_xray_txrv import XRVClassifier
from models import load_model_bundle
from gradcam_utils import preprocess_image_xrv
from ood_detector import OODDetector
from seg_utils import load_seg_bundle, compute_lung_crop, SEG_INPUT_SIZE, _to_seg_tensor, \
    MIN_BBOX_AREA_FRAC, MAX_BBOX_AREA_FRAC
from prepare_xray import border_crop, mask_bbox, MASK_PAD_PX, TB_BORDER_FRAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_model_bundle("xray", ROOT / "checkpoints/xray_txrv_densenet121_best.pt", ROOT / "data/xray")
ood = OODDetector("xray", device)
seg = load_seg_bundle(ROOT / "checkpoints/xray_seg_model.pt", ROOT / "seg_data")


def old_crop(image):
    """compute_lung_crop() as it was BEFORE the fix: returns the raw bbox."""
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
    return Image.fromarray(gray[int(r0 * sr):int(r1 * sr), int(c0 * sc):int(c1 * sc)])


def dist(pil):
    t = preprocess_image_xrv(pil, device)
    ok, d = ood.check(bundle.model, t)
    return bool(ok), float(d)


random.seed(3)
out = {}
for cls in bundle.class_names:
    fs = sorted((ROOT / "data/xray/test" / cls).glob("*"))
    random.shuffle(fs)
    rows = []
    for f in fs[:60]:
        img = Image.open(f)
        ok_o, d_o = dist(old_crop(img))
        ok_n, d_n = dist(compute_lung_crop(img, seg)[0])
        rows.append({"f": f.name, "old_ok": ok_o, "old_d": round(d_o, 3), "new_ok": ok_n, "new_d": round(d_n, 3)})
    out[cls] = {"n": len(rows),
                "old_rejects": sum(not r["old_ok"] for r in rows),
                "new_rejects": sum(not r["new_ok"] for r in rows),
                "old_mean_d": round(float(np.mean([r["old_d"] for r in rows])), 3),
                "new_mean_d": round(float(np.mean([r["new_d"] for r in rows])), 3),
                "rows": rows}
    print(cls, {k: out[cls][k] for k in ("n", "old_rejects", "new_rejects", "old_mean_d", "new_mean_d")})
(ROOT / "eval_run" / "results_cropfix_ood.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
