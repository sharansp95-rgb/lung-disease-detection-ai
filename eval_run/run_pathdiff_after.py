"""Root-cause probe for the documented open bug: compare the OFFLINE eval path
(what the checkpoint's reported test accuracy was measured with) against the
LIVE SERVING path (segmentation/border crop, then the same XRV transform) on
identical images. Read-only; no HTTP, so no rate-limit cost."""
import sys, json, pathlib, random, collections
ROOT = pathlib.Path(r"d:/7th sem/sdp")
for p in ("backend", "ml", "scripts"):
    sys.path.insert(0, str(ROOT / p))
import torch
from PIL import Image
from models import load_model_bundle  # noqa
from train_xray_txrv import build_xrv_transform, xrv_loader, XRVClassifier
from seg_utils import load_seg_bundle, compute_lung_crop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ROOT / "checkpoints/xray_txrv_densenet121_best.pt", map_location=device, weights_only=True)
classes = ckpt["class_names"]
model = XRVClassifier(ckpt["backbone"], len(classes)).to(device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
seg = load_seg_bundle(ROOT / "checkpoints/xray_seg_model.pt", ROOT / "seg_data")
tf = build_xrv_transform(train=False)


def infer(img):
    with torch.no_grad():
        logits = model(tf(img).unsqueeze(0).to(device))
        probs = torch.softmax(logits, 1)[0].cpu().numpy()
    return classes[int(probs.argmax())], float(probs.max())


random.seed(3)
rows = []
for cls in classes:
    fs = sorted((ROOT / "data/xray/test" / cls).glob("*"))
    random.shuffle(fs)
    for f in fs[:60]:
        off_pred, off_conf = infer(xrv_loader(str(f)))
        cropped, used_seg = compute_lung_crop(Image.open(f), seg)
        srv_pred, srv_conf = infer(cropped.convert("L"))
        rows.append({"true": cls, "file": f.name, "offline": off_pred, "offline_conf": round(off_conf, 4),
                     "serving": srv_pred, "serving_conf": round(srv_conf, 4), "used_seg": used_seg})

summary = {}
for cls in classes:
    sub = [r for r in rows if r["true"] == cls]
    summary[cls] = {
        "n": len(sub),
        "offline_acc": round(sum(r["offline"] == cls for r in sub) / len(sub), 4),
        "serving_acc": round(sum(r["serving"] == cls for r in sub) / len(sub), 4),
        "offline_preds": dict(collections.Counter(r["offline"] for r in sub)),
        "serving_preds": dict(collections.Counter(r["serving"] for r in sub)),
        "flipped": sum(r["offline"] != r["serving"] for r in sub),
        "serving_acc_when_seg_used": None, "serving_acc_when_fallback": None,
    }
    for key, want in (("serving_acc_when_seg_used", True), ("serving_acc_when_fallback", False)):
        g = [r for r in sub if r["used_seg"] == want]
        summary[cls][key] = [len(g), round(sum(r["serving"] == cls for r in g) / len(g), 4) if g else None]

overall = {"offline_acc": round(sum(r["offline"] == r["true"] for r in rows) / len(rows), 4),
           "serving_acc": round(sum(r["serving"] == r["true"] for r in rows) / len(rows), 4)}
(ROOT / "eval_run" / "results_pathdiff_after.json").write_text(
    json.dumps({"overall": overall, "per_class": summary, "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps({"overall": overall, "per_class": summary}, indent=2))
