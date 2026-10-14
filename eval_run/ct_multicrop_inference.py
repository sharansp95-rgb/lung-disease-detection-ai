"""Bounded, no-retraining experiment: does a sliding-window multi-crop ensemble over the
EXISTING live checkpoint (ct_best_model_phase35_v2.pt, untouched) improve external Normal
detection without regressing external Malignant / in-dataset accuracy? Deployable in principle
via the existing /predict/ct shape (aggregation happens server-side, one image in/one prediction
out) -- unlike the LIDC nodule-crop approach, which needs a separate detector.

Does NOT touch backend/main.py, the checkpoint, or data/ct_scan/. Standalone script only.
"""
import sys, json, pathlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = pathlib.Path(r"D:/7th sem/sdp")
sys.path.insert(0, str(ROOT / "ml"))
from model import build_model  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE  # noqa: E402

CKPT_PATH = ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
class_names = ckpt.get("class_names") or ckpt.get("classes")
backbone = ckpt.get("backbone") or ckpt.get("architecture")
model = build_model(backbone, len(class_names), pretrained=False).to(device)
sd = ckpt["model_state_dict"]
if not any(k.startswith("backbone.") for k in sd):
    sd = {f"backbone.{k}": v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()
print(f"Loaded {CKPT_PATH.name}, classes={class_names}, backbone={backbone}")

_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def gen_crop_boxes(w, h, frac=0.5, n=3):
    """3x3 (n=3) grid of overlapping crops, each `frac` of width/height, stepping so the
    grid exactly covers the image edge-to-edge with 50% overlap between neighbours."""
    cw, ch = int(w * frac), int(h * frac)
    xs = [int(i * (w - cw) / (n - 1)) for i in range(n)] if n > 1 else [0]
    ys = [int(i * (h - ch) / (n - 1)) for i in range(n)] if n > 1 else [0]
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, x + cw, y + ch))
    return boxes


@torch.no_grad()
def predict_probs(pil_img: Image.Image) -> np.ndarray:
    t = _tf(pil_img).unsqueeze(0).to(device)
    logits = model(t)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def run_image(path: pathlib.Path):
    im = Image.open(path).convert("RGB")
    w, h = im.size

    whole_probs = predict_probs(im)

    boxes = gen_crop_boxes(w, h, frac=0.5, n=3)
    crop_probs = [predict_probs(im.crop(b)) for b in boxes]
    crop_probs = np.stack(crop_probs)  # [9, 3]
    crop_preds = crop_probs.argmax(axis=1)

    # (a) max-malignant / max-per-class pooling: aggregate_score[c] = max over crops of P(c)
    max_pool = crop_probs.max(axis=0)
    pred_maxpool = class_names[int(max_pool.argmax())]

    # (b) majority vote over per-crop top-1, ties broken by mean crop probability
    counts = np.bincount(crop_preds, minlength=len(class_names))
    top = np.flatnonzero(counts == counts.max())
    if len(top) == 1:
        pred_vote = class_names[int(top[0])]
    else:
        mean_probs_tiebreak = crop_probs.mean(axis=0)
        pred_vote = class_names[int(top[np.argmax(mean_probs_tiebreak[top])])]

    # (c) average probability across crops
    avg_probs = crop_probs.mean(axis=0)
    pred_avg = class_names[int(avg_probs.argmax())]

    return {
        "file": path.name,
        "whole": {"pred": class_names[int(whole_probs.argmax())], "probs": whole_probs.round(4).tolist()},
        "maxpool": {"pred": pred_maxpool, "probs": max_pool.round(4).tolist()},
        "vote": {"pred": pred_vote, "crop_pred_counts": {class_names[i]: int(counts[i]) for i in range(len(class_names))}},
        "avg": {"pred": pred_avg, "probs": avg_probs.round(4).tolist()},
        "per_crop_preds": [class_names[i] for i in crop_preds],
    }


def run_group(folder: pathlib.Path, true_class: str):
    rows = []
    for p in sorted(folder.glob("*")):
        r = run_image(p)
        r["true"] = true_class
        rows.append(r)
    return rows


GROUPS = [
    (ROOT / "eval_run/external_samples/ct/malignant", "Malignant", "external"),
    (ROOT / "eval_run/external_samples/ct/normal", "Normal", "external"),
    (ROOT / "data/ct_scan/test/Benign", "Benign", "indataset"),
    (ROOT / "data/ct_scan/test/Malignant", "Malignant", "indataset"),
    (ROOT / "data/ct_scan/test/Normal", "Normal", "indataset"),
]

all_rows = []
for folder, true_class, source in GROUPS:
    rows = run_group(folder, true_class)
    for r in rows:
        r["source"] = source
    all_rows.extend(rows)
    print(f"{source}/{true_class}: {len(rows)} images done")

(ROOT / "eval_run/results_ct_multicrop.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

# ---- summary table ----
strategies = ["whole", "maxpool", "vote", "avg"]
print("\n=== Summary: correct/total per (source, class, strategy) ===")
summary = {}
for source in ["external", "indataset"]:
    for cls in ["Malignant", "Normal", "Benign"]:
        sub = [r for r in all_rows if r["source"] == source and r["true"] == cls]
        if not sub:
            continue
        line = f"{source:10s} {cls:10s} n={len(sub):3d}  "
        for strat in strategies:
            correct = sum(1 for r in sub if r[strat]["pred"] == cls)
            line += f"{strat}={correct}/{len(sub)}  "
            summary[f"{source}/{cls}/{strat}"] = f"{correct}/{len(sub)}"
        print(line)

(ROOT / "eval_run/results_ct_multicrop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nSaved results_ct_multicrop.json and results_ct_multicrop_summary.json")
