"""Standalone experiment (bypasses the live API entirely): does aspect-preserving
resize + percentile-based contrast normalization recover any of the 9 misclassified
external CT Normal images, without breaking the 12/12 correct external Malignant
predictions? Does NOT touch backend/main.py, checkpoints, or the live pipeline --
loads the CT checkpoint directly, read-only.
"""
import sys, json, pathlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = pathlib.Path(r"D:/7th sem/sdp")
sys.path.insert(0, str(ROOT / "ml"))
from model import build_model  # noqa: E402
from dataset import build_transforms, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

CKPT_PATH = ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt"
EXT = ROOT / "eval_run" / "external_samples" / "ct"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
class_names = ckpt.get("class_names") or ckpt.get("classes")
backbone = ckpt.get("backbone") or ckpt.get("architecture")
model = build_model(backbone, len(class_names), pretrained=False).to(device)
state_dict = ckpt["model_state_dict"]
if not any(k.startswith("backbone.") for k in state_dict):
    state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded CT checkpoint, classes={class_names}, backbone={backbone}, claimed val_acc={ckpt['val_acc']:.4f}")

# ---- baseline preprocessing: exactly ml/dataset.py's build_transforms(train=False) ----
baseline_transform = build_transforms(train=False)


# ---- candidate fix: aspect-preserving resize (pad to square) + percentile contrast
# normalization instead of relying on the fixed 8-bit range the source export used ----
def aspect_preserving_pad(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("L")
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def percentile_contrast_normalize(im: Image.Image, lo_pct=1, hi_pct=99) -> Image.Image:
    arr = np.array(im).astype(np.float32)
    lo, hi = np.percentile(arr, lo_pct), np.percentile(arr, hi_pct)
    if hi - lo < 1e-6:
        return im
    arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0
    return Image.fromarray(arr.astype(np.uint8))


_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.repeat(3, 1, 1)),  # 1ch -> 3ch, mirrors Grayscale(3ch)
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def fixed_transform(im: Image.Image) -> torch.Tensor:
    im = percentile_contrast_normalize(im.convert("L"))
    im = aspect_preserving_pad(im, IMAGE_SIZE)
    return _to_tensor_norm(im)


def predict(tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx]), {c: float(p) for c, p in zip(class_names, probs)}


def run_group(folder: pathlib.Path, true_class: str):
    rows = []
    for p in sorted(folder.glob("*")):
        im = Image.open(p)
        base_pred, base_conf, base_probs = predict(baseline_transform(im.convert("RGB")))
        fix_pred, fix_conf, fix_probs = predict(fixed_transform(im))
        rows.append({
            "file": p.name, "true": true_class,
            "baseline_pred": base_pred, "baseline_conf": round(base_conf, 4),
            "baseline_probs": {k: round(v, 4) for k, v in base_probs.items()},
            "fixed_pred": fix_pred, "fixed_conf": round(fix_conf, 4),
            "fixed_probs": {k: round(v, 4) for k, v in fix_probs.items()},
            "baseline_correct": base_pred == true_class,
            "fixed_correct": fix_pred == true_class,
        })
    return rows


results = {
    "normal": run_group(EXT / "normal", "Normal"),
    "malignant": run_group(EXT / "malignant", "Malignant"),
}

for group, rows in results.items():
    print(f"\n=== {group} (n={len(rows)}) ===")
    for r in rows:
        flip = ""
        if r["baseline_correct"] != r["fixed_correct"]:
            flip = " <-- FLIPPED " + ("Normal->correct" if r["fixed_correct"] else "correct->wrong")
        print(f"  {r['file']:35s} baseline={r['baseline_pred']:10s}({r['baseline_conf']:.2f}) "
              f"fixed={r['fixed_pred']:10s}({r['fixed_conf']:.2f}){flip}")

    n = len(rows)
    base_correct = sum(r["baseline_correct"] for r in rows)
    fix_correct = sum(r["fixed_correct"] for r in rows)
    print(f"  baseline correct: {base_correct}/{n}   fixed correct: {fix_correct}/{n}")

out = ROOT / "eval_run" / "results_ct_preprocessing_fix.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved {out}")
