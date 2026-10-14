"""Diagnosis-only: load original (phase35_v2, live) and retrained (phase36_normal_fix,
experimental) CT checkpoints, run Grad-CAM on the same 12 external Malignant images
through both, save overlays, record predicted class/confidence for both.
Does NOT touch backend/main.py, live checkpoint, or data/ct_scan/ -- read-only."""
import sys, json, pathlib
import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(r"D:/7th sem/sdp")
sys.path.insert(0, str(ROOT / "ml"))
from model import build_model  # noqa: E402
from dataset import build_transforms, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from gradcam import GradCAM, get_target_layer, denormalize, overlay_cam  # noqa: E402

CKPTS = {
    "original_phase35_v2": ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt",
    "retrained_phase36_normal_fix": ROOT / "checkpoints" / "ct_best_model_phase36_normal_fix.pt",
}
EXT_MALIGNANT = ROOT / "eval_run" / "external_samples" / "ct" / "malignant"
OUT = ROOT / "eval_run" / "ct_shortcut_gradcam_pngs"
OUT.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = build_transforms(train=False)


def load(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    class_names = ckpt.get("class_names") or ckpt.get("classes")
    backbone = ckpt.get("backbone") or ckpt.get("architecture")
    model = build_model(backbone, len(class_names), pretrained=False).to(device)
    sd = ckpt["model_state_dict"]
    if not any(k.startswith("backbone.") for k in sd):
        sd = {f"backbone.{k}": v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, class_names


models = {tag: load(p) for tag, p in CKPTS.items()}
for tag, (m, cn) in models.items():
    print(tag, "classes:", cn)

results = []
for img_path in sorted(EXT_MALIGNANT.glob("*")):
    im = Image.open(img_path).convert("RGB")
    tensor = transform(im).unsqueeze(0).to(device)
    row = {"file": img_path.name}
    for tag, (model, class_names) in models.items():
        cam_engine = GradCAM(model, get_target_layer(model))
        try:
            cam, pred_idx, probs = cam_engine.generate(tensor.clone(), class_idx=None)
        finally:
            cam_engine.remove_hooks()
        pred_class = class_names[pred_idx]
        conf = float(probs[pred_idx])
        probs_dict = {c: round(float(p), 4) for c, p in zip(class_names, probs)}

        display_img = denormalize(tensor[0].cpu(), IMAGENET_MEAN, IMAGENET_STD)
        overlay = overlay_cam(display_img, cam, alpha=0.45)
        fname = f"{img_path.stem}__{tag}.png"
        Image.fromarray(overlay).save(OUT / fname)

        row[tag] = {"pred": pred_class, "conf": round(conf, 4), "probs": probs_dict, "heatmap": fname}
        print(f"{img_path.name:35s} [{tag:28s}] -> {pred_class:10s} {conf:.3f}  probs={probs_dict}")
    results.append(row)

(ROOT / "eval_run" / "ct_shortcut_gradcam_results.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("saved ct_shortcut_gradcam_results.json")
