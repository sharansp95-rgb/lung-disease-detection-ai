"""Standalone CT-checkpoint evaluator, reused for the before/after comparison in
the Normal-augmentation retraining experiment.

Loads a checkpoint directly (same construction path as
backend/models.py::load_model_bundle, without the live API), and:

  * --repro DATA_DIR   : reproduce val accuracy against DATA_DIR/val, exactly as
                         backend/models.py::_reproduce_val_accuracy does, and
                         compare it to the checkpoint's claimed val_acc against
                         the same 0.001 tolerance backend/models.py uses.
  * --testdir DATA_DIR : per-class confusion on DATA_DIR/test.
  * --group NAME=DIR=TRUECLASS : per-image predictions over a flat folder of
                         images with one known true class (external sets).

Preprocessing is ml/dataset.py::build_transforms(train=False) throughout -- the
same transform used for training and for the live /predict/ct route. Nothing here
writes to backend/, checkpoints/, or data/ct_scan/.
"""
import argparse, json, sys, pathlib
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = pathlib.Path(r"D:/7th sem/sdp")
sys.path.insert(0, str(ROOT / "ml"))
from model import build_model  # noqa: E402
from dataset import build_transforms, build_datasets  # noqa: E402

REPRO_TOLERANCE = 0.001  # same as backend/models.py


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    class_names = ckpt.get("class_names") or ckpt.get("classes")
    backbone = ckpt.get("backbone") or ckpt.get("architecture")
    model = build_model(backbone, len(class_names), pretrained=False).to(device)
    sd = ckpt["model_state_dict"]
    if not any(k.startswith("backbone.") for k in sd):
        sd = {f"backbone.{k}": v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, class_names, backbone, float(ckpt["val_acc"])


def reproduce_val(model, data_dir, device, class_names):
    _, val_ds, _ = build_datasets(str(data_dir))
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    idx_map = torch.tensor([class_names.index(c) for c in val_ds.classes],
                           dtype=torch.long, device=device)
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == idx_map[labels]).sum().item()
            total += images.size(0)
    return correct / total, total


def eval_testdir(model, data_dir, device, class_names):
    _, _, test_ds = build_datasets(str(data_dir))
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    idx_map = [class_names.index(c) for c in test_ds.classes]
    per = {c: {"n": 0, "correct": 0, "pred_counts": {}} for c in test_ds.classes}
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            for lbl, pr in zip(labels.tolist(), preds.tolist()):
                cls = test_ds.classes[lbl]
                per[cls]["n"] += 1
                per[cls]["correct"] += int(pr == idx_map[lbl])
                pn = class_names[pr]
                per[cls]["pred_counts"][pn] = per[cls]["pred_counts"].get(pn, 0) + 1
    return per


def eval_group(model, folder, true_class, device, class_names):
    tf = build_transforms(train=False)
    rows = []
    for p in sorted(pathlib.Path(folder).rglob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        t = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[0].cpu()
        i = int(probs.argmax())
        rows.append({
            "file": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
            "true": true_class, "pred": class_names[i], "conf": round(float(probs[i]), 4),
            "probs": {c: round(float(v), 4) for c, v in zip(class_names, probs)},
            "correct": class_names[i] == true_class,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repro", default=None)
    ap.add_argument("--testdir", default=None)
    ap.add_argument("--group", action="append", default=[],
                    help="NAME=DIR=TRUECLASS, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, backbone, claimed = load_ckpt(args.ckpt, device)
    print(f"[eval] {args.ckpt}\n  backbone={backbone} classes={class_names} "
          f"claimed_val_acc={claimed:.4f} device={device}")

    out = {"label": args.label, "checkpoint": str(args.ckpt), "backbone": backbone,
           "class_names": class_names, "claimed_val_acc": claimed}

    if args.repro:
        acc, n = reproduce_val(model, args.repro, device, class_names)
        diff = abs(acc - claimed)
        out["reproduction"] = {"val_dir": str(args.repro), "n": n, "reproduced": acc,
                               "claimed": claimed, "diff": diff,
                               "tolerance": REPRO_TOLERANCE, "passed": diff <= REPRO_TOLERANCE}
        print(f"[eval] reproduction: claimed={claimed:.4f} reproduced={acc:.4f} "
              f"(n={n}) diff={diff:.4f} -> {'PASS' if diff <= REPRO_TOLERANCE else 'FAIL'}")

    if args.testdir:
        per = eval_testdir(model, args.testdir, device, class_names)
        out["testdir"] = {"dir": str(args.testdir), "per_class": per}
        print(f"[eval] test set {args.testdir}")
        tot = totc = 0
        for c, d in per.items():
            tot += d["n"]; totc += d["correct"]
            print(f"   {c:<12} {d['correct']:>4}/{d['n']:<4} "
                  f"({100*d['correct']/d['n']:5.1f}%)  preds={d['pred_counts']}")
        print(f"   {'OVERALL':<12} {totc:>4}/{tot:<4} ({100*totc/tot:5.1f}%)")

    out["groups"] = {}
    for spec in args.group:
        name, folder, true_class = spec.split("=", 2)
        rows = eval_group(model, folder, true_class, device, class_names)
        c = sum(r["correct"] for r in rows)
        out["groups"][name] = {"dir": folder, "true_class": true_class,
                               "n": len(rows), "correct": c, "rows": rows}
        print(f"[eval] group {name:<24} {c}/{len(rows)} correct ({true_class})")

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
