"""
xray_source_correlation_audit.py
=================================
The merged X-ray set draws COVID-19 only from covid19db and Tuberculosis
only from tbdb, so source database and diagnosis are perfectly confounded
for those two classes. Source is recoverable from the filename prefix
(covid19db_* / tbdb_*). This checks whether xray_best_model_phase35_v2.pt
is exploiting that confound instead of learning pathology -- the X-ray
equivalent of the shortcut-learning check already done on the CT set.

The control lives entirely inside the Normal class, since Normal is the
only class drawing from BOTH databases. Any source-separability found
there is pure source signal with zero diagnostic content (COVID vs TB
is never tested directly -- that comparison proves nothing given the
confound).

Four tests, read-only on data/ and checkpoints/:

  Q1 -- Does a source signal exist at all?
        32x32 greyscale pixel logistic-regression probe, Normal only,
        balanced across source, held-out 25% split. Accuracy + AUC vs
        0.5 chance.

  Q2 -- Do the model's errors track source?
        Trained 4-class model on TEST Normal images, split by source
        prefix. Recall + full wrong-prediction distribution per source,
        plus each source's share of errors landing on its own exclusive
        class (covid19db -> COVID-19, tbdb -> Tuberculosis).

  Q3 -- Occlusion sensitivity.
        Full / border-only (central 56% blanked) / centre-only
        (periphery blanked) accuracy per class, blanking done in
        normalized tensor space after the eval transform. A pathology
        model should fall toward 25% on border-only.

  Q4 -- Grad-CAM attention location.
        Fraction of CAM mass falling outside the central box, on
        correctly-classified test images (cap 120/class), vs the
        geometric baseline (peripheral band's share of image area).

Usage:  python scripts/xray_source_correlation_audit.py
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from dataset import build_transforms, IMAGE_SIZE  # noqa: E402
from model import build_model  # noqa: E402
from gradcam import GradCAM, get_target_layer  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "xray_best_model_phase35_v2.pt"
DATA_DIR = ROOT / "data" / "xray"
OUT_DIR = ROOT / "eval_outputs" / "xray_source_audit"
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".png", ".jpg", ".jpeg"}

SEED = 42
MARGIN_FRAC = 0.22
GRADCAM_CAP_PER_CLASS = 120


# ── shared helpers ───────────────────────────────────────────────────────────

def source_tag(filename: str) -> str:
    return Path(filename).stem.split("_")[0]


def list_images(d: Path):
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]


def central_box_bounds(size: int, margin_frac: float = MARGIN_FRAC):
    lo = int(round(size * margin_frac))
    hi = size - lo
    return lo, hi


def load_model(device):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]
    model = build_model(backbone, len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, ckpt.get("val_acc")


class FileListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── Q1: pixel-level source signal ────────────────────────────────────────────

def q1_source_signal(rng):
    print("\n=== Q1: does a source signal exist? (Normal class, all splits, 32x32 pixel probe) ===")
    by_src = defaultdict(list)
    for split in SPLITS:
        d = DATA_DIR / split / "Normal"
        if not d.is_dir():
            continue
        for p in list_images(d):
            by_src[source_tag(p.name)].append(p)

    sources = sorted(by_src)
    print(f"  raw counts: " + ", ".join(f"{s}={len(by_src[s])}" for s in sources))
    n = min(len(by_src[s]) for s in sources)
    balanced = []
    for s in sources:
        paths = by_src[s][:]
        rng.shuffle(paths)
        balanced.extend((p, s) for p in paths[:n])
    rng.shuffle(balanced)
    print(f"  balanced to n={n} per source, total={len(balanced)}")

    X, y = [], []
    for p, s in balanced:
        img = Image.open(p).convert("L").resize((32, 32), Image.BILINEAR)
        X.append((np.asarray(img, dtype=np.float32) / 255.0).flatten())
        y.append(s)
    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=SEED
    )
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    probs_pos = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_pos)

    result = {
        "classes_by_label": {str(i): c for i, c in enumerate(le.classes_)},
        "n_per_source_balanced": n,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_accuracy": round(float(acc), 4),
        "test_auc": round(float(auc), 4),
        "chance_accuracy": 0.5,
        "chance_auc": 0.5,
        "verdict": (
            "SOURCE SIGNAL PRESENT: even a weak 32x32 pixel probe separates source well above chance"
            if acc > 0.6 or auc > 0.65
            else "no strong source signal detected at the pixel level"
        ),
    }
    print(f"  test accuracy: {acc:.4f} (chance 0.5)   test AUC: {auc:.4f} (chance 0.5)")
    print(f"  {result['verdict']}")
    return result


# ── Q2: model error tracking by source ───────────────────────────────────────

def q2_error_tracking(model, device, transform, class_names):
    print("\n=== Q2: do the model's errors track source? (TEST Normal images) ===")
    d = DATA_DIR / "test" / "Normal"
    by_src = defaultdict(list)
    for p in list_images(d):
        by_src[source_tag(p.name)].append(p)

    normal_idx = class_names.index("Normal")
    exclusive_class = {"covid19db": "COVID-19", "tbdb": "Tuberculosis"}

    result = {}
    for src, paths in sorted(by_src.items()):
        ds = FileListDataset([(p, normal_idx) for p in paths], transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds.extend(out.argmax(dim=1).cpu().tolist())

        n = len(preds)
        correct = sum(1 for pr in preds if pr == normal_idx)
        recall = correct / n
        wrong = [class_names[pr] for pr in preds if pr != normal_idx]
        wrong_counts = Counter(wrong)
        n_wrong = len(wrong)
        exclusive = exclusive_class.get(src)
        exclusive_share = (
            round(wrong_counts.get(exclusive, 0) / n_wrong, 4)
            if n_wrong > 0 and exclusive else None
        )

        result[src] = {
            "n": n,
            "recall_on_normal": round(recall, 4),
            "n_errors": n_wrong,
            "error_counts": dict(wrong_counts),
            "error_distribution": (
                {k: round(v / n_wrong, 4) for k, v in wrong_counts.items()} if n_wrong else {}
            ),
            "exclusive_class": exclusive,
            "exclusive_class_share_of_errors": exclusive_share,
        }
        print(f"  {src:12s} n={n:4d}  recall={recall:.4f}  errors={n_wrong}  "
              f"dist={dict(wrong_counts)}  own-exclusive-class share={exclusive_share}")

    return result


# ── Q3: occlusion sensitivity ────────────────────────────────────────────────

def occlude_batch(batch, mode, lo, hi):
    if mode == "full":
        return batch
    out = batch.clone()
    if mode == "border_only":
        out[:, :, lo:hi, lo:hi] = 0.0
    elif mode == "centre_only":
        mask = torch.zeros_like(out)
        mask[:, :, lo:hi, lo:hi] = 1.0
        out = out * mask
    else:
        raise ValueError(mode)
    return out


def q3_occlusion(model, device, transform, class_names):
    print("\n=== Q3: occlusion sensitivity (per test class) ===")
    lo, hi = central_box_bounds(IMAGE_SIZE)
    print(f"  central box: [{lo}:{hi}] of {IMAGE_SIZE} "
          f"(box side frac={round((hi-lo)/IMAGE_SIZE, 4)}, margin={MARGIN_FRAC})")

    modes = ("full", "border_only", "centre_only")
    result = {}
    pooled_pred_counts = {mode: Counter() for mode in modes}
    pooled_total = 0

    for cls in class_names:
        d = DATA_DIR / "test" / cls
        paths = list_images(d)
        cls_idx = class_names.index(cls)
        ds = FileListDataset([(p, cls_idx) for p in paths], transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False)

        accs = {}
        pred_dists = {}
        for mode in modes:
            correct, total = 0, 0
            pred_counts = Counter()
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = occlude_batch(imgs, mode, lo, hi).to(device)
                    labels = labels.to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    for pr in preds.cpu().tolist():
                        pred_counts[class_names[pr]] += 1
            accs[mode] = round(correct / total, 4)
            pred_dists[mode] = {
                c: round(pred_counts.get(c, 0) / total, 4) for c in class_names
            }
            pooled_pred_counts[mode].update(pred_counts)

        pooled_total += len(paths)
        result[cls] = {
            "n": len(paths), **accs, "chance_4class": 0.25,
            "pred_distribution": pred_dists,
        }
        print(f"  {cls:15s} n={len(paths):4d}  full={accs['full']:.4f}  "
              f"border_only={accs['border_only']:.4f}  centre_only={accs['centre_only']:.4f}  "
              f"(chance=0.25)")
        for mode in modes:
            print(f"      {mode:12s} pred_dist={pred_dists[mode]}")

    pooled_dist = {
        mode: {
            c: round(pooled_pred_counts[mode].get(c, 0) / pooled_total, 4)
            for c in class_names
        }
        for mode in modes
    }
    result["pooled_pred_distribution"] = pooled_dist
    result["pooled_n"] = pooled_total

    print(f"\n  pooled across all {pooled_total} test images, per condition:")
    for mode in modes:
        print(f"      {mode:12s} pred_dist={pooled_dist[mode]}")

    return result


# ── Q4: Grad-CAM attention location ──────────────────────────────────────────

def q4_gradcam_attention(model, device, transform, class_names, rng):
    print("\n=== Q4: where does attention land? (Grad-CAM, correctly-classified test images) ===")
    lo, hi = central_box_bounds(IMAGE_SIZE)
    box_side_frac = (hi - lo) / IMAGE_SIZE
    geometric_baseline = round(1.0 - box_side_frac ** 2, 4)
    print(f"  central box: [{lo}:{hi}] of {IMAGE_SIZE}  "
          f"geometric peripheral-area baseline={geometric_baseline}")

    target_layer = get_target_layer(model)

    result = {"geometric_baseline_peripheral_area_frac": geometric_baseline}
    saved_figs = []

    for cls in class_names:
        cls_idx = class_names.index(cls)
        d = DATA_DIR / "test" / cls
        paths = list_images(d)

        # first pass: find correctly-classified images (batched, no grad,
        # GradCAM hook not yet attached so retain_grad() is never hit here)
        ds = FileListDataset([(p, cls_idx) for p in paths], transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        correct_flags = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu()
                correct_flags.extend((preds == labels).tolist())

        correct_paths = [p for p, ok in zip(paths, correct_flags) if ok]
        rng.shuffle(correct_paths)
        selected = correct_paths[:GRADCAM_CAP_PER_CLASS]

        cam_engine = GradCAM(model, target_layer)
        fracs_outside = []
        fracs_by_src = defaultdict(list)
        cam_sum = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)

        for p in selected:
            img = Image.open(p).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            cam, _, _ = cam_engine.generate(tensor, class_idx=cls_idx)
            cam_resized = np.asarray(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
                ),
                dtype=np.float64,
            ) / 255.0

            total_mass = cam_resized.sum()
            if total_mass <= 0:
                continue
            inside_mass = cam_resized[lo:hi, lo:hi].sum()
            frac_outside = 1.0 - (inside_mass / total_mass)
            fracs_outside.append(frac_outside)
            fracs_by_src[source_tag(p.name)].append(frac_outside)
            cam_sum += cam_resized

        n_used = len(fracs_outside)
        by_src_summary = {
            src: {
                "n": len(v),
                "mean_frac_outside": round(float(np.mean(v)), 4),
                "std_frac_outside": round(float(np.std(v)), 4),
            }
            for src, v in fracs_by_src.items()
        }

        result[cls] = {
            "n_correct_available": len(correct_paths),
            "n_used": n_used,
            "mean_frac_outside": round(float(np.mean(fracs_outside)), 4) if n_used else None,
            "std_frac_outside": round(float(np.std(fracs_outside)), 4) if n_used else None,
            "by_source": by_src_summary,
        }
        print(f"  {cls:15s} n_used={n_used:3d}  mean_frac_outside={result[cls]['mean_frac_outside']}  "
              f"std={result[cls]['std_frac_outside']}  by_source={by_src_summary}  "
              f"(geometric baseline={geometric_baseline})")

        if n_used:
            mean_cam = cam_sum / n_used
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(mean_cam, cmap="jet", vmin=0, vmax=mean_cam.max())
            rect = plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False,
                                  edgecolor="white", linewidth=1.5, linestyle="--")
            ax.add_patch(rect)
            ax.set_title(f"Mean Grad-CAM: {cls}\n(n={n_used}, frac_outside={result[cls]['mean_frac_outside']})")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig_path = OUT_DIR / f"gradcam_mean_{cls.replace('/', '-')}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_figs.append(str(fig_path.relative_to(ROOT)))

        cam_engine.remove_hooks()

    result["saved_figures"] = saved_figs
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=["q1", "q2", "q3", "q4"], default=None,
                         help="Run only these tests (default: all four). Other sections of "
                              "source_correlation_report.json are left untouched.")
    args = parser.parse_args()
    which = set(args.only) if args.only else {"q1", "q2", "q3", "q4"}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[xray_source_correlation_audit] device={device}  running={sorted(which)}")
    print(f"[xray_source_correlation_audit] loading checkpoint: {CHECKPOINT}")
    model, class_names, val_acc = load_model(device)
    print(f"[xray_source_correlation_audit] classes={class_names}  checkpoint val_acc={val_acc}")
    transform = build_transforms(train=False)

    report_path = OUT_DIR / "source_correlation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "class_names": class_names,
            "checkpoint_val_acc": val_acc,
            "margin_frac": MARGIN_FRAC,
            "gradcam_cap_per_class": GRADCAM_CAP_PER_CLASS,
        }

    if "q1" in which:
        report["q1_source_signal"] = q1_source_signal(rng)
    if "q2" in which:
        report["q2_error_tracking"] = q2_error_tracking(model, device, transform, class_names)
    if "q3" in which:
        report["q3_occlusion"] = q3_occlusion(model, device, transform, class_names)
    if "q4" in which:
        report["q4_gradcam_attention"] = q4_gradcam_attention(model, device, transform, class_names, rng)

    name_map = {
        "q1": ("q1_source_signal.json", "q1_source_signal"),
        "q2": ("q2_error_tracking.json", "q2_error_tracking"),
        "q3": ("q3_occlusion.json", "q3_occlusion"),
        "q4": ("q4_gradcam_attention.json", "q4_gradcam_attention"),
    }
    for key in which:
        fname, report_key = name_map[key]
        with open(OUT_DIR / fname, "w") as f:
            json.dump(report[report_key], f, indent=2)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[xray_source_correlation_audit] JSON + figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
