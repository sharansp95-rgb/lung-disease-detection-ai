"""
xray_source_audit.py
====================
Tests whether the X-ray classifier (xray_best_model_phase35_v2.pt)
separates classes on SOURCE DATABASE (covid19db vs tbdb) rather than
on pathology.

Measurements
------------
1. Per-source accuracy within every class.
   - COVID-19   : 100 % covid19db  (no tbdb images)
   - Pneumonia  : 100 % covid19db  (no tbdb images)
   - Tuberculosis: 100 % tbdb      (no covid19db images)
   - Normal     : MIXED -- both sources.  This is the key signal:
       if the model is relying on source characteristics, Normal/covid19db
       and Normal/tbdb will show different confidence distributions even
       though the label is identical.

2. Source-separability probe on the Normal class only.
   We run the model on Normal images from both sources and ask:
   can we predict the SOURCE from the model's softmax output alone?
   A linear probe (logistic regression on the 4-dim softmax vector)
   that achieves >> 50 % accuracy means the activations carry
   source-database signal.

3. Confidence distributions per (class, source) pair -- printed as
   mean ± std and saved to JSON.

Output
------
eval_outputs/xray_source_audit/source_audit_report.json

Usage
-----
    python scripts/xray_source_audit.py [--max-per-group 400]

No training, no model modification, read-only except for the JSON output.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from dataset import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import build_model

CHECKPOINT = ROOT / "checkpoints" / "xray_best_model_phase35_v2.pt"
DATA_DIR   = ROOT / "data" / "xray"
OUT_DIR    = ROOT / "eval_outputs" / "xray_source_audit"

SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def source_tag(filename: str) -> str:
    """Extract source tag from filename prefix (e.g. 'covid19db_COVID-1.png' → 'covid19db')."""
    stem = Path(filename).stem          # drop .png
    return stem.split("_")[0]          # first underscore-delimited token


def load_model(device):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    backbone    = ckpt["backbone"]
    model = build_model(backbone, len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, ckpt.get("val_acc")


class FileListDataset(Dataset):
    """Dataset built from an explicit list of (path, label_int) tuples."""
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def collect_samples(split: str, max_per_group: int, rng: random.Random):
    """
    Walk data/xray/<split>/<class>/ and collect (path, class_idx, source_tag).
    Returns list of (path, class_idx, source_tag) dicts, capped at
    max_per_group per (class, source) bucket.
    """
    split_dir = DATA_DIR / split
    classes   = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    records = []
    for cls in classes:
        cls_idx  = class_to_idx[cls]
        by_src: dict[str, list[Path]] = {}
        for p in sorted((split_dir / cls).iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            tag = source_tag(p.name)
            by_src.setdefault(tag, []).append(p)

        for tag, paths in by_src.items():
            rng.shuffle(paths)
            chosen = paths[:max_per_group]
            for p in chosen:
                records.append({"path": p, "class_name": cls, "class_idx": cls_idx, "source": tag})

    return records, classes


@torch.no_grad()
def run_inference(records, transform, model, device, batch_size=32):
    """
    Returns list of dicts: each input record enriched with
    'pred_idx', 'pred_class', 'probs' (numpy array), 'conf'.
    """
    dataset = FileListDataset([(r["path"], r["class_idx"]) for r in records], transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_preds = []
    for images, _ in loader:
        images = images.to(device)
        outputs = model(images)
        probs   = F.softmax(outputs, dim=1).cpu().numpy()
        preds   = probs.argmax(axis=1)
        all_probs.extend(probs)
        all_preds.extend(preds.tolist())

    enriched = []
    for i, rec in enumerate(records):
        r = dict(rec)
        r["probs"]    = all_probs[i]
        r["pred_idx"] = all_preds[i]
        r["conf"]     = float(all_probs[i][all_preds[i]])
        enriched.append(r)
    return enriched


def per_source_accuracy(enriched, class_names):
    """
    Returns nested dict:
      { class_name: { source_tag: {"n": int, "correct": int, "acc": float,
                                    "mean_conf": float, "std_conf": float} } }
    """
    from collections import defaultdict
    buckets: dict = defaultdict(lambda: defaultdict(list))
    for r in enriched:
        buckets[r["class_name"]][r["source"]].append(r)

    result = {}
    for cls in class_names:
        result[cls] = {}
        for src, recs in buckets[cls].items():
            correct  = sum(1 for r in recs if r["pred_idx"] == r["class_idx"])
            confs    = [r["conf"] for r in recs]
            result[cls][src] = {
                "n":         len(recs),
                "correct":   correct,
                "acc":       round(correct / len(recs), 4),
                "mean_conf": round(float(np.mean(confs)), 4),
                "std_conf":  round(float(np.std(confs)),  4),
            }
    return result


def normal_source_probe(enriched, class_names):
    """
    Logistic-regression probe: can we predict source (covid19db vs tbdb)
    from the 4-dim softmax vector, restricted to Normal-class images only?

    Returns dict with probe accuracy, baseline (majority-class), and
    mean softmax per source.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    normal_recs = [r for r in enriched if r["class_name"] == "Normal"]
    sources     = sorted({r["source"] for r in normal_recs})

    if len(sources) < 2:
        return {"note": "Normal class has only one source — probe not applicable",
                "sources_found": sources}

    X = np.array([r["probs"] for r in normal_recs])   # (N, 4)
    y = np.array([r["source"] for r in normal_recs])   # string labels

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 5-fold stratified CV
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs   = []
    for train_idx, test_idx in skf.split(X, y_enc):
        clf = LogisticRegression(max_iter=500, random_state=SEED)
        clf.fit(X[train_idx], y_enc[train_idx])
        accs.append(clf.score(X[test_idx], y_enc[test_idx]))

    majority_baseline = float(np.bincount(y_enc).max()) / len(y_enc)

    # Mean softmax per source
    mean_softmax = {}
    for src in sources:
        mask = y == src
        mean_softmax[src] = {
            class_names[i]: round(float(X[mask, i].mean()), 4)
            for i in range(len(class_names))
        }

    return {
        "n_normal_images":      len(normal_recs),
        "sources":              sources,
        "probe_cv_accs":        [round(a, 4) for a in accs],
        "probe_mean_acc":       round(float(np.mean(accs)), 4),
        "probe_std_acc":        round(float(np.std(accs)),  4),
        "majority_baseline":    round(majority_baseline, 4),
        "mean_softmax_by_src":  mean_softmax,
        "interpretation": (
            "probe_mean_acc >> majority_baseline  →  model output carries source-database signal"
            if float(np.mean(accs)) > majority_baseline + 0.05
            else "probe_mean_acc ≈ majority_baseline  →  model does NOT separate sources via its softmax"
        ),
    }


def pred_distribution(enriched, class_names):
    """
    For each (true_class, source) group: distribution of predicted classes.
    Useful for seeing if, e.g., tbdb Normal images get classified differently
    from covid19db Normal images.
    """
    from collections import defaultdict
    buckets: dict = defaultdict(lambda: defaultdict(int))
    totals:  dict = defaultdict(int)

    for r in enriched:
        key = (r["class_name"], r["source"])
        buckets[key][class_names[r["pred_idx"]]] += 1
        totals[key] += 1

    result = {}
    for (cls, src), counts in sorted(buckets.items()):
        n = totals[(cls, src)]
        result[f"{cls}|{src}"] = {
            pred: round(cnt / n, 4)
            for pred, cnt in sorted(counts.items(), key=lambda x: -x[1])
        }
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="X-ray source-database confound audit")
    parser.add_argument("--max-per-group", type=int, default=400,
                        help="Max images per (class, source) bucket (default: 400)")
    parser.add_argument("--split", default="test",
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng    = random.Random(SEED)
    device = torch.device("cpu")

    print(f"[source_audit] Loading checkpoint: {CHECKPOINT}")
    model, class_names, val_acc = load_model(device)
    print(f"[source_audit] Classes: {class_names}  val_acc={val_acc}")

    print(f"[source_audit] Collecting samples from data/xray/{args.split}/ "
          f"(max {args.max_per_group} per group) …")
    records, _ = collect_samples(args.split, args.max_per_group, rng)

    # Summary of what we have
    from collections import Counter
    bucket_counts = Counter((r["class_name"], r["source"]) for r in records)
    print("[source_audit] Sample counts per (class, source):")
    for (cls, src), n in sorted(bucket_counts.items()):
        print(f"  {cls:20s}  {src:15s}  n={n}")

    transform = build_transforms(train=False)
    print(f"[source_audit] Running inference on {len(records)} images …")
    enriched = run_inference(records, transform, model, device, batch_size=args.batch_size)

    print("[source_audit] Computing per-source accuracy …")
    acc_table = per_source_accuracy(enriched, class_names)

    print("\n=== Per-source accuracy ===")
    for cls in class_names:
        for src, stats in acc_table[cls].items():
            print(f"  {cls:20s}  {src:15s}  acc={stats['acc']:.4f}  "
                  f"conf={stats['mean_conf']:.4f}±{stats['std_conf']:.4f}  n={stats['n']}")

    print("\n[source_audit] Running Normal-class source probe …")
    probe = normal_source_probe(enriched, class_names)
    print(f"  probe CV accs:  {probe.get('probe_cv_accs')}")
    print(f"  probe mean acc: {probe.get('probe_mean_acc')}  "
          f"(majority baseline: {probe.get('majority_baseline')})")
    print(f"  {probe.get('interpretation', '')}")

    print("\n=== Predicted-class distribution per (true_class, source) ===")
    dist = pred_distribution(enriched, class_names)
    for key, d in dist.items():
        print(f"  {key:40s}  {d}")

    report = {
        "checkpoint":              str(CHECKPOINT),
        "split":                   args.split,
        "max_per_group":           args.max_per_group,
        "class_names":             class_names,
        "val_acc_checkpoint":      val_acc,
        "per_source_accuracy":     acc_table,
        "normal_source_probe":     probe,
        "pred_dist_by_class_src":  dist,
    }

    out_path = OUT_DIR / "source_audit_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[source_audit] Report saved to {out_path}")


if __name__ == "__main__":
    main()
