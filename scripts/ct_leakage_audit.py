"""
Post-hoc leakage audit for data/ct_scan/ (train+val+test combined, ~1097
images). Cross-validation can rule out "lucky split" but not patient
leakage (near-duplicate slices of the same patient landing in different
splits) -- CV folds drawn from the same leaky pool would all be leaky
alike. This checks for that directly:

  1. Perceptual hash (phash, 64-bit) every image -- fast first pass.
  2. For pairs with a small Hamming distance, verify with SSIM (structural
     similarity on resized grayscale) to confirm they're actually
     near-duplicate, not just coincidentally hash-close.
  3. Cluster confirmed near-duplicates (union-find) and check whether each
     cluster's members landed in one split or were spread across
     train/val/test.

Read-only: does not touch data/ct_scan/, does not retrain anything.
"""

import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parent.parent
CT_DIR = ROOT / "data" / "ct_scan"
SPLITS = ["train", "val", "test"]
CLASSES = ["Benign", "Malignant", "Normal"]

HASH_SIZE = 8  # -> 64-bit phash
CANDIDATE_HAMMING_MAX = 12  # generous first-pass cutoff, verified below via SSIM
SSIM_CONFIRM_THRESHOLD = 0.90
SSIM_RESIZE = (256, 256)


def collect_images():
    records = []  # (path, split, cls)
    for split in SPLITS:
        for cls in CLASSES:
            d = CT_DIR / split / cls
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    records.append((p, split, cls))
    return records


def compute_hashes(records):
    hashes = []
    for p, split, cls in records:
        h = imagehash.phash(Image.open(p).convert("L"), hash_size=HASH_SIZE)
        hashes.append(h.hash.flatten())
    return np.array(hashes)


def compute_ssim(path_a: Path, path_b: Path) -> float:
    a = np.array(Image.open(path_a).convert("L").resize(SSIM_RESIZE))
    b = np.array(Image.open(path_b).convert("L").resize(SSIM_RESIZE))
    return float(ssim(a, b))


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    records = collect_images()
    n = len(records)
    print(f"[ct_leakage_audit] total CT images: {n}")

    hash_bits = compute_hashes(records)
    print("[ct_leakage_audit] computed perceptual hashes")

    dist_matrix = squareform(pdist(hash_bits, metric="hamming")) * hash_bits.shape[1]
    print("[ct_leakage_audit] computed pairwise Hamming distances")

    # candidate pairs below the generous first-pass cutoff
    iu = np.triu_indices(n, k=1)
    candidate_mask = dist_matrix[iu] <= CANDIDATE_HAMMING_MAX
    cand_i = iu[0][candidate_mask]
    cand_j = iu[1][candidate_mask]
    print(f"[ct_leakage_audit] candidate pairs (Hamming <= {CANDIDATE_HAMMING_MAX}): {len(cand_i)}")

    # show the Hamming-distance distribution among candidates for calibration
    if len(cand_i):
        dists = dist_matrix[cand_i, cand_j]
        print(f"[ct_leakage_audit] candidate Hamming distances: min={dists.min():.0f} "
              f"max={dists.max():.0f} median={np.median(dists):.1f}")

    confirmed_pairs = []
    for idx, (i, j) in enumerate(zip(cand_i, cand_j)):
        path_i, split_i, cls_i = records[i]
        path_j, split_j, cls_j = records[j]
        s = compute_ssim(path_i, path_j)
        if s >= SSIM_CONFIRM_THRESHOLD:
            confirmed_pairs.append((int(i), int(j), float(dist_matrix[i, j]), s))
        if (idx + 1) % 50 == 0:
            print(f"  ...verified {idx+1}/{len(cand_i)} candidates via SSIM")

    print(f"[ct_leakage_audit] confirmed near-duplicate pairs (SSIM >= {SSIM_CONFIRM_THRESHOLD}): {len(confirmed_pairs)}")

    uf = UnionFind(n)
    for i, j, d, s in confirmed_pairs:
        uf.union(i, j)

    clusters = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)
    clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    print(f"[ct_leakage_audit] near-duplicate clusters (size >= 2): {len(clusters)}")

    cluster_reports = []
    n_cross_split = 0
    n_images_in_leaky_clusters = 0
    for root, members in clusters.items():
        splits_involved = sorted(set(records[m][1] for m in members))
        classes_involved = sorted(set(records[m][2] for m in members))
        is_cross_split = len(splits_involved) > 1
        if is_cross_split:
            n_cross_split += 1
            n_images_in_leaky_clusters += len(members)
        cluster_reports.append({
            "size": len(members),
            "splits": splits_involved,
            "classes": classes_involved,
            "cross_split": is_cross_split,
            "files": [str(records[m][0].relative_to(ROOT)) for m in members],
        })

    cluster_reports.sort(key=lambda c: (-c["cross_split"], -c["size"]))

    print(f"[ct_leakage_audit] clusters split across train/val/test: {n_cross_split} / {len(clusters)}")
    print(f"[ct_leakage_audit] images involved in cross-split clusters: {n_images_in_leaky_clusters} / {n}")

    out = {
        "total_images": n,
        "candidate_pairs": len(cand_i),
        "confirmed_pairs": len(confirmed_pairs),
        "clusters_total": len(clusters),
        "clusters_cross_split": n_cross_split,
        "images_in_cross_split_clusters": n_images_in_leaky_clusters,
        "clusters": cluster_reports,
    }
    out_path = ROOT / "audit" / "ct_leakage_audit.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ct_leakage_audit] full report saved to {out_path}")

    return out


if __name__ == "__main__":
    main()
