"""Leakage audit for an arbitrary CT data dir -- a copy of
scripts/ct_leakage_audit.py (unmodified there) parameterised by --data_dir and
--out so it can be pointed at data/ct_scan_normal_augmented/ without touching the
original script or its audit/ct_leakage_audit.json output.

Same method: phash first pass -> SSIM confirmation -> union-find clustering ->
report which clusters span train/val/test. A cross-split cluster is the exact bug
that made this project's original CT test accuracy meaningless (35/109 test
images had a near-duplicate sibling in train).

IMPORTANT: point this only at training data. The held-out Group B set
(eval_run/ct_normal_heldout/) is deliberately never audited against the training
data -- deduping it away would destroy the thing it is measuring.
"""
import argparse, json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from skimage.metrics import structural_similarity as ssim

SPLITS = ["train", "val", "test"]
CLASSES = ["Benign", "Malignant", "Normal"]
HASH_SIZE = 8
CANDIDATE_HAMMING_MAX = 12
SSIM_CONFIRM_THRESHOLD = 0.90
SSIM_RESIZE = (256, 256)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    records = []
    for split in SPLITS:
        for cls in CLASSES:
            d = data_dir / split / cls
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    records.append((p, split, cls))
    n = len(records)
    print(f"[audit] {data_dir}: {n} images")

    bits = np.array([imagehash.phash(Image.open(p).convert("L"), hash_size=HASH_SIZE).hash.flatten()
                     for p, _, _ in records])
    print("[audit] hashed")
    dist = squareform(pdist(bits, metric="hamming")) * bits.shape[1]
    iu = np.triu_indices(n, k=1)
    mask = dist[iu] <= CANDIDATE_HAMMING_MAX
    ci, cj = iu[0][mask], iu[1][mask]
    print(f"[audit] candidate pairs (hamming<={CANDIDATE_HAMMING_MAX}): {len(ci)}")

    cache = {}

    def gray(p):
        if p not in cache:
            cache[p] = np.array(Image.open(p).convert("L").resize(SSIM_RESIZE))
        return cache[p]

    confirmed = []
    for k, (i, j) in enumerate(zip(ci, cj)):
        s = float(ssim(gray(records[i][0]), gray(records[j][0])))
        if s >= SSIM_CONFIRM_THRESHOLD:
            confirmed.append((int(i), int(j), float(dist[i, j]), s))
        if (k + 1) % 1000 == 0:
            print(f"  ...ssim {k+1}/{len(ci)}")
    print(f"[audit] SSIM-confirmed near-duplicate pairs: {len(confirmed)}")

    uf = UnionFind(n)
    for i, j, _, _ in confirmed:
        uf.union(i, j)
    clusters = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)
    clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    reports, n_cross, n_imgs_cross = [], 0, 0
    for members in clusters.values():
        splits_in = sorted(set(records[m][1] for m in members))
        classes_in = sorted(set(records[m][2] for m in members))
        cross = len(splits_in) > 1
        if cross:
            n_cross += 1
            n_imgs_cross += len(members)
        reports.append({"size": len(members), "splits": splits_in, "classes": classes_in,
                        "cross_split": cross,
                        "files": [str(records[m][0].relative_to(data_dir)) for m in members]})
    reports.sort(key=lambda c: (-c["cross_split"], -c["size"]))

    print(f"[audit] clusters (size>=2): {len(clusters)}")
    print(f"[audit] clusters spanning train/val/test: {n_cross}")
    print(f"[audit] images in cross-split clusters: {n_imgs_cross} / {n}")
    mixed = [r for r in reports if len(r["classes"]) > 1]
    print(f"[audit] clusters spanning >1 CLASS (label-conflict risk): {len(mixed)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "data_dir": str(data_dir), "total_images": n,
        "candidate_pairs": int(len(ci)), "confirmed_pairs": len(confirmed),
        "clusters_total": len(clusters), "clusters_cross_split": n_cross,
        "images_in_cross_split_clusters": n_imgs_cross,
        "clusters_cross_class": len(mixed),
        "clusters": reports,
    }, indent=2), encoding="utf-8")
    print(f"[audit] wrote {args.out}")


if __name__ == "__main__":
    main()
