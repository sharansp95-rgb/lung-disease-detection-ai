"""Group the external (mohamedhanyyy/chest-ctscan-images) Normal CT pool into
patient-proxy groups, using the SAME method as scripts/ct_leakage_audit.py
(phash first pass -> SSIM confirmation -> union-find clustering), additionally
unioned with the filename base-ID heuristic ('10 - Copy (2).png' -> '10').

Read-only. Writes eval_run/ct_normal_ext_groups.json.
"""
import json, os, re, pathlib
import numpy as np
import imagehash
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from skimage.metrics import structural_similarity as ssim

ROOT = pathlib.Path(r"D:/7th sem/sdp")
SRC = ROOT / "eval_run" / "external_samples" / "_downloads" / "ctscan" / "Data"
SPLITS = ["train", "valid", "test"]

HASH_SIZE = 8
CANDIDATE_HAMMING_MAX = 12
SSIM_CONFIRM_THRESHOLD = 0.90
SSIM_RESIZE = (256, 256)


def base_id(name):
    n = os.path.splitext(name)[0]
    n = re.sub(r"\s*-\s*Copy", "", n)
    n = re.sub(r"\s*\(\d+\)", "", n)
    return n.strip()


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


files = []
for sp in SPLITS:
    d = SRC / sp / "normal"
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            files.append(p)
n = len(files)
print(f"[ext_cluster] external Normal files: {n}")

bits = np.array([imagehash.phash(Image.open(p).convert("L"), hash_size=HASH_SIZE).hash.flatten()
                 for p in files])
dist = squareform(pdist(bits, metric="hamming")) * bits.shape[1]
iu = np.triu_indices(n, k=1)
mask = dist[iu] <= CANDIDATE_HAMMING_MAX
ci, cj = iu[0][mask], iu[1][mask]
print(f"[ext_cluster] candidate pairs (hamming<={CANDIDATE_HAMMING_MAX}): {len(ci)}")

cache = {}


def gray(p):
    if p not in cache:
        cache[p] = np.array(Image.open(p).convert("L").resize(SSIM_RESIZE))
    return cache[p]


uf = UnionFind(n)
confirmed = 0
for k, (i, j) in enumerate(zip(ci, cj)):
    s = float(ssim(gray(files[i]), gray(files[j])))
    if s >= SSIM_CONFIRM_THRESHOLD:
        uf.union(int(i), int(j))
        confirmed += 1
    if (k + 1) % 500 == 0:
        print(f"  ...ssim {k+1}/{len(ci)}")
print(f"[ext_cluster] SSIM-confirmed near-duplicate pairs: {confirmed}")

# also union by filename base id -- belt and braces
by_base = {}
for idx, p in enumerate(files):
    by_base.setdefault(base_id(p.name), []).append(idx)
for b, idxs in by_base.items():
    for x in idxs[1:]:
        uf.union(idxs[0], x)

groups = {}
for idx in range(n):
    groups.setdefault(uf.find(idx), []).append(idx)
print(f"[ext_cluster] distinct patient-proxy groups: {len(groups)}")

out = []
for gid, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    bases = sorted(set(base_id(files[m].name) for m in members))
    out.append({
        "group_id": len(out),
        "size": len(members),
        "base_ids": bases,
        "files": [str(files[m].relative_to(ROOT)) for m in members],
    })
    print(f"  group{len(out)-1:3d} n={len(members):3d} base_ids={bases}")

(ROOT / "eval_run" / "ct_normal_ext_groups.json").write_text(
    json.dumps({"n_files": n, "n_groups": len(out),
                "ssim_confirmed_pairs": confirmed, "groups": out}, indent=2),
    encoding="utf-8")
print("[ext_cluster] wrote eval_run/ct_normal_ext_groups.json")
