"""Step 1+2 of the CT-Normal-augmentation experiment: build a deduped pool of
external Normal chest-CT images and split it into a train pool (Group A) and a
never-trained-on held-out set (Group B), disjoint by patient/study.

Two independent sources, neither of them IQ-OTH/NCCD:

  mohamedhanyyy/chest-ctscan-images   -- 215 raw 'normal' files that collapse to
      28 patient-proxy groups (phash+SSIM near-duplicate clustering, same method
      as scripts/ct_leakage_audit.py, unioned with the filename base-ID
      heuristic). Groups come from eval_run/ct_normal_ext_groups.json.
  maedemaftouni/large-covid19-ct-slice-dataset -- the curated '1NonCOVID'
      (normal) class, grouped by the study ID embedded in each filename.

Group B (held out, never seen by training or the leakage audit against training
data):
  * mohamedhanyyy: the 9 patient groups that the live checkpoint
    ct_best_model_phase35_v2.pt got wrong in the original external validation
    (eval_run/external_samples/ct/normal/) -- keeping these out gives a direct
    before/after on the exact documented failure set.
  * maedemaftouni: a seeded random quota of patients per sub-source, disjoint
    from the Group A patients -- an unbiased held-out sample, since the
    mohamedhanyyy half of Group B is deliberately the known-hard set.

Outputs (nothing under data/ct_scan/ or checkpoints/ is touched):
  datasets/ct_normal_extra/groupA/<source>/   -- train pool
  eval_run/ct_normal_heldout/groupB_<source>/ -- held-out test set
  eval_run/ct_normal_pool_manifest.json
"""
import hashlib, json, random, re, shutil
from pathlib import Path

from PIL import Image

SEED = 1337
ROOT = Path(__file__).resolve().parent.parent
MH_GROUPS_JSON = ROOT / "eval_run" / "ct_normal_ext_groups.json"
MH_EVAL_DIR = ROOT / "eval_run" / "external_samples" / "ct" / "normal"
MF_DIR = ROOT / "eval_run" / "external_samples" / "_downloads" / "covidct_slice" / "1NonCOVID"

OUT_A = ROOT / "datasets" / "ct_normal_extra" / "groupA"
OUT_B = ROOT / "eval_run" / "ct_normal_heldout"
MANIFEST = ROOT / "eval_run" / "ct_normal_pool_manifest.json"

# mohamedhanyyy group ids to drop entirely (not chest CT / unusable)
MH_EXCLUDE_GROUPS = {23}  # '004007_01_01_519.png' -- a pelvis/abdomen slice, no lung fields

# patients sampled per maftouni sub-source (see build_maftouni)
MF_STUDY_QUOTA_A = {"rahimzadeh": 40, "morozov": 40, "n_series": 30}
MF_STUDY_QUOTA_B = {"rahimzadeh": 16, "morozov": 16, "n_series": 12}
MF_SLICES_PER_STUDY = 4   # evenly spaced through each study's slice range
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mf_study_id(name: str) -> str:
    """Patient/study id from a maftouni 1NonCOVID filename. Three sub-sources,
    three shapes:
      '6_Rahimzadeh_normal10_patient151_SR_4_IM00012.png'
                                     -> '6_Rahimzadeh_patient151'
      '16_Morozov_study_0001_12.png' -> '16_Morozov_study_0001'
      'N001_121.png'                 -> 'N001'

    NOTE on the Rahimzadeh set: the 'normalN' token is a batch/folder number, not
    part of the patient's identity -- '..._normal4_patient171_...' and
    '..._normal5_patient171_...' are the same patient, confirmed by the leakage
    audit (phash+SSIM matched them as near-duplicates when an earlier version of
    this function treated them as two patients and split them across train/val).
    So the patient key for that sub-source is the 'patientNNN' token alone.
    """
    stem = Path(name).stem
    m = re.search(r"patient\d+", stem)
    if m:
        return f"6_Rahimzadeh_{m.group(0)}"
    return re.sub(r"_\d+$", "", stem)


def mf_slice_index(name: str) -> int:
    nums = re.findall(r"\d+", Path(name).stem)
    return int(nums[-1]) if nums else 0


def mf_subsource(study_id: str) -> str:
    if study_id.startswith("6_"):
        return "rahimzadeh"
    if study_id.startswith("16_"):
        return "morozov"
    return "n_series"


def build_mohamedhanyyy():
    groups = json.loads(MH_GROUPS_JSON.read_text())["groups"]

    # which groups do the 9 previously-evaluated external Normal images belong to?
    hash_to_group = {}
    for g in groups:
        for f in g["files"]:
            hash_to_group[sha(ROOT / f)] = g["group_id"]
    eval_groups = set()
    eval_map = {}
    for p in sorted(MH_EVAL_DIR.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            gid = hash_to_group[sha(p)]
            eval_groups.add(gid)
            eval_map[p.name] = gid

    a, b = [], []
    for g in groups:
        gid = g["group_id"]
        if gid in MH_EXCLUDE_GROUPS:
            continue
        (b if gid in eval_groups else a).append(g)
    return a, b, eval_map


def build_maftouni():
    """1NonCOVID has ~470 patients across three sub-sources -- far more than this
    experiment can absorb without the Normal class swamping Benign (96 train) and
    Malignant (400 train). So sample a fixed quota of *patients* per sub-source
    (rather than a flat fraction, which would be ~75% Rahimzadeh) and take a few
    evenly-spaced slices from each. Group A and Group B draw disjoint patients
    from the same shuffled per-sub-source order."""
    if not MF_DIR.is_dir():
        raise SystemExit(f"missing {MF_DIR} -- extract the 1NonCOVID class there first")
    studies = {}
    for p in sorted(MF_DIR.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            studies.setdefault(mf_study_id(p.name), []).append(p)

    by_sub = {}
    for sid in studies:
        by_sub.setdefault(mf_subsource(sid), []).append(sid)

    rng = random.Random(SEED)
    a_ids, b_ids = [], []
    for sub, ids in sorted(by_sub.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        na, nb = MF_STUDY_QUOTA_A[sub], MF_STUDY_QUOTA_B[sub]
        a_ids += ids[:na]
        b_ids += ids[na:na + nb]
        print(f"[pool]   maftouni/{sub}: {len(ids)} patients available -> "
              f"{len(ids[:na])} A / {len(ids[na:na+nb])} B")
    return studies, sorted(a_ids), sorted(b_ids)


def pick_slices(paths, k):
    """k slices spread evenly across the study's slice-index range, so one study
    contributes different anatomical levels rather than k adjacent duplicates."""
    ps = sorted(paths, key=lambda p: mf_slice_index(p.name))
    if len(ps) <= k:
        return ps
    step = (len(ps) - 1) / (k - 1) if k > 1 else 0
    return [ps[round(i * step)] for i in range(k)]


def copy_all(pairs, dest: Path):
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for src, name in pairs:
        try:
            with Image.open(src) as im:
                im.verify()
        except Exception:
            print(f"  skipping unreadable {src}")
            continue
        shutil.copy2(src, dest / name)
        written.append(name)
    return written


def main():
    mh_a, mh_b, eval_map = build_mohamedhanyyy()
    print(f"[pool] mohamedhanyyy: {len(mh_a)} groups -> A, {len(mh_b)} groups -> B "
          f"(excluded {sorted(MH_EXCLUDE_GROUPS)})")
    print(f"[pool]   held-out groups = the {len(set(eval_map.values()))} that the "
          f"original external-validation Normal images belong to: {sorted(set(eval_map.values()))}")

    # one representative per group is the diversity that matters; the near-duplicate
    # siblings add sample weight without adding information, so take the group's
    # first file only.
    mh_a_pairs = [(ROOT / g["files"][0], f"mh_g{g['group_id']:02d}_{g['base_ids'][0]}.png") for g in mh_a]
    # Group B keeps every file of each held-out group, so the 9 original
    # external-validation images are present verbatim.
    mh_b_pairs = []
    for g in mh_b:
        for i, f in enumerate(g["files"]):
            mh_b_pairs.append((ROOT / f, f"mh_g{g['group_id']:02d}_{i:02d}.png"))

    studies, mf_a_ids, mf_b_ids = build_maftouni()
    print(f"[pool] maftouni 1NonCOVID: {len(studies)} studies, "
          f"{sum(len(v) for v in studies.values())} slices -> "
          f"{len(mf_a_ids)} studies A / {len(mf_b_ids)} studies B")

    # p.stem already carries the study id + slice index, so `mf_<stem>` keeps the
    # study recoverable by prepare_ct_scan_normal_augmented.py::group_key
    mf_a_pairs = [(p, f"mf_{p.stem}.png")
                  for sid in mf_a_ids for p in pick_slices(studies[sid], MF_SLICES_PER_STUDY)]
    mf_b_pairs = [(p, f"mf_{p.stem}.png")
                  for sid in mf_b_ids for p in pick_slices(studies[sid], MF_SLICES_PER_STUDY)]

    written = {
        "A_mohamedhanyyy": copy_all(mh_a_pairs, OUT_A / "mohamedhanyyy"),
        "A_maftouni": copy_all(mf_a_pairs, OUT_A / "maftouni"),
        "B_mohamedhanyyy": copy_all(mh_b_pairs, OUT_B / "groupB_mohamedhanyyy"),
        "B_maftouni": copy_all(mf_b_pairs, OUT_B / "groupB_maftouni"),
    }
    for k, v in written.items():
        print(f"[pool] {k}: {len(v)} images")

    # sanity: A and B must share no image content at all
    ha = {sha(p) for p in (OUT_A).rglob("*.png")}
    hb = {sha(p) for p in (OUT_B).rglob("*.png")}
    overlap = ha & hb
    print(f"[pool] content-hash overlap between Group A and Group B: {len(overlap)} "
          f"({'OK' if not overlap else 'VIOLATION'})")
    if overlap:
        raise SystemExit("Group A / Group B overlap -- aborting")

    MANIFEST.write_text(json.dumps({
        "seed": SEED,
        "sources": {
            "mohamedhanyyy": "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images",
            "maftouni": "https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset",
        },
        "mohamedhanyyy": {
            "excluded_groups": sorted(MH_EXCLUDE_GROUPS),
            "groupA_group_ids": [g["group_id"] for g in mh_a],
            "groupB_group_ids": [g["group_id"] for g in mh_b],
            "original_external_eval_image_to_group": eval_map,
        },
        "maftouni": {
            "n_studies": len(studies), "groupA_studies": mf_a_ids, "groupB_studies": mf_b_ids,
            "slices_per_study": MF_SLICES_PER_STUDY,
            "study_quota_A": MF_STUDY_QUOTA_A, "study_quota_B": MF_STUDY_QUOTA_B,
        },
        "counts": {k: len(v) for k, v in written.items()},
        "groupA_groupB_hash_overlap": len(overlap),
    }, indent=2), encoding="utf-8")
    print(f"[pool] wrote {MANIFEST}")


if __name__ == "__main__":
    main()
