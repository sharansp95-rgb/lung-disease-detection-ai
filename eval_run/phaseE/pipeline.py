"""Phase E: the two-stage detector -> classifier pipeline, in one place.

STANDALONE. Nothing under `backend/` imports this, and this imports nothing from
`backend/`. The live `/predict/ct` path and `checkpoints/ct_best_model_phase35_v2.pt`
are untouched by design.

Stage 1  MONAI `lung_nodule_ct_detection` (3D RetinaNet, LUNA16) -- reused verbatim from
         Phase D (`run_detector.build_detector` / `preprocess`), pinned to the bundle's
         NATIVE 192x192x80 training patch. Phase D measured that the bundle's shipped
         512x512x192 default silently spills to host RAM on this 4 GB card and runs 4.4x
         slower, so the ROI is pinned here and not read from configs/inference.json.
Stage 2  `checkpoints/ct_lidc_nodulecrop_v2_jitter.pt` (Phase C.1), on a 28 mm
         nodule-centred crop cut with Phase B's own `box_for` / `window` / 128 px
         resize -- imported from `lidc_full/extract_crops_full.py`, not reimplemented.

Coordinate handling is Phase D's `volume_io` (validated in Phase D S3) plus the exact
algebraic inverse of its `annotation_to_ras`, so a detector box centre in RAS mm maps
back to a (slice, row, col) in the ORIGINAL series geometry. Crops are cut from the
original slices, NOT from the 0.703/0.703/1.25 mm resampled volume, so they are
distributed like the crops the classifier was trained on.
"""
import ctypes
import ctypes.wintypes as wintypes
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PHASE_D = os.path.join(ROOT, "eval_run", "phaseD")
LIDC_FULL = os.path.join(ROOT, "eval_run", "lidc_full")
for p in (PHASE_D, LIDC_FULL, os.path.join(ROOT, "ml")):
    if p not in sys.path:
        sys.path.insert(0, p)

from volume_io import read_series, LPS_TO_RAS                     # noqa: E402
from run_detector import build_detector, preprocess, vox_to_ras   # noqa: E402
from extract_crops_full import box_for, window, OUT_PX            # noqa: E402
from model import build_model                                     # noqa: E402
from dataset import build_transforms                              # noqa: E402

CLASS_NAMES = ["Benign", "Malignant", "Normal"]
CROP_MM = 28.0
DEFAULT_ROI = (192, 192, 80)          # Phase D S2 -- do NOT use the bundle default
DEFAULT_OVERLAP = 0.25
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "ct_lidc_nodulecrop_v2_jitter.pt")
PIXDIM = (0.703125, 0.703125, 1.25)


# --------------------------------------------------------------- resource probes
class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_k32 = ctypes.windll.kernel32
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_gpmi = ctypes.windll.psapi.GetProcessMemoryInfo
_gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
_gpmi.restype = wintypes.BOOL


def host_rss_gb():
    """(current, peak) process working set in GB. psutil is not installed and this
    phase deliberately adds no dependency, so this goes straight to psapi.
    (argtypes matter: the default int restype truncates the 64-bit pseudo-handle and
    the call silently returns 0.)"""
    c = _PMC()
    c.cb = ctypes.sizeof(_PMC)
    if not _gpmi(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None, None
    return c.WorkingSetSize / 2**30, c.PeakWorkingSetSize / 2**30


def gpu_gb():
    if not torch.cuda.is_available():
        return {}
    return {"allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)}


# --------------------------------------------------- RAS mm -> original series voxel
def ras_to_series_rc(ras, meta):
    """Exact inverse of volume_io.annotation_to_ras.

    Returns (k, sop_uid, center_row, center_col, dz_mm) where k indexes the slice
    ordering read_series produced (sorted along the true slice normal) and dz_mm is
    how far the requested point is from that slice's plane -- the residual the
    nearest-slice snap costs, worth reporting rather than hiding.
    """
    iop = np.asarray(meta["iop"], dtype=np.float64)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    ps = meta["pixel_spacing"]                      # [row_spacing, col_spacing]
    lps = LPS_TO_RAS[:3, :3] @ np.asarray(ras, dtype=np.float64)

    sops = sorted(meta["sop_to_z"], key=lambda s: meta["sop_to_z"][s])
    zs = np.array([meta["sop_to_z"][s] for s in sops], dtype=np.float64)
    t = float(np.dot(lps, normal))
    k = int(np.argmin(np.abs(zs - t)))
    ipp = np.asarray(meta["sop_to_ipp"][sops[k]], dtype=np.float64)
    d = lps - ipp
    center_col = float(np.dot(d, row_dir)) / ps[1]
    center_row = float(np.dot(d, col_dir)) / ps[0]
    return k, sops[k], center_row, center_col, float(t - zs[k])


def crop_at_ras(vol_xyz, meta, ras, crop_mm=CROP_MM):
    """Phase B's 28 mm axial crop, cut from the ORIGINAL slice geometry at a RAS point.

    Uses extract_crops_full.box_for with px_mm = PixelSpacing[0], exactly as Phase B did,
    so a crop taken at an annotation's own position reproduces the stored PNG bit for bit
    (asserted by run_eval.py's fidelity check, not assumed).
    """
    k, sop, cr, cc, dz = ras_to_series_rc(ras, meta)
    hu = vol_xyz[:, :, k].T                          # (row, col)
    px_mm = float(meta["pixel_spacing"][0])
    b = box_for(hu.shape, cr, cc, px_mm, crop_mm)
    if b is None:
        return None
    r0, c0, r1, c1 = b
    return {"hu": hu[r0:r1, c0:c1], "slice_index": k, "sop_uid": sop,
            "center_row": round(cr, 2), "center_col": round(cc, 2),
            "dz_to_slice_mm": round(dz, 2), "px_mm": round(px_mm, 4),
            "box": [int(r0), int(c0), int(r1), int(c1)]}


def crop_to_pil(hu_crop):
    """Identical pixels to extract_crops_full.save_png, without touching disk."""
    img = (window(hu_crop) * 255).astype(np.uint8)
    return Image.fromarray(img).resize((OUT_PX, OUT_PX), Image.BILINEAR)


# --------------------------------------------------------------- the pipeline
def load_classifier(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = build_model(ck["backbone"], ck["num_classes"], pretrained=False).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    meta = {"file": os.path.basename(ckpt_path), "epoch": ck.get("epoch"),
            "val_acc": ck.get("val_acc"), "val_balanced_acc": ck.get("val_balanced_acc"),
            "class_names": ck.get("class_names")}
    if meta["class_names"] and list(meta["class_names"]) != CLASS_NAMES:
        raise RuntimeError(f"class order changed: {meta['class_names']}")
    return m, meta


def aggregate(candidates, threshold):
    """Scan-level call from per-candidate classifications.

    Rule: any candidate the classifier calls Malignant flags the scan Malignant; the
    single candidate with the highest P(Malignant) is named as the driver. All
    candidates are still returned -- a scan report that hides the other findings is
    not a clinical-style report.
    """
    kept = [c for c in candidates if c["detector_score"] >= threshold]
    kept = sorted(kept, key=lambda c: -c["probs"][1])
    if not kept:
        return {"threshold": threshold, "scan_call": "No candidate detected",
                "n_candidates": 0, "driver": None, "candidates": [],
                "counts": {k: 0 for k in CLASS_NAMES}}
    counts = {k: sum(1 for c in kept if c["pred"] == k) for k in CLASS_NAMES}
    if counts["Malignant"]:
        call = "Malignant"
    elif counts["Benign"]:
        call = "Benign"
    else:
        call = "No nodule confirmed (all candidates classified Normal)"
    return {"threshold": threshold, "scan_call": call, "n_candidates": len(kept),
            "driver": kept[0]["candidate_id"], "max_p_malignant": round(kept[0]["probs"][1], 4),
            "counts": counts, "candidates": kept}


class TwoStagePipeline:
    """Both models resident in one process -- the configuration Phase E exists to test."""

    def __init__(self, device="cuda", roi=DEFAULT_ROI, overlap=DEFAULT_OVERLAP,
                 ckpt=DEFAULT_CKPT, sw_batch=1):
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available())
                                   else "cpu")
        self.roi = list(roi)
        self.overlap = overlap
        self.mem = {"start": gpu_gb()}

        t0 = time.time()
        self.detector = build_detector(self.device)
        self.detector.set_sliding_window_inferer(
            roi_size=self.roi, sw_batch_size=sw_batch, overlap=overlap, mode="constant",
            device=torch.device("cpu"), sw_device=self.device)
        self.sec_load_detector = round(time.time() - t0, 1)
        self.mem["after_detector"] = gpu_gb()

        t0 = time.time()
        self.classifier, self.ckpt_meta = load_classifier(ckpt, self.device)
        self.sec_load_classifier = round(time.time() - t0, 1)
        self.mem["after_both"] = gpu_gb()
        self.transform = build_transforms(train=False)
        rss, peak = host_rss_gb()
        self.mem["host_rss_gb_after_both"] = round(rss, 2) if rss else None

    # -------------------------------------------------------------- stage 2 helper
    def classify_crops(self, hu_crops, bs=32):
        if not hu_crops:
            return np.zeros((0, 3))
        out = []
        with torch.no_grad():
            for i in range(0, len(hu_crops), bs):
                batch = torch.stack([
                    self.transform(crop_to_pil(c).convert("RGB"))
                    for c in hu_crops[i:i + bs]]).to(self.device)
                out.append(torch.softmax(self.classifier(batch), 1).cpu().numpy())
        return np.concatenate(out)

    # -------------------------------------------------------------- the whole thing
    def run(self, series_dir, thresholds=(0.5, 0.3), progress=None):
        def say(msg, frac):
            if progress:
                progress(msg, frac)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        timings, t_all = {}, time.time()

        say("reading DICOM series", 0.05)
        t0 = time.time()
        vol, aff, meta = read_series(series_dir)
        img, _hu = preprocess(vol, aff)
        timings["read_and_resample_sec"] = round(time.time() - t0, 1)
        aff_pp = np.asarray(img.affine.cpu().numpy(), dtype=np.float64)

        say(f"stage 1: detecting on {list(img.shape[1:])} volume", 0.15)
        t0 = time.time()
        with torch.no_grad():
            with torch.autocast("cuda", enabled=(self.device.type == "cuda")):
                pred = self.detector(img[None].to(torch.float32), use_inferer=True)
        timings["detect_sec"] = round(time.time() - t0, 1)
        mem_after_detect = gpu_gb()

        p = pred[0]
        boxes = p["box"].detach().cpu().numpy().astype(np.float64)
        scores = p["label_scores"].detach().cpu().numpy().astype(np.float64)
        if len(boxes):
            ctr_vox = (boxes[:, :3] + boxes[:, 3:]) / 2.0
            ctr_ras = vox_to_ras(aff_pp, ctr_vox)
            size_mm = (boxes[:, 3:] - boxes[:, :3]) * np.array(PIXDIM)
        else:
            ctr_vox = ctr_ras = size_mm = np.zeros((0, 3))

        lo = float(min(thresholds))
        keep = np.where(scores >= lo)[0]
        say(f"stage 2: classifying {len(keep)} candidates", 0.75)

        t0 = time.time()
        cands, crops = [], []
        for n, i in enumerate(keep):
            c = crop_at_ras(vol, meta, ctr_ras[i])
            rec = {"candidate_id": f"c{n:02d}",
                   "detector_score": round(float(scores[i]), 4),
                   "center_ras_mm": [round(float(v), 1) for v in ctr_ras[i]],
                   "box_size_mm": [round(float(v), 1) for v in size_mm[i]]}
            if c is None:
                rec.update({"crop": None, "probs": [0.0, 0.0, 1.0], "pred": "Normal",
                            "note": "28 mm crop falls outside the slice - not classified"})
            else:
                rec["crop"] = {k: v for k, v in c.items() if k != "hu"}
                crops.append(c["hu"])
                rec["_crop_idx"] = len(crops) - 1
            cands.append(rec)
        probs = self.classify_crops(crops)
        for rec in cands:
            if "_crop_idx" in rec:
                pr = probs[rec.pop("_crop_idx")]
                rec["probs"] = [round(float(v), 4) for v in pr]
                rec["pred"] = CLASS_NAMES[int(pr.argmax())]
                rec["confidence"] = round(float(pr.max()), 4)
        timings["classify_sec"] = round(time.time() - t0, 1)
        timings["total_sec"] = round(time.time() - t_all, 1)
        say("done", 1.0)

        rss, rss_peak = host_rss_gb()
        return {
            "series_dir": series_dir,
            "series": {"n_slices": meta["n_slices"], "shape_xyz": meta["shape_xyz"],
                       "pixel_spacing": meta["pixel_spacing"],
                       "slice_spacing_mm": round(meta["slice_spacing_mm"], 3),
                       "manufacturer": meta["manufacturer"],
                       "resampled_shape": list(img.shape[1:])},
            "config": {"roi": self.roi, "overlap": self.overlap,
                       "device": self.device.type, "crop_mm": CROP_MM,
                       "classifier": self.ckpt_meta},
            "timings": timings,
            "resources": {"gpu_after_detect": mem_after_detect, "gpu_end": gpu_gb(),
                          "host_rss_gb": round(rss, 2) if rss else None,
                          "host_rss_peak_gb": round(rss_peak, 2) if rss_peak else None},
            "n_raw_detections": int(len(scores)),
            "candidates": cands,
            "reports": {str(t): aggregate(cands, t) for t in thresholds},
            # kept for the evaluation harness; not part of the "API" payload
            "_meta": meta, "_aff_pp": aff_pp, "_vol": vol,
        }
