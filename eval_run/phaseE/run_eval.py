"""Phase E step 2: run the FULL pipeline end to end on Phase D's 10 volumes / 22 nodules
and count real compounded outcomes.

This is deliberately NOT detection-rate x classifier-accuracy. Both stages actually run in
sequence on real DICOM, and every nodule gets one of:

    missed_by_detector          no candidate within MATCH_MM of the annotation
    correct                     detected AND the matched candidate's class == LIDC label
    misclassified_BM            detected, but Benign<->Malignant crossed
    called_Normal               detected, but the classifier says "no nodule in this patch"

Two controls run alongside, because a compounded number is only interpretable with them:
  * the GT-centred crop through the same code path (isolates "did the detector's crop
    centre hurt the classifier" from "the classifier was going to be wrong anyway");
  * a bit-exactness check of that GT-centred crop against Phase B's stored PNG, so the
    extraction path is proven identical to the classifier's training distribution rather
    than assumed to be.

Matching logic is Phase D's analyze.py: nearest detection per GT nodule, independently
per nodule; an unmatched detection is one >15 mm from every annotation.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import (TwoStagePipeline, crop_at_ras, crop_to_pil, CLASS_NAMES,   # noqa: E402
                      gpu_gb, host_rss_gb)

PHASE_D_WORK = os.path.abspath(os.path.join(HERE, "..", "phaseD", "_work"))
LIDC_FULL = os.path.abspath(os.path.join(HERE, "..", "lidc_full"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "phaseD")))
from volume_io import annotation_to_ras                                          # noqa: E402

MATCH_MM = 8.0        # Phase D's headline detection radius
FP_MM = 15.0          # Phase D's "unmatched detection" radius
THRESHOLDS = (0.5, 0.3)
OUT = os.path.join(HERE, "_work")


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = json.load(open(os.path.join(PHASE_D_WORK, "cases.json")))

    rss0, _ = host_rss_gb()
    print(f"[mem] host RSS before any model: {rss0:.2f} GB")
    pipe = TwoStagePipeline(device="cuda")
    print(f"[load] detector {pipe.sec_load_detector}s, classifier "
          f"{pipe.sec_load_classifier}s ({pipe.ckpt_meta['file']}, "
          f"epoch {pipe.ckpt_meta['epoch']}, val_bal {pipe.ckpt_meta['val_balanced_acc']:.4f})")
    print(f"[mem] {json.dumps(pipe.mem)}")

    results, per_nodule, per_fp, per_case = [], [], [], []
    peak_gpu, peak_rss = 0.0, 0.0

    for ci, c in enumerate(cases, 1):
        pid = c["patient_id"]
        d = os.path.join(PHASE_D_WORK, "volumes", pid)
        if not os.path.exists(os.path.join(d, "_complete.json")):
            print(f"[{ci}/{len(cases)}] {pid}: not downloaded, skip")
            continue

        r = pipe.run(d, thresholds=THRESHOLDS)
        vol, meta = r.pop("_vol"), r.pop("_meta")
        r.pop("_aff_pp")
        cands = r["candidates"]
        ct = np.array([cd["center_ras_mm"] for cd in cands],
                      dtype=np.float64).reshape(-1, 3)
        sc = np.array([cd["detector_score"] for cd in cands], dtype=np.float64)

        # ---- ground truth in the same RAS frame, plus the GT-centred control crop
        gts, gt_ras, gt_crops, gt_rows = [], [], [], []
        for n in c["nodules"]:
            if n["sop_uid"] not in meta["sop_to_ipp"]:
                continue
            ras = annotation_to_ras(n["sop_uid"], n["center_row"], n["center_col"], meta)
            cp = crop_at_ras(vol, meta, ras)
            if cp is None:
                continue
            gts.append(n)
            gt_ras.append(ras)
            gt_crops.append(cp["hu"])
            # fidelity of this extraction path vs Phase B's stored PNG for this nodule
            png = os.path.join(LIDC_FULL, n["image"].replace("/", os.sep))
            diff = None
            if os.path.exists(png):
                ref = np.asarray(Image.open(png).convert("L"), dtype=np.int32)
                got = np.asarray(crop_to_pil(cp["hu"]), dtype=np.int32)
                diff = int(np.abs(got - ref).max()) if got.shape == ref.shape else -1
            gt_rows.append({"sop_match": cp["sop_uid"] == n["sop_uid"],
                            "png_max_abs_diff": diff})
        gt_ras = np.array(gt_ras, dtype=np.float64).reshape(-1, 3)
        gt_probs = pipe.classify_crops(gt_crops)

        # ---- per-nodule compounded outcome, at each operating point
        for gi, n in enumerate(gts):
            row = {"patient_id": pid, "image": os.path.basename(n["image"]),
                   "label": n["label"], "diam_mm": n["nodule_diam_mm"],
                   "gt_centred_pred": CLASS_NAMES[int(gt_probs[gi].argmax())],
                   "gt_centred_probs": [round(float(v), 4) for v in gt_probs[gi]],
                   "extraction_fidelity": gt_rows[gi]}
            for t in THRESHOLDS:
                m = sc >= t
                key = f"@{t}"
                if m.sum() == 0:
                    row[f"outcome{key}"] = "missed_by_detector"
                    row[f"nearest_mm{key}"] = None
                    continue
                dist = np.linalg.norm(ct[m] - gt_ras[gi], axis=1)
                k = int(np.argmin(dist))
                idx = int(np.where(m)[0][k])
                row[f"nearest_mm{key}"] = round(float(dist[k]), 2)
                if dist[k] > MATCH_MM:
                    row[f"outcome{key}"] = "missed_by_detector"
                    continue
                cd = cands[idx]
                row[f"cand{key}"] = cd["candidate_id"]
                row[f"det_score{key}"] = cd["detector_score"]
                row[f"pred{key}"] = cd["pred"]
                row[f"probs{key}"] = cd["probs"]
                if cd["pred"] == n["label"]:
                    row[f"outcome{key}"] = "correct"
                elif cd["pred"] == "Normal":
                    row[f"outcome{key}"] = "called_Normal"
                else:
                    row[f"outcome{key}"] = "misclassified_BM"
            per_nodule.append(row)

        # ---- unmatched candidates (Phase D's FP definition), with their classes
        for i, cd in enumerate(cands):
            if len(gt_ras) and np.linalg.norm(ct[i] - gt_ras, axis=1).min() <= FP_MM:
                continue
            per_fp.append({"patient_id": pid, "candidate_id": cd["candidate_id"],
                           "detector_score": cd["detector_score"],
                           "pred": cd["pred"], "probs": cd["probs"],
                           "nearest_gt_mm": (round(float(np.linalg.norm(
                               ct[i] - gt_ras, axis=1).min()), 1) if len(gt_ras) else None)})

        pg = r["resources"]["gpu_end"].get("peak_allocated_gb", 0) if r["resources"]["gpu_end"] else 0
        peak_gpu = max(peak_gpu, pg or 0)
        peak_rss = max(peak_rss, r["resources"]["host_rss_peak_gb"] or 0)
        per_case.append({"patient_id": pid, "timings": r["timings"],
                         "n_raw": r["n_raw_detections"],
                         "n_cand@0.5": r["reports"]["0.5"]["n_candidates"],
                         "n_cand@0.3": r["reports"]["0.3"]["n_candidates"],
                         "call@0.5": r["reports"]["0.5"]["scan_call"],
                         "call@0.3": r["reports"]["0.3"]["scan_call"],
                         "peak_gpu_gb": pg,
                         "host_rss_gb": r["resources"]["host_rss_gb"]})
        results.append(r)
        t = r["timings"]
        print(f"[{ci}/{len(cases)}] {pid}: read {t['read_and_resample_sec']}s + "
              f"detect {t['detect_sec']}s + classify {t['classify_sec']}s = "
              f"{t['total_sec']}s | cand@0.5={r['reports']['0.5']['n_candidates']} "
              f"call={r['reports']['0.5']['scan_call']} | peak {pg}GB "
              f"rss {r['resources']['host_rss_gb']}GB", flush=True)
        del vol

    # ------------------------------------------------------------------ summary
    summary = {"n_cases": len(per_case), "n_nodules": len(per_nodule),
               "match_radius_mm": MATCH_MM, "fp_radius_mm": FP_MM,
               "classifier": pipe.ckpt_meta,
               "load_sec": {"detector": pipe.sec_load_detector,
                            "classifier": pipe.sec_load_classifier},
               "memory": pipe.mem, "peak_gpu_gb": round(peak_gpu, 3),
               "peak_host_rss_gb": round(peak_rss, 2)}
    for t in THRESHOLDS:
        k = f"@{t}"
        oc = [r[f"outcome{k}"] for r in per_nodule]
        cnt = {o: oc.count(o) for o in
               ("correct", "misclassified_BM", "called_Normal", "missed_by_detector")}
        by_lab = {}
        for lab in ("Benign", "Malignant"):
            sub = [r for r in per_nodule if r["label"] == lab]
            by_lab[lab] = {"n": len(sub),
                           "correct": sum(1 for r in sub if r[f"outcome{k}"] == "correct"),
                           "detected": sum(1 for r in sub
                                           if r[f"outcome{k}"] != "missed_by_detector")}
        det = len(per_nodule) - cnt["missed_by_detector"]
        summary[f"end_to_end{k}"] = {
            "counts": cnt,
            "detected": det,
            "detection_rate": round(det / len(per_nodule), 3),
            "end_to_end_accuracy": round(cnt["correct"] / len(per_nodule), 3),
            "classifier_accuracy_on_detected": round(cnt["correct"] / det, 3) if det else None,
            "by_label": by_lab}
    gtc = sum(1 for r in per_nodule if r["gt_centred_pred"] == r["label"])
    summary["control_gt_centred_accuracy"] = round(gtc / len(per_nodule), 3)
    summary["control_gt_centred_correct"] = gtc
    summary["extraction_fidelity"] = {
        "n": len(per_nodule),
        "sop_match": sum(1 for r in per_nodule if r["extraction_fidelity"]["sop_match"]),
        "png_bit_exact": sum(1 for r in per_nodule
                             if r["extraction_fidelity"]["png_max_abs_diff"] == 0),
        "max_png_diff": max((r["extraction_fidelity"]["png_max_abs_diff"] or 0)
                            for r in per_nodule)}
    fp05 = [f for f in per_fp if f["detector_score"] >= 0.5]
    fp03 = [f for f in per_fp if f["detector_score"] >= 0.3]
    summary["unmatched_candidates"] = {
        "@0.5": {"n": len(fp05), "per_scan": round(len(fp05) / max(len(per_case), 1), 2),
                 "by_class": {c: sum(1 for f in fp05 if f["pred"] == c) for c in CLASS_NAMES}},
        "@0.3": {"n": len(fp03), "per_scan": round(len(fp03) / max(len(per_case), 1), 2),
                 "by_class": {c: sum(1 for f in fp03 if f["pred"] == c) for c in CLASS_NAMES}}}
    tt = [c["timings"] for c in per_case]
    summary["latency_sec"] = {
        k: {"mean": round(float(np.mean([x[k] for x in tt])), 1),
            "min": round(float(np.min([x[k] for x in tt])), 1),
            "max": round(float(np.max([x[k] for x in tt])), 1)}
        for k in ("read_and_resample_sec", "detect_sec", "classify_sec", "total_sec")}

    json.dump({"summary": summary, "per_case": per_case, "per_nodule": per_nodule,
               "unmatched": per_fp},
              open(os.path.join(OUT, "e2e_results.json"), "w"), indent=1)
    # full per-scan reports, the thing a clinician-facing UI would render
    for r in results:
        r.pop("_vol", None); r.pop("_meta", None); r.pop("_aff_pp", None)
    json.dump(results, open(os.path.join(OUT, "scan_reports.json"), "w"), indent=1)

    print("\n=== END-TO-END, compounded, 22 annotated nodules ===")
    for t in THRESHOLDS:
        s = summary[f"end_to_end@{t}"]
        print(f"score>={t}: detected {s['detected']}/{len(per_nodule)} "
              f"({s['detection_rate']:.3f}); END-TO-END CORRECT "
              f"{s['counts']['correct']}/{len(per_nodule)} "
              f"({s['end_to_end_accuracy']:.3f})   "
              f"[B->M/M->B {s['counts']['misclassified_BM']}, "
              f"called Normal {s['counts']['called_Normal']}, "
              f"missed {s['counts']['missed_by_detector']}]")
    print(f"control (GT-centred crop, same code path): {gtc}/{len(per_nodule)} "
          f"({summary['control_gt_centred_accuracy']:.3f})")
    print(f"extraction fidelity vs Phase B PNGs: "
          f"{summary['extraction_fidelity']['png_bit_exact']}/{len(per_nodule)} bit-exact")
    print(f"unmatched candidates @0.5: {summary['unmatched_candidates']['@0.5']}")
    print(f"latency: {json.dumps(summary['latency_sec'])}")
    print(f"peak GPU {summary['peak_gpu_gb']} GB, peak host RSS "
          f"{summary['peak_host_rss_gb']} GB")
    print(f"\nwrote {os.path.join(OUT, 'e2e_results.json')}")


if __name__ == "__main__":
    main()
