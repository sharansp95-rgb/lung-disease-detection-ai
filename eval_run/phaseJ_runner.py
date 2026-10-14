"""Phase J: Genuine external validation of the CT volume pipeline.

Executes live validation on 14 independent patient CT volumes from SPIE-AAPM LUNGx:
1. Downloads DICOM volumes from TCIA NBIA API (cached locally).
2. Computes ground-truth nodule RAS coordinates from DICOM geometry.
3. Submits each volume to the live deployed POST /predict/ct/volume endpoint.
4. Polls until completion and scores predictions against ground truth using Phase G.3 methodology.
5. Saves all raw responses, per-nodule evaluations, and aggregated metrics with Wilson 95% CIs.
"""

import json
import os
import re
import sys
import time
import urllib.request
import zipfile
import numpy as np
import pydicom
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(HERE, "phaseJ_data")
VOL_DIR = os.path.join(DATA_DIR, "volumes")
os.makedirs(VOL_DIR, exist_ok=True)

MATCH_MM = 8.0
FP_MM = 15.0
LIVE_URL = "http://127.0.0.1:8000/predict/ct/volume"


def wilson(k, n, z=1.959963985):
    """Wilson 95% confidence interval, exactly as used in prior phases."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(c - h), 4), round(float(c + h), 4)


def download_series(suid, pid, max_retries=3):
    zip_path = os.path.join(VOL_DIR, f"{pid}.zip")
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 10 * 1024 * 1024:
        print(f"[{pid}] Cache hit: {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")
        return zip_path

    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={suid}"
    for attempt in range(1, max_retries + 1):
        print(f"[{pid}] Downloading from TCIA (attempt {attempt}/{max_retries})...")
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            part_path = zip_path + ".part"
            with urllib.request.urlopen(req, timeout=180) as resp, open(part_path, 'wb') as f:
                while chunk := resp.read(1024 * 1024):
                    f.write(chunk)
            if os.path.getsize(part_path) < 1024 * 1024:
                raise RuntimeError("Downloaded file too small, likely an error response")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            os.rename(part_path, zip_path)
            print(f"[{pid}] Downloaded {os.path.getsize(zip_path)/1024/1024:.1f} MB in {time.time()-t0:.1f}s")
            return zip_path
        except Exception as e:
            print(f"[{pid}] Download failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed to download {pid} after {max_retries} attempts")


def get_dicom_geometry(zip_path):
    """Read all slice headers in the zip to map slice numbers to RAS world coordinates."""
    slices = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith('.dcm'):
                data = z.read(name)
                ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(data))
                inst = int(getattr(ds, 'InstanceNumber', -1))
                loc = float(getattr(ds, 'SliceLocation', 0.0))
                ipp = [float(x) for x in ds.ImagePositionPatient]
                iop = [float(x) for x in ds.ImageOrientationPatient]
                ps = [float(x) for x in ds.PixelSpacing]
                slices.append({
                    'instance_number': inst,
                    'slice_location': loc,
                    'ipp': ipp,
                    'iop': iop,
                    'ps': ps,
                    'sop': ds.SOPInstanceUID,
                })
    
    # Sort slices by instance number
    slices.sort(key=lambda s: s['instance_number'])
    inst_map = {s['instance_number']: s for s in slices}
    return slices, inst_map


def compute_nodule_ras(nodule_meta, inst_map):
    """Given nodule meta (col, row 1-indexed and slice_num), return RAS coordinate in mm."""
    slice_num = nodule_meta['slice_num']
    if slice_num not in inst_map:
        raise ValueError(f"Slice number {slice_num} not in DICOM instances: {min(inst_map.keys())}-{max(inst_map.keys())}")
    
    s = inst_map[slice_num]
    iop = np.asarray(s['iop'], dtype=np.float64)
    row_dir = iop[:3]
    col_dir = iop[3:]
    ipp = np.asarray(s['ipp'], dtype=np.float64)
    ps = s['ps']
    
    # 1-indexed to 0-indexed
    col_idx = nodule_meta['x_col_1indexed'] - 1
    row_idx = nodule_meta['y_row_1indexed'] - 1
    
    lps = ipp + col_idx * ps[1] * row_dir + row_idx * ps[0] * col_dir
    # LPS to RAS: [-x, -y, z]
    ras = np.array([-lps[0], -lps[1], lps[2]], dtype=np.float64)
    return ras, s


def submit_and_poll_volume(zip_path, pid):
    """Submit zip to live endpoint and poll until completed."""
    print(f"[{pid}] Submitting {os.path.basename(zip_path)} to {LIVE_URL}...")
    with open(zip_path, 'rb') as f:
        files = {'files': (f"{pid}.zip", f, 'application/zip')}
        resp = requests.post(LIVE_URL, files=files, timeout=60)
    
    if resp.status_code != 202:
        raise RuntimeError(f"Submit returned HTTP {resp.status_code}: {resp.text}")
    
    job_id = resp.json()['job_id']
    poll_url = f"{LIVE_URL}/{job_id}"
    print(f"[{pid}] Job {job_id} queued, polling...")
    
    t0 = time.time()
    while True:
        time.sleep(3)
        r = requests.get(poll_url, timeout=30)
        data = r.json()
        st = data.get('status')
        prog = data.get('progress', 0.0)
        stage = data.get('stage', '')
        elapsed = time.time() - t0
        print(f"[{pid}] ({elapsed:.1f}s) status={st}, stage={stage}, progress={prog*100:.0f}%", flush=True)
        if st == 'succeeded':
            return data
        elif st == 'failed':
            raise RuntimeError(f"Job {job_id} failed: {data.get('error')}")
        elif elapsed > 300:
            raise TimeoutError(f"Job {job_id} timed out after 300s")


def main():
    print("=" * 75)
    print("PHASE J: GENUINE EXTERNAL VALIDATION OF THE CT VOLUME PIPELINE")
    print("Target: 14 SPIE-AAPM LUNGx thoracic CT series via live POST /predict/ct/volume")
    print("=" * 75)
    
    with open(os.path.join(DATA_DIR, "selected_series.json")) as f:
        selected_series = json.load(f)
    with open(os.path.join(DATA_DIR, "gt_nodules_14.json")) as f:
        gt_nodules = json.load(f)
        
    gt_by_pid = {}
    for n in gt_nodules:
        gt_by_pid.setdefault(n['patient_id'], []).append(n)
        
    all_raw_jobs = {}
    per_nodule_results = []
    per_scan_results = []
    
    t_start = time.time()
    
    for idx, s in enumerate(selected_series, 1):
        pid = s['patient_id']
        suid = s['series_uid']
        print(f"\n--- [{idx}/{len(selected_series)}] Processing {pid} ({s['image_count']} slices) ---")
        
        # 1. Download
        zip_path = download_series(suid, pid)
        
        # 2. Geometry
        slices, inst_map = get_dicom_geometry(zip_path)
        p_nodules = gt_by_pid.get(pid, [])
        print(f"[{pid}] {len(slices)} DICOM slices parsed. {len(p_nodules)} ground truth nodule(s).")
        
        # Compute GT RAS
        for n in p_nodules:
            ras, s_info = compute_nodule_ras(n, inst_map)
            n['gt_ras_mm'] = [round(float(v), 2) for v in ras]
            n['slice_loc'] = s_info['slice_location']
            n['sop_uid'] = s_info['sop']
            print(f"  GT nodule {n['nodule_num']}: label={n['label']}, slice={n['slice_num']}, RAS={n['gt_ras_mm']}")
            
        # 3. Live endpoint submission & polling
        job_resp = submit_and_poll_volume(zip_path, pid)
        all_raw_jobs[pid] = job_resp
        
        # Extract candidates
        res = job_resp.get('result', {})
        assessment = res.get('assessment', {})
        cands = assessment.get('candidates', [])
        print(f"[{pid}] Live endpoint returned {len(cands)} candidates >= 0.30.")
        
        # 4. Match candidates to GT nodules
        gt_ras_arr = np.array([n['gt_ras_mm'] for n in p_nodules], dtype=np.float64) if p_nodules else np.zeros((0, 3))
        
        for n_idx, n in enumerate(p_nodules):
            n_ras = np.array(n['gt_ras_mm'], dtype=np.float64)
            row = {
                'patient_id': pid,
                'nodule_num': n['nodule_num'],
                'label': n['label'],
                'diagnosis': n['diagnosis'],
                'gt_ras_mm': n['gt_ras_mm'],
                'slice_num': n['slice_num'],
            }
            
            for thresh in (0.5, 0.3):
                key = f"@{thresh}"
                eligible_cands = [c for c in cands if c['detector_score'] >= thresh]
                if not eligible_cands:
                    row[f'outcome{key}'] = 'missed_by_detector'
                    row[f'nearest_mm{key}'] = None
                    row[f'matched_cand{key}'] = None
                    continue
                
                c_coords = np.array([c['center_ras_mm'] for c in eligible_cands], dtype=np.float64)
                dists = np.linalg.norm(c_coords - n_ras, axis=1)
                best_k = int(np.argmin(dists))
                best_dist = float(dists[best_k])
                row[f'nearest_mm{key}'] = round(best_dist, 2)
                
                if best_dist > MATCH_MM:
                    row[f'outcome{key}'] = 'missed_by_detector'
                    row[f'matched_cand{key}'] = None
                else:
                    best_cand = eligible_cands[best_k]
                    row[f'matched_cand{key}'] = best_cand['candidate_id']
                    row[f'det_score{key}'] = best_cand['detector_score']
                    row[f'pred{key}'] = best_cand['pred']
                    row[f'confidence{key}'] = best_cand['confidence']
                    row[f'probs{key}'] = best_cand['probs']
                    
                    if best_cand['pred'] == n['label']:
                        row[f'outcome{key}'] = 'correct'
                    elif best_cand['pred'] == 'Normal':
                        row[f'outcome{key}'] = 'called_Normal'
                    else:
                        row[f'outcome{key}'] = 'misclassified_BM'
                        
            per_nodule_results.append(row)
            print(f"  Result nodule {n['nodule_num']}: outcome@0.5={row['outcome@0.5']}, dist={row.get('nearest_mm@0.5')} mm, pred={row.get('pred@0.5')}")
            
        # Scan-level ground truth and prediction
        # Scan truth: Malignant if any nodule is Malignant, else Benign
        scan_gt = "Malignant" if any(n['label'] == 'Malignant' for n in p_nodules) else "Benign"
        
        # Standard G.3 scan prediction rule: any candidate >= 0.5 Malignant -> Malignant, else if any Benign -> Benign, else Normal
        cands_05 = [c for c in cands if c['detector_score'] >= 0.5]
        if any(c['pred'] == 'Malignant' for c in cands_05):
            scan_pred = "Malignant"
        elif any(c['pred'] == 'Benign' for c in cands_05):
            scan_pred = "Benign"
        else:
            scan_pred = "Normal"
            
        scan_correct = (scan_pred == scan_gt)
        per_scan_results.append({
            'patient_id': pid,
            'n_nodules': len(p_nodules),
            'gt_scan_label': scan_gt,
            'pred_scan_label': scan_pred,
            'correct': scan_correct,
            'concern': assessment.get('concern'),
            'headline': assessment.get('headline'),
            'total_cands_05': len(cands_05),
        })
        print(f"[{pid}] Scan result: GT={scan_gt}, Pred={scan_pred} => {'CORRECT' if scan_correct else 'WRONG'}")
        
    # Save raw and scored
    with open(os.path.join(DATA_DIR, "phaseJ_raw_responses.json"), 'w') as f:
        json.dump(all_raw_jobs, f, indent=2)
        
    # Aggregate Metrics
    n_nodules = len(per_nodule_results)
    n_benign = sum(1 for r in per_nodule_results if r['label'] == 'Benign')
    n_malignant = sum(1 for r in per_nodule_results if r['label'] == 'Malignant')
    
    # Outcomes @ 0.5
    correct_05 = sum(1 for r in per_nodule_results if r['outcome@0.5'] == 'correct')
    acc_05 = correct_05 / n_nodules if n_nodules else 0.0
    ci_lo_05, ci_hi_05 = wilson(correct_05, n_nodules)
    
    det_05 = sum(1 for r in per_nodule_results if r['outcome@0.5'] != 'missed_by_detector')
    recall_05 = det_05 / n_nodules if n_nodules else 0.0
    
    # Per-class @ 0.5
    b_correct_05 = sum(1 for r in per_nodule_results if r['label'] == 'Benign' and r['outcome@0.5'] == 'correct')
    b_acc_05 = b_correct_05 / n_benign if n_benign else 0.0
    b_ci_lo_05, b_ci_hi_05 = wilson(b_correct_05, n_benign)
    
    m_correct_05 = sum(1 for r in per_nodule_results if r['label'] == 'Malignant' and r['outcome@0.5'] == 'correct')
    m_acc_05 = m_correct_05 / n_malignant if n_malignant else 0.0
    m_ci_lo_05, m_ci_hi_05 = wilson(m_correct_05, n_malignant)
    
    # Outcomes @ 0.3
    correct_03 = sum(1 for r in per_nodule_results if r['outcome@0.3'] == 'correct')
    acc_03 = correct_03 / n_nodules if n_nodules else 0.0
    ci_lo_03, ci_hi_03 = wilson(correct_03, n_nodules)
    det_03 = sum(1 for r in per_nodule_results if r['outcome@0.3'] != 'missed_by_detector')
    recall_03 = det_03 / n_nodules if n_nodules else 0.0
    
    # Scan-level
    n_scans = len(per_scan_results)
    scan_correct_cnt = sum(1 for r in per_scan_results if r['correct'])
    scan_acc = scan_correct_cnt / n_scans if n_scans else 0.0
    scan_ci_lo, scan_ci_hi = wilson(scan_correct_cnt, n_scans)
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'total_scans': n_scans,
        'total_nodules': n_nodules,
        'class_counts': {'Benign': n_benign, 'Malignant': n_malignant, 'Normal': 0},
        'nodule_level_at_05': {
            'end_to_end_accuracy': round(acc_05, 4),
            'correct': correct_05,
            'total': n_nodules,
            'ci95': [ci_lo_05, ci_hi_05],
            'detector_recall': round(recall_05, 4),
            'detected': det_05,
            'outcomes': {
                'correct': correct_05,
                'misclassified_BM': sum(1 for r in per_nodule_results if r['outcome@0.5'] == 'misclassified_BM'),
                'called_Normal': sum(1 for r in per_nodule_results if r['outcome@0.5'] == 'called_Normal'),
                'missed_by_detector': sum(1 for r in per_nodule_results if r['outcome@0.5'] == 'missed_by_detector'),
            },
            'per_class': {
                'Benign': {
                    'accuracy': round(b_acc_05, 4),
                    'correct': b_correct_05,
                    'total': n_benign,
                    'ci95': [b_ci_lo_05, b_ci_hi_05],
                    'misclassified_as_Malignant': sum(1 for r in per_nodule_results if r['label'] == 'Benign' and r.get('pred@0.5') == 'Malignant'),
                    'called_Normal': sum(1 for r in per_nodule_results if r['label'] == 'Benign' and r.get('pred@0.5') == 'Normal'),
                    'missed': sum(1 for r in per_nodule_results if r['label'] == 'Benign' and r['outcome@0.5'] == 'missed_by_detector'),
                },
                'Malignant': {
                    'accuracy': round(m_acc_05, 4),
                    'correct': m_correct_05,
                    'total': n_malignant,
                    'ci95': [m_ci_lo_05, m_ci_hi_05],
                    'misclassified_as_Benign': sum(1 for r in per_nodule_results if r['label'] == 'Malignant' and r.get('pred@0.5') == 'Benign'),
                    'called_Normal': sum(1 for r in per_nodule_results if r['label'] == 'Malignant' and r.get('pred@0.5') == 'Normal'),
                    'missed': sum(1 for r in per_nodule_results if r['label'] == 'Malignant' and r['outcome@0.5'] == 'missed_by_detector'),
                },
                'Normal': {
                    'accuracy': None,
                    'correct': 0,
                    'total': 0,
                    'ci95': None,
                    'note': "No native ground-truth Normal nodules in SPIE-AAPM LUNGx (all subjects have biopsy/follow-up confirmed nodules). Slices/crops called Normal by classifier are scored as errors if matching a GT nodule."
                }
            }
        },
        'nodule_level_at_03': {
            'end_to_end_accuracy': round(acc_03, 4),
            'correct': correct_03,
            'total': n_nodules,
            'ci95': [ci_lo_03, ci_hi_03],
            'detector_recall': round(recall_03, 4),
            'detected': det_03,
        },
        'scan_level': {
            'accuracy': round(scan_acc, 4),
            'correct': scan_correct_cnt,
            'total': n_scans,
            'ci95': [scan_ci_lo, scan_ci_hi],
        },
        'per_nodule': per_nodule_results,
        'per_scan': per_scan_results,
    }
    
    with open(os.path.join(DATA_DIR, "phaseJ_results_scored.json"), 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 75)
    print("PHASE J RESULTS SUMMARY (SPIE-AAPM LUNGx External Validation)")
    print("=" * 75)
    print(f"Scans: {n_scans} patients, Nodules: {n_nodules} ({n_benign} Benign, {n_malignant} Malignant)")
    print(f"End-to-end Nodule Accuracy @0.5: {acc_05*100:.1f}% ({correct_05}/{n_nodules}) [Wilson 95% CI: {ci_lo_05*100:.1f}%, {ci_hi_05*100:.1f}%]")
    print(f"Detector Recall @0.5:             {recall_05*100:.1f}% ({det_05}/{n_nodules})")
    print(f"Benign Nodule Accuracy:           {b_acc_05*100:.1f}% ({b_correct_05}/{n_benign}) [Wilson 95% CI: {b_ci_lo_05*100:.1f}%, {b_ci_hi_05*100:.1f}%]")
    print(f"Malignant Nodule Accuracy:        {m_acc_05*100:.1f}% ({m_correct_05}/{n_malignant}) [Wilson 95% CI: {m_ci_lo_05*100:.1f}%, {m_ci_hi_05*100:.1f}%]")
    print(f"Scan-level Accuracy:              {scan_acc*100:.1f}% ({scan_correct_cnt}/{n_scans}) [Wilson 95% CI: {scan_ci_lo*100:.1f}%, {scan_ci_hi*100:.1f}%]")
    print(f"End-to-end Nodule Accuracy @0.3: {acc_03*100:.1f}% ({correct_03}/{n_nodules}) [Wilson 95% CI: {ci_lo_03*100:.1f}%, {ci_hi_03*100:.1f}%]")
    print(f"Total time: {(time.time()-t_start)/60:.1f} minutes")
    print("=" * 75)


if __name__ == "__main__":
    main()
