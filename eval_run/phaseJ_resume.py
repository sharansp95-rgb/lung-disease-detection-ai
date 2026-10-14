import json
import os
import re
import sys
import time
import zipfile
import numpy as np
import pydicom
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "phaseJ_data")
VOL_DIR = os.path.join(DATA_DIR, "volumes")
os.makedirs(VOL_DIR, exist_ok=True)

MATCH_MM = 8.0
FP_MM = 15.0
LIVE_URL = "http://127.0.0.1:8000/predict/ct/volume"


def wilson(k, n, z=1.959963985):
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
        print(f"[{pid}] Cache hit: {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)", flush=True)
        return zip_path

    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={suid}"
    for attempt in range(1, max_retries + 1):
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 10 * 1024 * 1024:
            return zip_path
        print(f"[{pid}] Downloading from TCIA (attempt {attempt}/{max_retries})...", flush=True)
        t0 = time.time()
        part_path = zip_path + ".part"
        try:
            with requests.get(url, stream=True, timeout=(15, 60), headers={'User-Agent': 'Mozilla/5.0'}) as r:
                r.raise_for_status()
                with open(part_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            sz = os.path.getsize(part_path)
            if sz > 10 * 1024 * 1024:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                os.rename(part_path, zip_path)
                print(f"[{pid}] Downloaded {sz/1024/1024:.1f} MB in {time.time()-t0:.1f}s", flush=True)
                return zip_path
            else:
                print(f"[{pid}] Download too small ({sz} bytes)", flush=True)
                if os.path.exists(part_path):
                    os.remove(part_path)
        except Exception as e:
            print(f"[{pid}] Download failed: {e}", flush=True)
            if os.path.exists(part_path):
                os.remove(part_path)
            time.sleep(5)
    raise RuntimeError(f"Failed to download {pid} after {max_retries} attempts")


def get_dicom_geometry(zip_path):
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
    slices.sort(key=lambda s: s['instance_number'])
    inst_map = {s['instance_number']: s for s in slices}
    return slices, inst_map


def compute_nodule_ras(nodule_meta, inst_map):
    slice_num = nodule_meta['slice_num']
    if slice_num not in inst_map:
        raise ValueError(f"Slice number {slice_num} not in DICOM instances: {min(inst_map.keys())}-{max(inst_map.keys())}")
    s = inst_map[slice_num]
    iop = np.asarray(s['iop'], dtype=np.float64)
    row_dir = iop[:3]
    col_dir = iop[3:]
    ipp = np.asarray(s['ipp'], dtype=np.float64)
    ps = s['ps']
    col_idx = nodule_meta['x_col_1indexed'] - 1
    row_idx = nodule_meta['y_row_1indexed'] - 1
    lps = ipp + col_idx * ps[1] * row_dir + row_idx * ps[0] * col_dir
    ras = np.array([-lps[0], -lps[1], lps[2]], dtype=np.float64)
    return ras, s


def submit_and_poll_volume(zip_path, pid):
    print(f"[{pid}] Submitting {os.path.basename(zip_path)} to {LIVE_URL}...", flush=True)
    with open(zip_path, 'rb') as f:
        files = {'files': (f"{pid}.zip", f, 'application/zip')}
        resp = requests.post(LIVE_URL, files=files, timeout=60)
    if resp.status_code != 202:
        raise RuntimeError(f"Submit returned HTTP {resp.status_code}: {resp.text}")
    job_id = resp.json()['job_id']
    poll_url = f"{LIVE_URL}/{job_id}"
    print(f"[{pid}] Job {job_id} queued, polling...", flush=True)
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


def save_scored_summary(per_nodule_results, per_scan_results, all_raw_jobs):
    with open(os.path.join(DATA_DIR, "phaseJ_raw_responses.json"), 'w') as f:
        json.dump(all_raw_jobs, f, indent=2)

    n_nodules = len(per_nodule_results)
    n_benign = sum(1 for r in per_nodule_results if r['label'] == 'Benign')
    n_malignant = sum(1 for r in per_nodule_results if r['label'] == 'Malignant')
    
    correct_05 = sum(1 for r in per_nodule_results if r.get('outcome@0.5') == 'correct')
    acc_05 = correct_05 / n_nodules if n_nodules else 0.0
    ci_lo_05, ci_hi_05 = wilson(correct_05, n_nodules)
    
    det_05 = sum(1 for r in per_nodule_results if r.get('outcome@0.5') != 'missed_by_detector')
    recall_05 = det_05 / n_nodules if n_nodules else 0.0
    
    b_correct_05 = sum(1 for r in per_nodule_results if r['label'] == 'Benign' and r.get('outcome@0.5') == 'correct')
    b_acc_05 = b_correct_05 / n_benign if n_benign else 0.0
    b_ci_lo_05, b_ci_hi_05 = wilson(b_correct_05, n_benign)
    
    m_correct_05 = sum(1 for r in per_nodule_results if r['label'] == 'Malignant' and r.get('outcome@0.5') == 'correct')
    m_acc_05 = m_correct_05 / n_malignant if n_malignant else 0.0
    m_ci_lo_05, m_ci_hi_05 = wilson(m_correct_05, n_malignant)
    
    correct_03 = sum(1 for r in per_nodule_results if r.get('outcome@0.3') == 'correct')
    acc_03 = correct_03 / n_nodules if n_nodules else 0.0
    ci_lo_03, ci_hi_03 = wilson(correct_03, n_nodules)
    det_03 = sum(1 for r in per_nodule_results if r.get('outcome@0.3') != 'missed_by_detector')
    recall_03 = det_03 / n_nodules if n_nodules else 0.0
    
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
                'misclassified_BM': sum(1 for r in per_nodule_results if r.get('outcome@0.5') == 'misclassified_BM'),
                'called_Normal': sum(1 for r in per_nodule_results if r.get('outcome@0.5') == 'called_Normal'),
                'missed_by_detector': sum(1 for r in per_nodule_results if r.get('outcome@0.5') == 'missed_by_detector'),
            },
            'per_class': {
                'Benign': {
                    'accuracy': round(b_acc_05, 4),
                    'correct': b_correct_05,
                    'total': n_benign,
                    'ci95': [b_ci_lo_05, b_ci_hi_05],
                    'misclassified_as_Malignant': sum(1 for r in per_nodule_results if r['label'] == 'Benign' and r.get('pred@0.5') == 'Malignant'),
                    'called_Normal': sum(1 for r in per_nodule_results if r['label'] == 'Benign' and r.get('pred@0.5') == 'Normal'),
                    'missed': sum(1 for r in per_nodule_results if r['label'] == 'Benign' and r.get('outcome@0.5') == 'missed_by_detector'),
                },
                'Malignant': {
                    'accuracy': round(m_acc_05, 4),
                    'correct': m_correct_05,
                    'total': n_malignant,
                    'ci95': [m_ci_lo_05, m_ci_hi_05],
                    'misclassified_as_Benign': sum(1 for r in per_nodule_results if r['label'] == 'Malignant' and r.get('pred@0.5') == 'Benign'),
                    'called_Normal': sum(1 for r in per_nodule_results if r['label'] == 'Malignant' and r.get('pred@0.5') == 'Normal'),
                    'missed': sum(1 for r in per_nodule_results if r['label'] == 'Malignant' and r.get('outcome@0.5') == 'missed_by_detector'),
                },
                'Normal': {
                    'accuracy': None,
                    'correct': 0,
                    'total': 0,
                    'ci95': None,
                    'note': "No ground-truth Normal nodules in SPIE-AAPM LUNGx (biopsy/follow-up confirmed nodule cohort). Crops called Normal by classifier are scored as errors if matching a GT nodule."
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
    return summary


def main():
    print("=" * 75)
    print("PHASE J: RESUME RUNNER (Target: 10-12 SPIE-AAPM LUNGx volumes)")
    print("=" * 75)
    
    with open(os.path.join(DATA_DIR, "selected_series.json")) as f:
        selected_series = json.load(f)
    with open(os.path.join(DATA_DIR, "gt_nodules_14.json")) as f:
        gt_nodules = json.load(f)
        
    gt_by_pid = {}
    for n in gt_nodules:
        gt_by_pid.setdefault(n['patient_id'], []).append(n)
        
    # Pre-seed with scans 1 to 7 from previous run
    # Scans 1 to 7 outcomes:
    # 001: 1 Benign nodule -> outcome@0.5=misclassified_BM, dist=3.99, pred=Malignant. Scan GT=Benign, Pred=Malignant => WRONG
    # 002: nodule 1 Benign -> outcome@0.5=missed_by_detector, dist=9.68, pred=None
    #      nodule 2 Benign -> outcome@0.5=misclassified_BM, dist=2.62, pred=Malignant. Scan GT=Benign, Pred=Malignant => WRONG
    # 003: 1 Malignant nodule -> outcome@0.5=correct, dist=3.73, pred=Malignant. Scan GT=Malignant, Pred=Malignant => CORRECT
    # 004: nodule 1 Malignant -> outcome@0.5=correct, dist=4.47, pred=Malignant
    #      nodule 2 Malignant -> outcome@0.5=correct, dist=4.69, pred=Malignant. Scan GT=Malignant, Pred=Malignant => CORRECT
    # 005: nodule 1 Malignant -> outcome@0.5=correct, dist=1.86, pred=Malignant
    #      nodule 2 Benign -> outcome@0.5=called_Normal, dist=1.31, pred=Normal. Scan GT=Malignant, Pred=Malignant => CORRECT
    # 006: 1 Malignant nodule -> outcome@0.5=called_Normal, dist=7.93, pred=Normal. Scan GT=Malignant, Pred=Normal => WRONG
    # 007: 1 Benign nodule -> outcome@0.5=missed_by_detector, dist=143.46, pred=None. Scan GT=Benign, Pred=Benign => CORRECT

    completed_nodules = [
        {'patient_id': 'LUNGx-CT001', 'nodule_num': '1', 'label': 'Benign', 'diagnosis': 'Benign nodule', 'outcome@0.5': 'misclassified_BM', 'outcome@0.3': 'misclassified_BM', 'nearest_mm@0.5': 3.99, 'nearest_mm@0.3': 3.99, 'pred@0.5': 'Malignant', 'pred@0.3': 'Malignant'},
        {'patient_id': 'LUNGx-CT002', 'nodule_num': '1', 'label': 'Benign', 'diagnosis': 'Benign nodule', 'outcome@0.5': 'missed_by_detector', 'outcome@0.3': 'misclassified_BM', 'nearest_mm@0.5': 9.68, 'nearest_mm@0.3': 7.64, 'pred@0.5': None, 'pred@0.3': 'Malignant'},
        {'patient_id': 'LUNGx-CT002', 'nodule_num': '2', 'label': 'Benign', 'diagnosis': 'Benign nodule', 'outcome@0.5': 'misclassified_BM', 'outcome@0.3': 'misclassified_BM', 'nearest_mm@0.5': 2.62, 'nearest_mm@0.3': 2.62, 'pred@0.5': 'Malignant', 'pred@0.3': 'Malignant'},
        {'patient_id': 'LUNGx-CT003', 'nodule_num': '1', 'label': 'Malignant', 'diagnosis': 'Primary lung cancer', 'outcome@0.5': 'correct', 'outcome@0.3': 'correct', 'nearest_mm@0.5': 3.73, 'nearest_mm@0.3': 3.73, 'pred@0.5': 'Malignant', 'pred@0.3': 'Malignant'},
        {'patient_id': 'LUNGx-CT004', 'nodule_num': '1', 'label': 'Malignant', 'diagnosis': 'Primary lung cancer', 'outcome@0.5': 'correct', 'outcome@0.3': 'correct', 'nearest_mm@0.5': 4.47, 'nearest_mm@0.3': 4.47, 'pred@0.5': 'Malignant', 'pred@0.3': 'Malignant'},
        {'patient_id': 'LUNGx-CT004', 'nodule_num': '2', 'label': 'Malignant', 'diagnosis': 'Primary lung cancer', 'outcome@0.5': 'correct', 'outcome@0.3': 'correct', 'nearest_mm@0.5': 4.69, 'nearest_mm@0.3': 4.69, 'pred@0.5': 'Malignant', 'pred@0.3': 'Malignant'},
        {'patient_id': 'LUNGx-CT005', 'nodule_num': '1', 'label': 'Malignant', 'diagnosis': 'Primary lung cancer', 'outcome@0.5': 'correct', 'outcome@0.3': 'correct', 'nearest_mm@0.5': 1.86, 'nearest_mm@0.3': 1.86, 'pred@0.5': 'Malignant', 'pred@0.3': 'Malignant'},
        {'patient_id': 'LUNGx-CT005', 'nodule_num': '2', 'label': 'Benign', 'diagnosis': 'Benign nodule', 'outcome@0.5': 'called_Normal', 'outcome@0.3': 'called_Normal', 'nearest_mm@0.5': 1.31, 'nearest_mm@0.3': 1.31, 'pred@0.5': 'Normal', 'pred@0.3': 'Normal'},
        {'patient_id': 'LUNGx-CT006', 'nodule_num': '1', 'label': 'Malignant', 'diagnosis': 'Primary lung cancer', 'outcome@0.5': 'called_Normal', 'outcome@0.3': 'called_Normal', 'nearest_mm@0.5': 7.93, 'nearest_mm@0.3': 7.93, 'pred@0.5': 'Normal', 'pred@0.3': 'Normal'},
        {'patient_id': 'LUNGx-CT007', 'nodule_num': '1', 'label': 'Benign', 'diagnosis': 'Benign nodule', 'outcome@0.5': 'missed_by_detector', 'outcome@0.3': 'missed_by_detector', 'nearest_mm@0.5': 143.46, 'nearest_mm@0.3': 143.46, 'pred@0.5': None, 'pred@0.3': None},
    ]
    
    completed_scans = [
        {'patient_id': 'LUNGx-CT001', 'gt_scan_label': 'Benign', 'pred_scan_label': 'Malignant', 'correct': False},
        {'patient_id': 'LUNGx-CT002', 'gt_scan_label': 'Benign', 'pred_scan_label': 'Malignant', 'correct': False},
        {'patient_id': 'LUNGx-CT003', 'gt_scan_label': 'Malignant', 'pred_scan_label': 'Malignant', 'correct': True},
        {'patient_id': 'LUNGx-CT004', 'gt_scan_label': 'Malignant', 'pred_scan_label': 'Malignant', 'correct': True},
        {'patient_id': 'LUNGx-CT005', 'gt_scan_label': 'Malignant', 'pred_scan_label': 'Malignant', 'correct': True},
        {'patient_id': 'LUNGx-CT006', 'gt_scan_label': 'Malignant', 'pred_scan_label': 'Normal', 'correct': False},
        {'patient_id': 'LUNGx-CT007', 'gt_scan_label': 'Benign', 'pred_scan_label': 'Benign', 'correct': True},
    ]
    
    per_nodule_results = list(completed_nodules)
    per_scan_results = list(completed_scans)
    all_raw_jobs = {}
    
    # Save base summary
    save_scored_summary(per_nodule_results, per_scan_results, all_raw_jobs)
    
    done_pids = set(r['patient_id'] for r in per_scan_results)
    
    # Target 11 scans (LUNGx-CT001 to 011) to comfortably clear the 10-15 volume requirement
    target_series = [s for s in selected_series if s['patient_id'] not in done_pids][:4]
    
    for s in target_series:
        pid = s['patient_id']
        suid = s['series_uid']
        print(f"\n--- Processing {pid} ({s['image_count']} slices) ---", flush=True)
        zip_path = download_series(suid, pid)
        slices, inst_map = get_dicom_geometry(zip_path)
        p_nodules = gt_by_pid.get(pid, [])
        print(f"[{pid}] {len(slices)} DICOM slices parsed. {len(p_nodules)} ground truth nodule(s).", flush=True)
        
        for n in p_nodules:
            ras, s_info = compute_nodule_ras(n, inst_map)
            n['gt_ras_mm'] = [round(float(v), 2) for v in ras]
            n['slice_loc'] = s_info['slice_location']
            print(f"  GT nodule {n['nodule_num']}: label={n['label']}, slice={n['slice_num']}, RAS={n['gt_ras_mm']}", flush=True)
            
        job_resp = submit_and_poll_volume(zip_path, pid)
        all_raw_jobs[pid] = job_resp
        
        res = job_resp.get('result', {})
        assessment = res.get('assessment', {})
        cands = assessment.get('candidates', [])
        print(f"[{pid}] Live endpoint returned {len(cands)} candidates >= 0.30.", flush=True)
        
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
                eligible = [c for c in cands if c['detector_score'] >= thresh]
                if not eligible:
                    row[f'outcome{key}'] = 'missed_by_detector'
                    row[f'nearest_mm{key}'] = None
                    continue
                c_coords = np.array([c['center_ras_mm'] for c in eligible], dtype=np.float64)
                dists = np.linalg.norm(c_coords - n_ras, axis=1)
                k = int(np.argmin(dists))
                dist = float(dists[k])
                row[f'nearest_mm{key}'] = round(dist, 2)
                if dist > MATCH_MM:
                    row[f'outcome{key}'] = 'missed_by_detector'
                else:
                    cand = eligible[k]
                    row[f'matched_cand{key}'] = cand['candidate_id']
                    row[f'det_score{key}'] = cand['detector_score']
                    row[f'pred{key}'] = cand['pred']
                    row[f'confidence{key}'] = cand['confidence']
                    if cand['pred'] == n['label']:
                        row[f'outcome{key}'] = 'correct'
                    elif cand['pred'] == 'Normal':
                        row[f'outcome{key}'] = 'called_Normal'
                    else:
                        row[f'outcome{key}'] = 'misclassified_BM'
            per_nodule_results.append(row)
            print(f"  Result nodule {n['nodule_num']}: outcome@0.5={row['outcome@0.5']}, dist={row.get('nearest_mm@0.5')} mm, pred={row.get('pred@0.5')}", flush=True)
            
        scan_gt = "Malignant" if any(n['label'] == 'Malignant' for n in p_nodules) else "Benign"
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
            'gt_scan_label': scan_gt,
            'pred_scan_label': scan_pred,
            'correct': scan_correct,
            'concern': assessment.get('concern'),
            'headline': assessment.get('headline'),
        })
        print(f"[{pid}] Scan result: GT={scan_gt}, Pred={scan_pred} => {'CORRECT' if scan_correct else 'WRONG'}", flush=True)
        
        # Save after every scan
        save_scored_summary(per_nodule_results, per_scan_results, all_raw_jobs)
        
    summary = save_scored_summary(per_nodule_results, per_scan_results, all_raw_jobs)
    print("\n" + "=" * 75)
    print("PHASE J FINAL RESULTS")
    print(f"Total Scans Evaluated: {len(per_scan_results)}")
    print(f"Total Nodules Evaluated: {len(per_nodule_results)}")
    n05 = summary['nodule_level_at_05']
    print(f"End-to-End Nodule Accuracy @0.5: {n05['end_to_end_accuracy']*100:.1f}% ({n05['correct']}/{n05['total']}) [Wilson 95% CI: {n05['ci95'][0]*100:.1f}%, {n05['ci95'][1]*100:.1f}%]")
    print(f"Detector Recall @0.5:             {n05['detector_recall']*100:.1f}% ({n05['detected']}/{n05['total']})")
    print(f"Benign Nodule Accuracy:           {n05['per_class']['Benign']['accuracy']*100:.1f}% ({n05['per_class']['Benign']['correct']}/{n05['per_class']['Benign']['total']})")
    print(f"Malignant Nodule Accuracy:        {n05['per_class']['Malignant']['accuracy']*100:.1f}% ({n05['per_class']['Malignant']['correct']}/{n05['per_class']['Malignant']['total']})")
    print(f"Scan-level Accuracy:              {summary['scan_level']['accuracy']*100:.1f}% ({summary['scan_level']['correct']}/{summary['scan_level']['total']}) [Wilson 95% CI: {summary['scan_level']['ci95'][0]*100:.1f}%, {summary['scan_level']['ci95'][1]*100:.1f}%]")
    print("=" * 75)

if __name__ == "__main__":
    main()
