"""Phase E step 3: exercise the job-submission/polling API against a real volume.

Proves the async shape actually works -- POST returns in milliseconds with a job id
while a ~35 s pipeline runs on a background thread, and GET reports live stage/progress
until the result lands. Uses Starlette's TestClient so no separate uvicorn process is
needed; the app object and threading are the same ones uvicorn would drive.

Usage: python test_api.py [PATIENT_ID]
"""
import json
import os
import sys
import time

from fastapi.testclient import TestClient

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from app import app                                            # noqa: E402

VOLUMES = os.path.abspath(os.path.join(HERE, "..", "phaseD", "_work", "volumes"))


def main():
    pid = sys.argv[1] if len(sys.argv) > 1 else "LIDC-IDRI-0586"
    series = os.path.join(VOLUMES, pid)
    client = TestClient(app)

    print("health:", client.get("/phaseE/health").json())

    t0 = time.time()
    r = client.post("/phaseE/jobs", json={"series_dir": series,
                                          "thresholds": [0.5, 0.3]})
    submit_ms = (time.time() - t0) * 1000
    assert r.status_code == 202, r.text
    job = r.json()
    print(f"POST /phaseE/jobs -> {r.status_code} in {submit_ms:.0f} ms: {job}")

    # bad input is rejected synchronously, before any work is scheduled
    bad = client.post("/phaseE/jobs", json={"series_dir": os.path.join(VOLUMES, "nope")})
    print(f"POST with a bad path -> {bad.status_code} {bad.json()['detail']!r}")
    print(f"GET an unknown job id -> {client.get('/phaseE/jobs/deadbeef').status_code}")

    last = None
    while True:
        s = client.get(job["poll"]).json()
        line = f"{time.time()-t0:6.1f}s  {s['status']:9s} {s['progress']:.2f}  {s['stage']}"
        if line != last:
            print(line, flush=True)
            last = line
        if s["status"] in ("succeeded", "failed"):
            break
        time.sleep(1.0)

    if s["status"] == "failed":
        print("FAILED:", s["error"])
        return 1

    res = s["result"]
    print("\ntimings:", json.dumps(res["timings"]))
    print("resources:", json.dumps(res["resources"]))
    for t in ("0.5", "0.3"):
        rep = res["reports"][t]
        print(f"\n--- scan report @ score>={t} ---")
        print(f"scan call: {rep['scan_call']}  ({rep['n_candidates']} candidates, "
              f"driver {rep['driver']}, counts {rep['counts']})")
        for c in rep["candidates"]:
            print(f"  {c['candidate_id']}  det {c['detector_score']:.3f}  "
                  f"RAS {c['center_ras_mm']}  size {c['box_size_mm']}  -> "
                  f"{c['pred']:9s} B/M/N {c['probs']}")
    out = os.path.join(HERE, "_work", f"api_demo_{pid}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(s, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    print("jobs list:", json.dumps(client.get("/phaseE/jobs").json(), indent=1)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
