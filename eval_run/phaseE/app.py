"""Phase E: a minimal, standalone FastAPI app demonstrating the job-submission +
polling shape a ~35 s/scan volume pipeline needs.

THIS IS NOT `backend/main.py` AND IS NOT WIRED INTO IT. Different module, different
app object, different route prefix, its own model instances. `backend/main.py`,
`/predict/ct` and `checkpoints/ct_best_model_phase35_v2.pt` are untouched by this phase;
nothing here is imported by any live-serving file.

Deliberately NOT Celery/Redis/a real broker. `PROJECT_STATUS.md` S8 scopes this project
as local-only, and Phase D measured ~35 s/scan on this GPU -- one background thread per
job and an in-memory dict is the honest size of the problem. What that buys and what it
costs is written down in phaseE_pipeline_build.md rather than papered over: jobs die with
the process, there is no cross-worker sharing, and the dict grows until restart.

Run:  python -m uvicorn app:app --port 8010     (from eval_run/phaseE/)
"""
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import TwoStagePipeline                      # noqa: E402

app = FastAPI(title="Phase E — CT volume nodule pipeline (STANDALONE, not the live app)")

JOBS = {}
_JOBS_LOCK = threading.Lock()
_PIPE = None
_PIPE_LOCK = threading.Lock()
MAX_CONCURRENT = 1          # one 4 GB GPU; queueing beats OOM


def get_pipeline():
    """Both models are loaded once, lazily, and shared by every job. Loading per job
    would re-pay ~5 s and briefly double VRAM."""
    global _PIPE
    with _PIPE_LOCK:
        if _PIPE is None:
            _PIPE = TwoStagePipeline(device=os.environ.get("PHASE_E_DEVICE", "cuda"))
        return _PIPE


class SubmitBody(BaseModel):
    series_dir: str
    thresholds: list[float] = [0.5, 0.3]


def _public(result):
    """Strip the harness-only keys (raw volume, affine, DICOM meta) from the payload."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _worker(job_id, series_dir, thresholds, cleanup_dir=None):
    def progress(msg, frac):
        with _JOBS_LOCK:
            JOBS[job_id]["stage"] = msg
            JOBS[job_id]["progress"] = round(frac, 2)

    try:
        with _JOBS_LOCK:
            JOBS[job_id].update(status="running", started_at=time.time())
        pipe = get_pipeline()
        res = pipe.run(series_dir, thresholds=tuple(thresholds), progress=progress)
        with _JOBS_LOCK:
            JOBS[job_id].update(status="succeeded", progress=1.0, stage="done",
                                finished_at=time.time(), result=_public(res))
    except Exception as e:                                     # noqa: BLE001
        with _JOBS_LOCK:
            JOBS[job_id].update(status="failed", finished_at=time.time(),
                                error=f"{type(e).__name__}: {e}")
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def _submit(series_dir, thresholds, cleanup_dir=None):
    with _JOBS_LOCK:
        running = sum(1 for j in JOBS.values() if j["status"] == "running")
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"job_id": job_id, "status": "queued", "progress": 0.0,
                        "stage": "queued" if running >= MAX_CONCURRENT else "starting",
                        "series_dir": series_dir, "submitted_at": time.time()}
    threading.Thread(target=_worker, args=(job_id, series_dir, thresholds, cleanup_dir),
                     daemon=True).start()
    return job_id


@app.get("/phaseE/health")
def health():
    import torch
    return {"ok": True, "standalone": True,
            "note": "Phase E scratch app; the live /predict/ct is a different process",
            "cuda": torch.cuda.is_available(),
            "models_loaded": _PIPE is not None,
            "jobs": len(JOBS)}


@app.post("/phaseE/jobs", status_code=202)
def submit_path(body: SubmitBody):
    """Submit a DICOM series already on disk. Returns immediately with a job id --
    the pipeline takes ~35 s on this GPU, far outside a request/response budget."""
    if not os.path.isdir(body.series_dir):
        raise HTTPException(400, f"not a directory: {body.series_dir}")
    if not any(f.lower().endswith(".dcm") for f in os.listdir(body.series_dir)):
        raise HTTPException(400, "no .dcm files in that directory")
    job_id = _submit(body.series_dir, body.thresholds)
    return {"job_id": job_id, "status": "queued",
            "poll": f"/phaseE/jobs/{job_id}"}


@app.post("/phaseE/jobs/upload", status_code=202)
async def submit_zip(file: UploadFile = File(...), thresholds: str = "0.5,0.3"):
    """Submit a zipped DICOM series. Flattens the archive into a temp dir, which the
    worker deletes when it finishes."""
    tmp = tempfile.mkdtemp(prefix="phaseE_")
    zpath = os.path.join(tmp, "upload.zip")
    with open(zpath, "wb") as f:
        f.write(await file.read())
    ddir = os.path.join(tmp, "series")
    os.makedirs(ddir, exist_ok=True)
    try:
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if n.lower().endswith(".dcm") and not n.startswith(("/", "..")):
                    with z.open(n) as src, open(
                            os.path.join(ddir, os.path.basename(n)), "wb") as dst:
                        shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(400, "not a readable zip")
    if not os.listdir(ddir):
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(400, "zip contained no .dcm files")
    job_id = _submit(ddir, [float(v) for v in thresholds.split(",")], cleanup_dir=tmp)
    return {"job_id": job_id, "status": "queued", "n_slices": len(os.listdir(ddir)),
            "poll": f"/phaseE/jobs/{job_id}"}


@app.get("/phaseE/jobs/{job_id}")
def poll(job_id: str):
    with _JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job id")
        return dict(j)


@app.get("/phaseE/jobs")
def list_jobs():
    with _JOBS_LOCK:
        return [{k: v for k, v in j.items() if k != "result"} for j in JOBS.values()]
