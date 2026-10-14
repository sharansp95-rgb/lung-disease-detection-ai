"""Shared helpers for the QA verification run. Read-only w.r.t. the project."""
import json, time, pathlib, threading
import requests

BASE = "http://127.0.0.1:8000"
ROOT = pathlib.Path(r"d:/7th sem/sdp")
OUT = ROOT / "eval_run"
OUT.mkdir(exist_ok=True)

_lock = threading.Lock()
_times = []          # timestamps of recent /predict calls (rate-limit pacing)
PREDICT_LIMIT = 20   # per minute, per docs


def pace():
    """Keep /predict calls under 20/min so the rate limiter doesn't poison results."""
    while True:
        with _lock:
            now = time.time()
            _times[:] = [t for t in _times if now - t < 62]
            if len(_times) < PREDICT_LIMIT - 1:
                _times.append(now)
                return
            wait = 62 - (now - _times[0])
        time.sleep(max(wait, 0.5))


def predict(modality, path, token=None, filename=None, paced=True, raw_bytes=None):
    if paced:
        pace()
    url = f"{BASE}/predict/{modality}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = raw_bytes if raw_bytes is not None else pathlib.Path(path).read_bytes()
    name = filename or (pathlib.Path(path).name if path else "upload.png")
    t0 = time.time()
    r = requests.post(url, files={"file": (name, data, "image/png")}, headers=headers, timeout=180)
    dt = time.time() - t0
    try:
        body = r.json()
    except Exception:
        body = {"_unparseable": r.text[:500]}
    return {"status": r.status_code, "latency_s": round(dt, 3), "body": body,
            "file": str(path) if path else name, "modality": modality}


def slim(res):
    """Drop the base64 heatmap for logging."""
    b = dict(res.get("body") or {})
    hm = b.pop("heatmap_png_base64", None)
    b["_heatmap_len"] = len(hm) if hm else 0
    b["_heatmap_prefix"] = (hm[:40] if hm else None)
    out = dict(res)
    out["body"] = b
    return out


def save(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
