#!/usr/bin/env bash
# Warm up the web app's backend before the live demo (macOS).
#
#   bash demo/warmup_backend.sh
#
# 1. starts the FastAPI backend (uvicorn, 127.0.0.1:8000, same as start_public.sh)
#    in the background -- or reuses one that is already answering on that port
# 2. sends one real X-ray to /predict/xray and one real CT to /predict/ct
#    (the FIRST request lazy-loads + re-verifies every model: ~9-13 min on this
#    Mac's CPU; the second one is fast) and prints how long each took
# 3. leaves the server running, so the website answers instantly afterwards
#
# Anonymous requests: nothing is saved to anyone's scan history.
# Stop the server later with:   kill "$(cat logs/warmup_backend.pid)"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT"
PY="$ROOT/backend/.venv/bin/python"
LOG="$ROOT/logs/warmup_backend.log"
PIDFILE="$ROOT/logs/warmup_backend.pid"
XRAY_IMG="$ROOT/test images/x ray/normal/img_189af4.png"
CT_IMG="$ROOT/test images/ct scan/benign/img_159819.jpg"

mkdir -p "$ROOT/logs"
[ -x "$PY" ] || { echo "ERROR: $PY not found -- see demo/run_demo.sh" >&2; exit 1; }

if curl -s -m 5 "$URL/health" >/dev/null 2>&1; then
  echo "[warmup] backend already running on $URL -- reusing it"
elif lsof -Pi ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "ERROR: port $PORT is in use by something that isn't answering /health:" >&2
  lsof -Pi ":$PORT" -sTCP:LISTEN >&2
  exit 1
else
  echo "[warmup] starting backend on $URL (log: logs/warmup_backend.log)"
  cd "$ROOT/backend"
  PYTHONUNBUFFERED=1 nohup "$PY" -m uvicorn main:app --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  cd "$ROOT"
  # keep the Mac from idle-sleeping for as long as the server runs
  caffeinate -ims -w "$(cat "$PIDFILE")" >/dev/null 2>&1 &
  for _ in $(seq 1 60); do
    curl -s -m 2 "$URL/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -s -m 2 "$URL/health" >/dev/null 2>&1 || { echo "ERROR: backend did not come up, see $LOG" >&2; exit 1; }
  echo "[warmup] backend is up (pid $(cat "$PIDFILE"))"
fi

predict() {  # predict <xray|ct> <image path> <mime>
  local modality=$1 img=$2 mime=$3 out start end
  out="$(mktemp)"
  echo "[warmup] POST /predict/$modality  $(basename "$img")  ..."
  start=$(date +%s)
  code=$(curl -s -o "$out" -w '%{http_code}' -m 1800 \
           -F "file=@$img;type=$mime" "$URL/predict/$modality")
  end=$(date +%s)
  if [ "$code" = "200" ]; then
    "$PY" - "$out" <<'PYEOF'
import json, sys
b = json.load(open(sys.argv[1]))
print(f"[warmup]   -> {b['predicted_class']} {b['confidence']:.1%} (report: {b['report_source']})")
PYEOF
  else
    echo "[warmup]   -> HTTP $code: $(head -c 300 "$out")"
  fi
  echo "[warmup]   took $((end - start)) s"
  rm -f "$out"
  [ "$code" = "200" ]
}

T0=$(date +%s)
predict xray "$XRAY_IMG" image/png
predict ct "$CT_IMG" image/jpeg
echo "[warmup] done in $(( $(date +%s) - T0 )) s total -- models are loaded; the website will answer instantly."
echo "[warmup] health: $(curl -s "$URL/health")"
echo "[warmup] server left running on $URL.  Stop it with:  kill \"\$(cat logs/warmup_backend.pid)\""
