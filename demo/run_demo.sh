#!/usr/bin/env bash
# Demo helper for Review 2 (macOS). Prints the exact commands for each
# terminal -- it does NOT run any model or audit itself.
#
# Usage (from anywhere):
#   source demo/run_demo.sh        # activates the venv in THIS shell + prints commands
#   source demo/run_demo.sh 3      # same, but only prints Terminal 3 (also: 1, 2, web)
#   bash demo/run_demo.sh          # just prints (activation won't persist)

_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
_DEMO_VENV="$_DEMO_ROOT/backend/.venv"

if [ ! -x "$_DEMO_VENV/bin/python" ]; then
  echo "ERROR: $_DEMO_VENV/bin/python not found or not executable." >&2
  echo "Recreate it with:  cd \"$_DEMO_ROOT\" && python3 -m venv .venv-mac && .venv-mac/bin/pip install -r requirements.txt torchxrayvision==1.5.4" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$_DEMO_VENV/bin/activate"
echo "Activated: $(python --version) at $_DEMO_VENV"
echo "Project root: $_DEMO_ROOT"
echo

_t1() {
cat <<EOF
==================== TERMINAL 1 -- data audits (~2 min total) ====================
cd "$_DEMO_ROOT/scripts"
python ct_leakage_audit.py        # ~1 min 50 s  -> "clusters split across train/val/test: 0 / 111"
python tb_artifact_detect.py      # ~20 s        -> "Tuberculosis: 372/700 flagged"

EOF
}

_t2() {
cat <<EOF
==================== TERMINAL 2 -- test-set evaluation ====================
cd "$_DEMO_ROOT/scripts"
python test_new_xray_checkpoint.py   # accuracy line after ~1.5 min, whole script ~8 min -> "TEST accuracy: 0.9657"
cd "$_DEMO_ROOT/ml"
python evaluate.py --data_dir ../data/ct_scan --checkpoint ../checkpoints/ct_best_model_phase35_v2.pt --output_dir ../eval_outputs/ct_scan_phase35_v2 --run_name ct_phase35_v2   # ~12 s -> "TEST accuracy: 0.9806"

EOF
}

_t3() {
cat <<EOF
==================== TERMINAL 3 -- CLI inference (models load ONCE: ~9 min on CPU, then ~1-2 s per image) ====================
cd "$_DEMO_ROOT/scripts"
# RECOMMENDED: all six demo images in one run (list + comments in demo/demo_batch.txt)
caffeinate -ims python predict_cli.py --batch ../demo/demo_batch.txt

# single image (each = its own ~9 min startup):
python predict_cli.py --image "../test images/x ray/covid/img_19570d.png" --modality xray --save-heatmap ../demo/heatmap_covid.png
python predict_cli.py --image "../test images/ct scan/benign/img_159819.jpg" --modality xray    # -> REJECTED on purpose

Tip: to keep a copy of the output for a fallback, append   2>&1 | tee ../demo/out_batch.txt

EOF
}

_web() {
cat <<EOF
==================== WEBSITE -- warm the backend up before the demo ====================
bash "$_DEMO_ROOT/demo/warmup_backend.sh"   # starts backend on :8000, sends 1 X-ray + 1 CT (~9 min first time), leaves it running
# then start the frontend as usual; stop the backend later with:  kill "\$(cat "$_DEMO_ROOT/logs/warmup_backend.pid")"

EOF
}

case "${1:-all}" in
  1) _t1 ;;
  2) _t2 ;;
  3) _t3 ;;
  web) _web ;;
  *) _t1; _t2; _t3; _web ;;
esac

unset -f _t1 _t2 _t3 _web
unset _DEMO_VENV
