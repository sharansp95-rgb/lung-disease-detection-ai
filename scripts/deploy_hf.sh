#!/usr/bin/env bash
# Assembles a clean Hugging Face Space folder (Dockerfile, runtime code,
# model weights, Space README) and pushes it to the Space's git remote,
# with binaries (*.pt, images, fonts) stored via git-lfs.
#
#   HF_USERNAME=me HF_SPACE_NAME=lung-ai HF_TOKEN=hf_... scripts/deploy_hf.sh
#   DRY_RUN=1 HF_USERNAME=me HF_SPACE_NAME=lung-ai scripts/deploy_hf.sh   # assemble + size only
#
# Create the Space first on huggingface.co (SDK: Docker, blank template).
# The push replaces the Space's history with one fresh commit each time.
#
# Code comes from `git archive HEAD`, so only committed, tracked files ship
# (never backend/.env or anything else gitignored); the weights, which are
# gitignored, are copied in explicitly. HF_TOKEN is passed to git through
# an environment-provided credential helper: it is never echoed, never put
# in a remote URL or argv, and never written to disk or the keychain.
# Optional: DEPLOY_DIR=<path> to keep the assembled folder (default: temp dir).

set -euo pipefail

: "${HF_USERNAME:?set HF_USERNAME}"
: "${HF_SPACE_NAME:?set HF_SPACE_NAME}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != "1" ]]; then
  : "${HF_TOKEN:?set HF_TOKEN (a write token from huggingface.co/settings/tokens)}"
  command -v git-lfs >/dev/null || { echo "git-lfs is required (brew install git-lfs)" >&2; exit 1; }
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINTS=(
  xray_txrv_densenet121_best.pt
  ct_best_model_phase35_v2.pt
  modality_gate_best_model_v2.pt
  xray_sanity_best.pt
  ct_sanity_best.pt
  xray_seg_model.pt
)
OOD_STATS=(xray_ood.pt ct_ood.pt)

if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  echo "Note: uncommitted changes are NOT deployed -- only HEAD ($(git -C "$ROOT" rev-parse --short HEAD))." >&2
fi

if [[ -n "${DEPLOY_DIR:-}" ]]; then
  rm -rf "$DEPLOY_DIR"; mkdir -p "$DEPLOY_DIR"; OUT="$DEPLOY_DIR"
else
  OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fi

# ---- code (tracked files only) ----
git -C "$ROOT" archive HEAD \
  Dockerfile .dockerignore requirements.txt backend ml scripts/prepare_xray.py frontend \
  | tar -x -C "$OUT"
rm -rf "$OUT/backend/tests"

# ---- weights (gitignored) ----
mkdir -p "$OUT/checkpoints" "$OUT/backend/ood_stats"
for f in "${CHECKPOINTS[@]}"; do cp "$ROOT/checkpoints/$f" "$OUT/checkpoints/$f"; done
for f in "${OOD_STATS[@]}"; do cp "$ROOT/backend/ood_stats/$f" "$OUT/backend/ood_stats/$f"; done

# ---- Space README (frontmatter is read by HF) ----
cat > "$OUT/README.md" <<'EOF'
---
title: Lung Disease Detection
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Research demo for chest X-ray and CT classification
license: mit
pinned: false
---

Academic research prototype: chest X-ray (COVID-19 / Normal / Pneumonia /
Tuberculosis) and chest CT (Benign / Malignant / Normal) classification with
input gating, OOD checks and Grad-CAM. **Not a medical device -- not for
diagnosis or clinical use.**
EOF

cat > "$OUT/.gitattributes" <<'EOF'
*.pt filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.gif filter=lfs diff=lfs merge=lfs -text
*.webp filter=lfs diff=lfs merge=lfs -text
*.ico filter=lfs diff=lfs merge=lfs -text
*.woff filter=lfs diff=lfs merge=lfs -text
*.woff2 filter=lfs diff=lfs merge=lfs -text
EOF

# Never ship secrets, even if one were committed by mistake.
if find "$OUT" \( -name '.env' -o -name '*.env' -o -name '*.local' -o -name '*.db' \) | grep -q .; then
  echo "Refusing to deploy: env/db file found in the assembled folder:" >&2
  find "$OUT" \( -name '.env' -o -name '*.env' -o -name '*.local' -o -name '*.db' \) >&2
  exit 1
fi

TOTAL="$(find "$OUT" -type f -not -path '*/.git/*' -exec cat {} + | wc -c | tr -d ' ')"
echo "Assembled $(find "$OUT" -type f | wc -l | tr -d ' ') files, $(awk -v b="$TOTAL" 'BEGIN {printf "%.1f MB", b/1048576}') in $OUT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: not pushing."
  [[ -n "${DEPLOY_DIR:-}" ]] || echo "(temp folder removed on exit; set DEPLOY_DIR to keep it)"
  exit 0
fi

# ---- push ----
REMOTE="https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"
cd "$OUT"
git init -q -b main
git lfs install --local >/dev/null
git add -A
git -c user.name="$HF_USERNAME" -c user.email="${HF_USERNAME}@users.noreply.huggingface.co" \
  commit -q -m "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"

# The empty helper clears any configured ones (e.g. osxkeychain, which would
# store the token); ours reads HF_TOKEN from the environment at call time.
# GIT_CONFIG_* is inherited by git-lfs's own credential lookups.
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0=""
export GIT_CONFIG_KEY_1=credential.helper
export GIT_CONFIG_VALUE_1='!f() { test "$1" = get && echo "username=$HF_USERNAME" && echo "password=$HF_TOKEN"; }; f'
export GIT_TERMINAL_PROMPT=0
git push --force "$REMOTE" main:main
echo "Pushed. Build logs: $REMOTE (Logs tab). App: https://${HF_USERNAME}-${HF_SPACE_NAME}.hf.space"
