#!/usr/bin/env bash
# Deploys the committed code to the EC2 host prepared by server_setup.sh
# and (re)starts docker-compose.yml there. Safe to re-run: the SQLite
# database, logs and TLS certificates live in Docker volumes, which
# redeploys never touch, and an existing .env.production is never replaced.
#
#   SERVER_IP=54.206.x.x SSH_KEY=~/.ssh/lung-key-2 DOMAIN=54-206-x-x.sslip.io scripts/deploy_ec2.sh
#
# Optional: SSH_USER (default ubuntu); COPY_SECRETS=1 to copy GROQ_API_KEY,
# GOOGLE_CLIENT_ID/SECRET and GMAIL_ADDRESS/APP_PASSWORD from backend/.env
# into a NEW .env.production (values are sent over SSH stdin, never echoed).
#
# Code comes from `git archive HEAD`: only committed, tracked files ship,
# never backend/.env or anything else gitignored. The weights (gitignored)
# are copied in explicitly.

set -euo pipefail

: "${SERVER_IP:?set SERVER_IP (the Elastic IP)}"
: "${SSH_KEY:?set SSH_KEY (path to the private key, e.g. ~/.ssh/lung-key-2)}"
: "${DOMAIN:?set DOMAIN (e.g. 13-233-x-x.sslip.io)}"
SSH_USER="${SSH_USER:-ubuntu}"
COPY_SECRETS="${COPY_SECRETS:-0}"
REMOTE_DIR=/opt/lung-app

[[ "$DOMAIN" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "DOMAIN must be a bare hostname" >&2; exit 1; }
[[ "$SERVER_IP" =~ ^[0-9.]+$ ]] || { echo "SERVER_IP must be an IPv4 address" >&2; exit 1; }
[[ -f "$SSH_KEY" ]] || { echo "SSH_KEY not found: $SSH_KEY" >&2; exit 1; }

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
SECRET_KEYS=(GROQ_API_KEY GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET GMAIL_ADDRESS GMAIL_APP_PASSWORD)

SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$SSH_USER@$SERVER_IP")
remote() { "${SSH[@]}" "$@"; }

if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  echo "Note: uncommitted changes are NOT deployed -- only HEAD ($(git -C "$ROOT" rev-parse --short HEAD))." >&2
fi

# ---- assemble ----
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
git -C "$ROOT" archive HEAD \
  Dockerfile .dockerignore docker-compose.yml Caddyfile requirements.txt \
  backend ml scripts/prepare_xray.py frontend \
  | tar -x -C "$STAGE"
rm -rf "$STAGE/backend/tests"
mkdir -p "$STAGE/checkpoints" "$STAGE/backend/ood_stats"
# -p keeps mtimes, so rsync skips unchanged weights on later deploys.
for f in "${CHECKPOINTS[@]}"; do cp -p "$ROOT/checkpoints/$f" "$STAGE/checkpoints/$f"; done
for f in "${OOD_STATS[@]}"; do cp -p "$ROOT/backend/ood_stats/$f" "$STAGE/backend/ood_stats/$f"; done

if find "$STAGE" \( -name '.env' -o -name '*.env' -o -name '.env.production' -o -name '*.local' -o -name '*.db' \) | grep -q .; then
  echo "Refusing to deploy: env/db file found in the assembled folder." >&2
  exit 1
fi
echo "==> Assembled $(find "$STAGE" -type f | wc -l | tr -d ' ') files, $(du -sh "$STAGE" | cut -f1)"

# ---- upload ----
echo "==> rsync to $SSH_USER@$SERVER_IP:$REMOTE_DIR"
rsync -az --delete --exclude=/.env.production --exclude=/.env \
  -e "ssh -i '$SSH_KEY' -o StrictHostKeyChecking=accept-new" \
  "$STAGE/" "$SSH_USER@$SERVER_IP:$REMOTE_DIR/"

# ---- .env.production: create only if missing ----
if [[ "$(remote "test -f $REMOTE_DIR/.env.production && echo yes || echo no")" == "yes" ]]; then
  echo "==> .env.production exists on the server, keeping it"
  if [[ "$COPY_SECRETS" == "1" ]]; then
    echo "    (COPY_SECRETS ignored: edit $REMOTE_DIR/.env.production on the server instead)"
  fi
else
  # Reads backend/.env without sourcing it; values never reach argv or the terminal.
  local_value() {
    local v
    v="$(sed -n -E "s/^(export[[:space:]]+)?$1=//p" "$ROOT/backend/.env" 2>/dev/null | tail -n 1 | tr -d '\r')"
    v="${v#\"}"; v="${v%\"}"; v="${v#\'}"; v="${v%\'}"
    case "$v" in replace_with_*|your_email@example.com) v="" ;; esac
    if [[ "$v" == *"'"* ]]; then echo "skipping $1: contains a single quote" >&2; v=""; fi
    printf '%s' "$v"
  }
  ENV_CONTENT="# Production settings for docker-compose.yml. Never commit this file.
# Edit, then: cd $REMOTE_DIR && sudo docker compose up -d
DOMAIN=$DOMAIN
JWT_SECRET=__GENERATE__

# Optional. Empty = feature off (template reports, no Google sign-in,
# verification/reset links printed to the app log instead of emailed).
"
  copied=""
  for k in "${SECRET_KEYS[@]}"; do
    v=""
    if [[ "$COPY_SECRETS" == "1" ]]; then v="$(local_value "$k")"; fi
    if [[ -n "$v" ]]; then ENV_CONTENT+="$k='$v'"$'\n'; copied+=" $k"; else ENV_CONTENT+="$k="$'\n'; fi
  done
  ENV_CONTENT+="
# Optional: Postgres instead of SQLite on the app-data volume.
# DATABASE_URL=
"
  printf '%s' "$ENV_CONTENT" | remote "set -euo pipefail; cd $REMOTE_DIR; umask 077
    tmp=\$(mktemp .env.production.XXXXXX)
    sed \"s/^JWT_SECRET=__GENERATE__\$/JWT_SECRET=\$(openssl rand -hex 32)/\" > \"\$tmp\"
    mv \"\$tmp\" .env.production"
  unset ENV_CONTENT
  echo "==> Created .env.production (JWT_SECRET generated on the server)"
  if [[ -n "$copied" ]]; then echo "    copied from backend/.env:$copied"; fi
fi

# Compose reads .env by itself, so plain `docker compose ps/logs/...` work
# on the server without --env-file.
remote "cd $REMOTE_DIR && ln -sfn .env.production .env"

# DOMAIN follows this deploy (e.g. a new Elastic IP -> new sslip.io name).
remote "cd $REMOTE_DIR && if grep -q '^DOMAIN=' .env.production; then sed -i 's/^DOMAIN=.*/DOMAIN=$DOMAIN/' .env.production; else echo 'DOMAIN=$DOMAIN' >> .env.production; fi"

# ---- build + start ----
echo "==> docker compose up -d --build (first build takes several minutes)"
remote "cd $REMOTE_DIR && sudo docker compose up -d --build --remove-orphans && sudo docker image prune -f >/dev/null"

# ---- health ----
echo "==> waiting for https://$DOMAIN/api/health"
for _ in $(seq 1 60); do
  if body="$(curl -fsS --max-time 10 "https://$DOMAIN/api/health" 2>/dev/null)"; then
    echo "$body" | python3 -m json.tool
    echo "Deployed: https://$DOMAIN"
    exit 0
  fi
  sleep 5
done
echo "Health check did not pass within 5 minutes. Status on the server:" >&2
remote "cd $REMOTE_DIR && sudo docker compose ps" >&2 || true
echo "Logs: ssh -i $SSH_KEY $SSH_USER@$SERVER_IP 'cd $REMOTE_DIR && sudo docker compose logs --tail 100'" >&2
exit 1
