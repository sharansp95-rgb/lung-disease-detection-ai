# The app image: the React frontend and the FastAPI API served from one
# origin on :7860 (backend/serve.py). In production it runs behind Caddy via
# docker-compose.yml (EC2, docs/DEPLOYMENT_EC2.md). Build context is the repo
# root, or the folder scripts/deploy_ec2.sh uploads; .dockerignore keeps it
# to runtime files.
#
#   docker build -t lung-app .
#   docker run -p 7860:7860 -e JWT_SECRET=... lung-app

# ---------------------------------------------------------------- frontend --
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# Same-origin API calls, like the ngrok build (frontend/.env.public.example).
ENV VITE_API_BASE_URL=/api
RUN npm run build

# ----------------------------------------------------------------- runtime --
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch first (the default PyPI wheels bundle CUDA, ~5 GB more).
# requirements.txt pins the same versions, so pip treats them as satisfied.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0 \
 && pip install -r /tmp/requirements.txt

# Unprivileged uid 1000. The app dir, logs/ and /data (SQLite) must be
# theirs; Docker gives a new named volume its mount point's ownership.
RUN useradd -m -u 1000 user \
 && install -d -o user -g user /home/user/app /home/user/app/logs /data
USER user
ENV HOME=/home/user
WORKDIR /home/user/app

# Only what the API imports at runtime: backend (incl. ood_stats), ml/,
# scripts/prepare_xray.py (seg_utils.py), the 6 checkpoints, the built site.
COPY --chown=user backend/ backend/
COPY --chown=user ml/ ml/
COPY --chown=user scripts/prepare_xray.py scripts/prepare_xray.py
COPY --chown=user checkpoints/ checkpoints/
COPY --chown=user --from=frontend /frontend/dist frontend/dist

# No training data in the image, so the start-up reproduction check is off.
# SQLite lives on the /data volume. TRUSTED_PROXY_IPS (the reverse proxy's
# address) is set by docker-compose.yml.
ENV SKIP_REPRO_CHECK=1 \
    DB_PATH=/data/app.db

# Load every model once at build time: verifies all imports and checkpoints,
# and downloads torchxrayvision's pretrained DenseNet weights (fetched when
# XRVClassifier is constructed) into ~/.torchxrayvision, plus matplotlib's
# font cache, so start-up needs no network. Throwaway secret and SQLite file.
RUN cd backend \
 && JWT_SECRET=build-only DB_PATH=/tmp/build.db python -c "import main; main._ensure_models_loaded()" \
 && rm -rf /tmp/build.db ../logs/* \
 && test -n "$(ls "$HOME"/.torchxrayvision/models_data/*.pt)"

WORKDIR /home/user/app/backend
EXPOSE 7860
# --no-proxy-headers: uvicorn must not rewrite the client address from
# X-Forwarded-For itself; rate_limit.py does that against TRUSTED_PROXY_IPS.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "7860", "--no-proxy-headers"]
