#!/bin/bash
# macOS launcher for public ngrok hosting
# Usage: ./start_public.sh [--stop]

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
LOG_DIR="$ROOT/logs"
BACKEND_PORT=8000
FRONTEND_PORT=4173

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}==>${NC} $1"
}

log_error() {
    echo -e "${RED}Error:${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

stop_server() {
    log_info "Stopping all services"
    pkill -f "uvicorn" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    pkill -f "ngrok" 2>/dev/null || true
    sleep 1
    echo "Stopped. Public URL is offline."
    exit 0
}

# Handle --stop flag
if [[ "$1" == "--stop" ]]; then
    stop_server
fi

# Preflight checks
log_info "Running preflight checks"
mkdir -p "$LOG_DIR"

# Check required tools
for cmd in python3 node ngrok; do
    if ! command -v $cmd &> /dev/null; then
        log_error "'$cmd' not found. Please install it first."
        exit 1
    fi
done

# Check Python venv
if [[ ! -f "$BACKEND_DIR/.venv/bin/python" ]]; then
    log_error "Python venv not found in $BACKEND_DIR/.venv"
    exit 1
fi

# Check Vite
if [[ ! -f "$FRONTEND_DIR/node_modules/.bin/vite" ]]; then
    log_error "Vite not installed. Run 'npm install' in frontend/"
    exit 1
fi

# Check config files
if [[ ! -f "$FRONTEND_DIR/.env.public.local" ]]; then
    log_error "Missing $FRONTEND_DIR/.env.public.local"
    exit 1
fi

# Get ngrok domain
DOMAIN=$(grep "FRONTEND_ALLOWED_HOSTS" "$FRONTEND_DIR/.env.public.local" | cut -d'=' -f2 | xargs)
if [[ -z "$DOMAIN" ]]; then
    log_error "No FRONTEND_ALLOWED_HOSTS in .env.public.local"
    exit 1
fi
DOMAIN=${DOMAIN#https://}
DOMAIN=${DOMAIN#http://}
DOMAIN=${DOMAIN%%/*}
PUBLIC_URL="https://$DOMAIN"

log_info "Public URL: $PUBLIC_URL"

# Check if ports are in use
if lsof -Pi :$BACKEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_error "Port $BACKEND_PORT is already in use"
    exit 1
fi

if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_error "Port $FRONTEND_PORT is already in use"
    exit 1
fi

# Trap for cleanup
trap "stop_server" SIGINT SIGTERM

# Build frontend
log_info "Building frontend for $PUBLIC_URL"
cd "$FRONTEND_DIR"
node node_modules/.bin/vite build --mode public

# Start backend
log_info "Starting backend on 127.0.0.1:$BACKEND_PORT"
cd "$BACKEND_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -m uvicorn main:app --host 127.0.0.1 --port $BACKEND_PORT --no-proxy-headers > "$LOG_DIR/public_backend.out.log" 2> "$LOG_DIR/public_backend.err.log" &
BACKEND_PID=$!

# Wait for backend health
log_info "Waiting for backend health check (this takes 60-90 seconds)..."
TIMEOUT=600
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s http://127.0.0.1:$BACKEND_PORT/health >/dev/null 2>&1; then
        log_info "Backend is healthy"
        break
    fi
    echo -n "."
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    log_error "Backend did not become healthy within $TIMEOUT seconds"
    log_warn "Check $LOG_DIR/public_backend.err.log for details"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start vite preview
log_info "Starting vite preview on 127.0.0.1:$FRONTEND_PORT"
cd "$FRONTEND_DIR"
node node_modules/.bin/vite preview --mode public --host 127.0.0.1 --port $FRONTEND_PORT --strictPort > "$LOG_DIR/public_preview.out.log" 2> "$LOG_DIR/public_preview.err.log" &
PREVIEW_PID=$!

# Wait for preview
TIMEOUT=60
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s http://127.0.0.1:$FRONTEND_PORT/ >/dev/null 2>&1; then
        log_info "Frontend preview is up"
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    log_error "Frontend preview did not start within $TIMEOUT seconds"
    log_warn "Check $LOG_DIR/public_preview.err.log for details"
    kill $BACKEND_PID $PREVIEW_PID 2>/dev/null || true
    exit 1
fi

# Start ngrok
log_info "Starting ngrok. Press Ctrl+C to stop everything."
echo ""
ngrok http 127.0.0.1:$FRONTEND_PORT --url $PUBLIC_URL

# Cleanup on exit
kill $BACKEND_PID $PREVIEW_PID 2>/dev/null || true
log_info "Stopped. Public URL is offline."
