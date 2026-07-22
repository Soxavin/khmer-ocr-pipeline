#!/usr/bin/env bash
# Start the UI development environment: pipeline backend + Vite hot-reload frontend.
#
#   ./dev.sh          backend (:8600) + Vite HMR (:5173/app/)  <- normal UI work
#   ./dev.sh build    rebuild frontend/dist so :8600/app serves the real bundle
#   ./dev.sh restart  force a fresh backend  <- REQUIRED after editing webapp/ or
#                     src/khmer_pipeline/, because plain `./dev.sh` reuses the
#                     running process and would silently serve the OLD Python code.
#
# Why HMR instead of `npm run build`: the built bundle is served statically from
# frontend/dist, so every edit would need a rebuild + hard refresh. Vite proxies
# /api to :8600 (see frontend/vite.config.ts), so the dev server talks to the same
# backend and edits appear instantly.
#
# An already-running backend is REUSED, never restarted: it holds the multi-GB
# Surya/Kiri models and the in-memory document registry, so a restart costs a slow
# model reload and drops every uploaded document.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "build" ]; then
  cd frontend && npm run build
  echo "Built. Hard-refresh http://localhost:8600/app"
  exit 0
fi

if [ "${1:-}" = "restart" ]; then
  # Backend code is loaded once at import; reuse would keep serving the OLD module.
  pkill -f "webapp.main" 2>/dev/null || true
  sleep 2
  echo "→ stopped the running backend; starting fresh"
fi

backend_pid=""
if lsof -ti:8600 >/dev/null 2>&1; then
  echo "→ backend already on :8600 — reusing it (models stay loaded)"
  echo "  NOTE: edited webapp/ or src/khmer_pipeline/? run ./dev.sh restart"
else
  echo "→ starting backend on :8600 (first run loads models, give it a moment)"
  uv run python -m webapp.main &
  backend_pid=$!
fi

# Stop only what this script started; a reused backend keeps running.
cleanup() {
  [ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "→ starting Vite on http://localhost:5173/app/  (Ctrl-C stops)"
cd frontend && npm run dev
