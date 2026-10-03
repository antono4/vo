#!/usr/bin/env bash
# Start the AI Video Maker server.
set -e
cd "$(dirname "$0")"
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
