#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$APP_DIR/.venv/bin/python" "$APP_DIR/app.py" "$@"
