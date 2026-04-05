#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"
mkdir -p reports
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_DIR}/.uv-cache}"

echo "[predict_next] repo: ${REPO_DIR}"
echo "[predict_next] cache: ${UV_CACHE_DIR}"
echo "[predict_next] generating next-draw prediction"

uv run python prediction_engine.py predict-next \
  --output reports/next_draw_prediction.json \
  "$@"
