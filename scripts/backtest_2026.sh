#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"
mkdir -p reports
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_DIR}/.uv-cache}"

echo "[backtest_2026] repo: ${REPO_DIR}"
echo "[backtest_2026] cache: ${UV_CACHE_DIR}"
echo "[backtest_2026] running frozen 2026 backtest"

uv run python prediction_engine.py backtest \
  --train-end 2025-12-31 \
  --test-start 2026-01-01 \
  --test-end 2026-12-31 \
  --mode frozen \
  --output reports/backtest_2026.json \
  "$@"
