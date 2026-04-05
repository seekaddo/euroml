#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p reports .uv-cache

echo "[compare_strategies] Comparing engine strategies on held-out 2025..."
UV_CACHE_DIR=.uv-cache uv run python prediction_engine.py compare-strategies \
  --train-end 2024-12-31 \
  --test-start 2025-01-01 \
  --test-end 2025-12-31 \
  --mode frozen \
  --strategies baseline multi_history \
  --output reports/strategy_compare_2025.json \
  "$@"
