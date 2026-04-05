#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-research_outputs/latest}"
SAMPLES="${SAMPLES:-50}"
PREDICTION_STRATEGY="${PREDICTION_STRATEGY:-baseline}"
DEFAULT_STRATEGIES="baseline star_guard1_soft_screen_multi_history hybrid_main_star_focus_soft_guard_screen hybrid_main_hybrid_star_soft_guard_screen"

if (($# > 0)); then
  STRATEGIES=("$@")
else
  read -r -a STRATEGIES <<< "${RESEARCH_STRATEGIES:-$DEFAULT_STRATEGIES}"
fi

mkdir -p "$OUTPUT_DIR"

if [[ -f "$OUTPUT_DIR/engine_state.json" ]]; then
  cp "$OUTPUT_DIR/engine_state.json" .engine.mem.json
fi

echo "Running research cycle into $OUTPUT_DIR"
echo "Strategies: ${STRATEGIES[*]}"
echo "Prediction strategy: $PREDICTION_STRATEGY"
echo "Samples: $SAMPLES"

uv run python prediction_engine.py compare-strategies \
  --train-end 2024-12-31 \
  --test-start 2025-01-01 \
  --test-end 2025-12-31 \
  --mode frozen \
  --samples "$SAMPLES" \
  --strategies "${STRATEGIES[@]}" \
  --output "$OUTPUT_DIR/compare_2025.json"

uv run python prediction_engine.py compare-strategies \
  --train-end 2025-12-31 \
  --test-start 2026-01-01 \
  --test-end 2026-12-31 \
  --mode frozen \
  --samples "$SAMPLES" \
  --strategies "${STRATEGIES[@]}" \
  --output "$OUTPUT_DIR/compare_2026.json"

uv run python prediction_engine.py predict-next \
  --strategy "$PREDICTION_STRATEGY" \
  --samples "$SAMPLES" \
  --output "$OUTPUT_DIR/next_draw_prediction.json"

if [[ -f .engine.mem.json ]]; then
  cp .engine.mem.json "$OUTPUT_DIR/engine_state.json"
fi

python3 scripts/build_research_summary.py \
  --output-dir "$OUTPUT_DIR" \
  --prediction-strategy "$PREDICTION_STRATEGY" \
  --strategies "${STRATEGIES[@]}"
